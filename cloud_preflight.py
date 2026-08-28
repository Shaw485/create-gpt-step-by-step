"""Fail-fast CUDA and artifact preflight for the unattended A10 run.

There is intentionally no CPU fallback.  Tests inject a fake backend, while the
command-line path always uses :class:`TorchCudaBackend` and performs an actual
BF16 forward/backward optimizer step on the selected CUDA device.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Protocol, Sequence

import torch

from training_runtime import (
    LoadedConfig,
    RunStateWriter,
    atomic_write_json,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    load_versioned_config,
    redact_sensitive,
    utc_now,
)


GIB = 1024 ** 3


class PreflightError(RuntimeError):
    """A failed gate with an explicit operator action."""

    def __init__(
        self,
        code: str,
        message: str,
        action: str,
        *,
        details: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.action = action
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "action": self.action,
            "details": redact_sensitive(self.details),
        }


@dataclass(frozen=True)
class CudaDeviceSnapshot:
    index: int
    name: str
    total_memory_bytes: int
    used_memory_bytes: int
    compute_capability: tuple[int, int]


@dataclass(frozen=True)
class Bf16ProbeResult:
    forward_finite: bool
    loss_finite: bool
    gradients_finite: bool
    output_dtype: str
    peak_allocated_bytes: int
    peak_reserved_bytes: int


class CudaBackend(Protocol):
    """Small injectable surface that keeps CPU unit tests CUDA-independent."""

    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...

    def select_device(self, index: int) -> None: ...

    def device_snapshot(self, index: int) -> CudaDeviceSnapshot: ...

    def is_bf16_supported(self, index: int) -> bool: ...

    def run_bf16_probe(
        self,
        index: int,
        *,
        matrix_size: int,
        batch_size: int,
    ) -> Bf16ProbeResult: ...


@contextmanager
def _cuda_bf16_autocast() -> Iterator[None]:
    """Support both the current unified AMP API and older PyTorch releases."""
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "autocast"):
        with amp_module.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:  # pragma: no cover - compatibility for older cloud images.
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            yield


class TorchCudaBackend:
    """Production adapter around torch.cuda."""

    def is_available(self) -> bool:
        return bool(torch.cuda.is_available())

    def device_count(self) -> int:
        return int(torch.cuda.device_count())

    def select_device(self, index: int) -> None:
        torch.cuda.set_device(index)

    def device_snapshot(self, index: int) -> CudaDeviceSnapshot:
        properties = torch.cuda.get_device_properties(index)
        torch.cuda.set_device(index)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except TypeError:  # Older torch versions use the currently selected device.
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        return CudaDeviceSnapshot(
            index=index,
            name=str(properties.name),
            total_memory_bytes=int(total_bytes),
            used_memory_bytes=max(0, int(total_bytes) - int(free_bytes)),
            compute_capability=(int(properties.major), int(properties.minor)),
        )

    def is_bf16_supported(self, index: int) -> bool:
        self.select_device(index)
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        if checker is None:
            return False
        try:
            return bool(checker(including_emulation=False))
        except TypeError:
            return bool(checker())

    def run_bf16_probe(
        self,
        index: int,
        *,
        matrix_size: int,
        batch_size: int,
    ) -> Bf16ProbeResult:
        if matrix_size <= 0 or batch_size <= 0:
            raise ValueError("BF16 probe dimensions must be positive")
        self.select_device(index)
        device = torch.device(f"cuda:{index}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(index)
        model = torch.nn.Sequential(
            torch.nn.Linear(matrix_size, matrix_size),
            torch.nn.GELU(),
            torch.nn.Linear(matrix_size, 32),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        inputs = torch.randn(batch_size, matrix_size, device=device)
        targets = torch.randn(batch_size, 32, device=device)
        optimizer.zero_grad(set_to_none=True)
        with _cuda_bf16_autocast():
            output = model(inputs)
            loss = torch.nn.functional.mse_loss(output.float(), targets.float())
        loss.backward()
        forward_finite = bool(torch.isfinite(output.detach()).all().item())
        loss_finite = bool(torch.isfinite(loss.detach()).all().item())
        gradients_finite = all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad.detach()).all().item())
            for parameter in model.parameters()
        )
        optimizer.step()
        torch.cuda.synchronize(index)
        result = Bf16ProbeResult(
            forward_finite=forward_finite,
            loss_finite=loss_finite,
            gradients_finite=gradients_finite,
            output_dtype=str(output.dtype),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(index)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(index)),
        )
        del optimizer, model, inputs, targets, output, loss
        torch.cuda.empty_cache()
        return result


def _artifact_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Normalize the two documented release-manifest artifact shapes."""
    artifacts = manifest.get("artifacts")
    result: dict[str, str] = {}
    if isinstance(artifacts, Mapping):
        for path, value in artifacts.items():
            if isinstance(value, str):
                digest = value
            elif isinstance(value, Mapping):
                digest = value.get("sha256")
            else:
                continue
            if isinstance(digest, str):
                result[str(path)] = digest.lower()
    elif isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes)):
        for value in artifacts:
            if not isinstance(value, Mapping):
                continue
            path = value.get("path")
            digest = value.get("sha256")
            if isinstance(path, str) and isinstance(digest, str):
                result[path] = digest.lower()
    return result


def _matching_manifest_hash(hashes: Mapping[str, str], relative_path: str) -> str | None:
    normalized = Path(relative_path).as_posix()
    candidates = {
        normalized,
        normalized.removeprefix("./"),
        Path(normalized).name,
    }
    for key, digest in hashes.items():
        key_normalized = Path(key).as_posix().removeprefix("./")
        if key_normalized in candidates or key_normalized.endswith(f"/{normalized}"):
            return digest
    return None


def _verify_manifest_sidecar(manifest_path: Path) -> None:
    sidecar = Path(f"{manifest_path}.sha256")
    if not sidecar.is_file():
        raise PreflightError(
            "ARTIFACT_MANIFEST_CHECKSUM_MISSING",
            f"release manifest checksum is missing: {sidecar}",
            "Regenerate the reviewed release manifest and its .sha256 sidecar before renting the GPU.",
        )
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    expected = fields[0].lower() if fields else ""
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise PreflightError(
            "ARTIFACT_MANIFEST_CHECKSUM_INVALID",
            f"release manifest checksum is malformed: {sidecar}",
            "Regenerate the checksum with SHA-256; do not bypass verification.",
        )
    calculated = file_sha256(manifest_path)
    if calculated != expected:
        raise PreflightError(
            "ARTIFACT_MANIFEST_CHECKSUM_MISMATCH",
            f"release manifest changed after review: {manifest_path}",
            "Stop, inspect the changed manifest, approve it again, and regenerate the sidecar.",
            details={"expected": expected, "calculated": calculated},
        )


def verify_release_artifacts(
    project_root: Path,
    data_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Require ready manifests and verify every declared artifact SHA-256."""
    specifications = data_config.get("release_manifests")
    if not isinstance(specifications, list) or not specifications:
        raise PreflightError(
            "ARTIFACT_MANIFESTS_NOT_CONFIGURED",
            "data.release_manifests is empty",
            "List corpus, tokenizer/tensor, and SFT release manifests in the versioned config.",
        )
    reports: list[dict[str, Any]] = []
    for specification in specifications:
        if not isinstance(specification, Mapping):
            raise PreflightError(
                "ARTIFACT_MANIFEST_CONFIG_INVALID",
                "each release manifest configuration must be an object",
                "Fix configs/cloud_a10.json and regenerate its SHA-256 sidecar.",
            )
        relative_manifest = specification.get("path")
        if not isinstance(relative_manifest, str):
            raise PreflightError(
                "ARTIFACT_MANIFEST_CONFIG_INVALID",
                "release manifest path is missing",
                "Add a relative path to every data.release_manifests entry.",
            )
        manifest_path = project_root / relative_manifest
        if not manifest_path.is_file():
            raise PreflightError(
                "ARTIFACT_MANIFEST_MISSING",
                f"reviewed release manifest does not exist: {manifest_path}",
                "Complete the corresponding local preparation/review gate before launching cloud training.",
            )
        _verify_manifest_sidecar(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PreflightError(
                "ARTIFACT_MANIFEST_UNREADABLE",
                f"cannot parse {manifest_path}: {error}",
                "Regenerate the manifest from the preparation script; do not edit it by hand.",
            ) from error
        expected_status = specification.get("required_status", "ready")
        if manifest.get("status") != expected_status:
            raise PreflightError(
                "ARTIFACT_REVIEW_INCOMPLETE",
                f"{manifest_path} status is {manifest.get('status')!r}, expected {expected_status!r}",
                "Resolve all review-queue items and rerun the release step before cloud training.",
            )
        required_files = specification.get("required_files", [])
        if not isinstance(required_files, list) or not required_files:
            raise PreflightError(
                "ARTIFACT_REQUIRED_FILES_EMPTY",
                f"no required files are configured for {manifest_path}",
                "Declare every training artifact that the manifest must authenticate.",
            )
        hashes = _artifact_hashes(manifest)
        verified: list[dict[str, Any]] = []
        for relative_file in required_files:
            if not isinstance(relative_file, str):
                raise PreflightError(
                    "ARTIFACT_REQUIRED_FILE_INVALID",
                    f"non-string required file in {manifest_path}",
                    "Use relative path strings in required_files.",
                )
            file_path = project_root / relative_file
            if not file_path.is_file() or file_path.stat().st_size <= 0:
                raise PreflightError(
                    "ARTIFACT_FILE_MISSING",
                    f"training artifact is missing or empty: {file_path}",
                    "Rerun the corresponding deterministic preparation stage and its validation.",
                )
            expected_hash = _matching_manifest_hash(hashes, relative_file)
            if expected_hash is None:
                raise PreflightError(
                    "ARTIFACT_HASH_MISSING",
                    f"{manifest_path} does not authenticate {relative_file}",
                    "Regenerate the release manifest with path, size, and SHA-256 for every required artifact.",
                )
            calculated_hash = file_sha256(file_path)
            if calculated_hash != expected_hash:
                raise PreflightError(
                    "ARTIFACT_HASH_MISMATCH",
                    f"SHA-256 mismatch for {file_path}",
                    "Do not train. Restore or regenerate the reviewed artifact, then rerun preflight.",
                    details={
                        "expected": expected_hash,
                        "calculated": calculated_hash,
                    },
                )
            verified.append(
                {
                    "path": relative_file,
                    "size_bytes": file_path.stat().st_size,
                    "sha256": calculated_hash,
                }
            )
        reports.append(
            {
                "manifest": relative_manifest,
                "manifest_sha256": file_sha256(manifest_path),
                "status": manifest.get("status"),
                "verified_artifacts": verified,
            }
        )
    return reports


def _tuple_version(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) for item in value)
    ):
        raise PreflightError(
            "HARDWARE_CONFIG_INVALID",
            f"{label} must be a two-integer JSON array",
            "Correct the versioned cloud configuration and regenerate its checksum.",
        )
    return int(value[0]), int(value[1])


def verify_cuda_requirements(
    hardware: Mapping[str, Any],
    backend: CudaBackend,
) -> dict[str, Any]:
    """Require the selected target GPU and prove native BF16 training works."""
    if hardware.get("device") != "cuda":
        raise PreflightError(
            "CPU_FALLBACK_FORBIDDEN",
            f"hardware.device must be 'cuda', got {hardware.get('device')!r}",
            "Set device=cuda. This cloud run must stop instead of silently training on CPU.",
        )
    if not backend.is_available():
        raise PreflightError(
            "CUDA_UNAVAILABLE",
            "torch.cuda.is_available() is false",
            "Select a GPU instance, install a CUDA-compatible PyTorch build, and rerun preflight.",
        )
    available_count = backend.device_count()
    required_count = int(hardware.get("required_gpu_count", 1))
    if available_count < required_count:
        raise PreflightError(
            "CUDA_DEVICE_COUNT_TOO_SMALL",
            f"found {available_count} visible GPU(s), require at least {required_count}",
            "Fix CUDA_VISIBLE_DEVICES or rent the configured GPU count.",
        )
    device_index = int(hardware.get("device_index", 0))
    if device_index < 0 or device_index >= available_count:
        raise PreflightError(
            "CUDA_DEVICE_INDEX_INVALID",
            f"device index {device_index} is outside 0..{available_count - 1}",
            "Correct hardware.device_index or CUDA_VISIBLE_DEVICES.",
        )
    backend.select_device(device_index)
    snapshot = backend.device_snapshot(device_index)
    expected_name = hardware.get("required_gpu_name_regex")
    if not isinstance(expected_name, str) or not re.search(
        expected_name, snapshot.name, flags=re.IGNORECASE
    ):
        raise PreflightError(
            "WRONG_GPU_MODEL",
            f"GPU {snapshot.name!r} does not match required pattern {expected_name!r}",
            "Stop this instance and select the configured target card; do not benchmark on an accidental SKU.",
        )
    min_memory_bytes = float(hardware.get("min_vram_gib", 0.0)) * GIB
    if snapshot.total_memory_bytes < min_memory_bytes:
        raise PreflightError(
            "GPU_MEMORY_TOO_SMALL",
            f"GPU has {snapshot.total_memory_bytes / GIB:.2f} GiB, require at least {min_memory_bytes / GIB:.2f} GiB",
            "Rent a card with enough VRAM or create a separately reviewed lower-memory config.",
        )
    minimum_capability = _tuple_version(
        hardware.get("min_compute_capability", [0, 0]),
        "hardware.min_compute_capability",
    )
    if snapshot.compute_capability < minimum_capability:
        raise PreflightError(
            "COMPUTE_CAPABILITY_TOO_OLD",
            f"compute capability {snapshot.compute_capability} is below {minimum_capability}",
            "Select the target A10-or-newer GPU image/instance.",
        )
    used_fraction = snapshot.used_memory_bytes / max(1, snapshot.total_memory_bytes)
    max_existing_fraction = float(hardware.get("max_existing_vram_fraction", 0.20))
    if used_fraction > max_existing_fraction:
        raise PreflightError(
            "GPU_ALREADY_BUSY",
            f"{used_fraction:.1%} of GPU VRAM is already in use, limit is {max_existing_fraction:.1%}",
            "Stop unrelated GPU processes or launch a clean instance before the smoke test.",
        )
    estimated_peak_bytes = float(hardware.get("estimated_training_peak_gib", 0.0)) * GIB
    max_training_fraction = float(hardware.get("max_training_vram_fraction", 0.80))
    estimated_fraction = estimated_peak_bytes / max(1, snapshot.total_memory_bytes)
    if estimated_fraction > max_training_fraction:
        raise PreflightError(
            "ESTIMATED_TRAINING_MEMORY_TOO_HIGH",
            f"estimated training peak is {estimated_fraction:.1%} of VRAM, limit is {max_training_fraction:.1%}",
            "Reduce micro-batch size and add gradient accumulation, or rent a larger GPU, then update and re-sign the config.",
        )
    if bool(hardware.get("require_native_bf16", True)) and not backend.is_bf16_supported(
        device_index
    ):
        raise PreflightError(
            "BF16_UNSUPPORTED",
            "native BF16 is not supported by the selected CUDA stack/device",
            "Use the A10 CUDA image with a compatible PyTorch build; do not silently switch precision.",
        )
    probe_config = hardware.get("bf16_probe", {})
    if not isinstance(probe_config, Mapping):
        raise PreflightError(
            "BF16_PROBE_CONFIG_INVALID",
            "hardware.bf16_probe must be an object",
            "Correct the versioned configuration and regenerate its SHA-256 sidecar.",
        )
    probe = backend.run_bf16_probe(
        device_index,
        matrix_size=int(probe_config.get("matrix_size", 1024)),
        batch_size=int(probe_config.get("batch_size", 64)),
    )
    if not (probe.forward_finite and probe.loss_finite and probe.gradients_finite):
        raise PreflightError(
            "BF16_PROBE_NONFINITE",
            "BF16 forward/backward produced NaN or Inf",
            "Stop. Verify CUDA/PyTorch compatibility and test the GPU for instability before training.",
            details=asdict(probe),
        )
    if probe.output_dtype != "torch.bfloat16":
        raise PreflightError(
            "BF16_PROBE_WRONG_DTYPE",
            f"autocast output dtype is {probe.output_dtype}, expected torch.bfloat16",
            "Install a native-BF16 PyTorch/CUDA image; do not continue with an implicit fallback.",
        )
    probe_fraction = probe.peak_reserved_bytes / max(1, snapshot.total_memory_bytes)
    max_probe_fraction = float(hardware.get("max_probe_vram_fraction", 0.20))
    if probe_fraction > max_probe_fraction:
        raise PreflightError(
            "BF16_PROBE_MEMORY_TOO_HIGH",
            f"BF16 probe reserved {probe_fraction:.1%} of VRAM, limit is {max_probe_fraction:.1%}",
            "Investigate allocator/GPU contention before starting the full model.",
        )
    return {
        "device": asdict(snapshot),
        "device_total_vram_gib": snapshot.total_memory_bytes / GIB,
        "device_used_vram_gib": snapshot.used_memory_bytes / GIB,
        "device_used_vram_fraction": used_fraction,
        "estimated_training_peak_gib": estimated_peak_bytes / GIB,
        "estimated_training_vram_fraction": estimated_fraction,
        "bf16_probe": asdict(probe),
        "bf16_probe_vram_fraction": probe_fraction,
    }


def _resolve_project_root(
    loaded: LoadedConfig,
    explicit_project_root: Path | None,
) -> Path:
    if explicit_project_root is not None:
        return explicit_project_root.resolve()
    relative = loaded.values["project"].get("root_relative_to_config", "..")
    return (loaded.path.parent / str(relative)).resolve()


def run_preflight(
    config_path: Path | str,
    *,
    backend: CudaBackend | None = None,
    project_root: Path | None = None,
    run_dir: Path | None = None,
    run_id: str | None = None,
    check_artifacts: bool = True,
) -> dict[str, Any]:
    """Run all release and hardware gates and persist an auditable report."""
    loaded = load_versioned_config(config_path)
    values = loaded.values
    resolved_root = _resolve_project_root(loaded, project_root)
    run_identifier = run_id or generate_run_id(values["project"].get("run_id_prefix", "doupo"))
    if run_dir is None:
        run_root = Path(values["artifacts"].get("run_root", "runs"))
        run_directory = resolved_root / run_root / run_identifier
    else:
        run_directory = Path(run_dir)
    logging_config = values["logging"]
    loggers = configure_module_loggers(
        run_directory / "logs",
        run_identifier,
        logging_config.get("module_levels", {}),
        max_bytes=int(logging_config.get("max_bytes", 10 * 1024 * 1024)),
        backup_count=int(logging_config.get("backup_count", 5)),
        console=bool(logging_config.get("console", True)),
    )
    status = RunStateWriter(
        run_directory,
        run_identifier,
        loaded.file_sha256,
    )
    started_at = utc_now()
    loggers["preflight"].info(
        "preflight started",
        extra={
            "context": {
                "config": loaded.path,
                "config_sha256": loaded.file_sha256,
                "project_root": resolved_root,
            }
        },
    )
    try:
        artifact_report = (
            verify_release_artifacts(resolved_root, values["data"])
            if check_artifacts
            else []
        )
        hardware_report = verify_cuda_requirements(
            values["hardware"],
            backend or TorchCudaBackend(),
        )
        report = {
            "schema_version": "cloud-preflight-report/v1",
            "status": "PASSED",
            "run_id": run_identifier,
            "started_at": started_at,
            "finished_at": utc_now(),
            "config_path": str(loaded.path),
            "config_version": values["config_version"],
            "config_file_sha256": loaded.file_sha256,
            "config_canonical_sha256": loaded.canonical_sha256,
            "project_root": str(resolved_root),
            "artifacts": artifact_report,
            "hardware": hardware_report,
        }
        report_path = run_directory / "preflight_report.json"
        atomic_write_json(report_path, report)
        loggers["preflight"].info(
            "preflight passed",
            extra={"context": {"report": report_path, "gpu": hardware_report["device"]["name"]}},
        )
        return report
    except Exception as error:
        if isinstance(error, PreflightError):
            failure = error.as_dict()
        else:
            failure = {
                "code": "PREFLIGHT_INTERNAL_ERROR",
                "message": str(error),
                "action": "Inspect the preflight log and stack trace; fix the preflight itself before training.",
            }
        status.mark_failed(error, {"preflight": failure})
        loggers["preflight"].exception(
            "preflight failed",
            extra={"context": failure},
        )
        raise
    finally:
        close_module_loggers(loggers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cloud_a10.json"),
        help="versioned cloud configuration with a matching .sha256 sidecar",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="optional explicit output directory for this preflight",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="override project root resolution (mainly for packaged cloud jobs)",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        report = run_preflight(
            arguments.config,
            project_root=arguments.project_root,
            run_dir=arguments.run_dir,
        )
    except PreflightError as error:
        print(json.dumps({"status": "FAILED", **error.as_dict()}, ensure_ascii=False, indent=2))
        return 2
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "code": "PREFLIGHT_INTERNAL_ERROR",
                    "message": str(error),
                    "action": "Inspect the preflight logs; do not start training.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 3
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
