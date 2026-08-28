"""Reusable safety primitives for unattended cloud training.

The module deliberately does not import the existing training entry points.  It
can therefore be tested on a CPU-only development machine and integrated into
pretraining and SFT one stage at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import random
import re
import signal
import time
from types import FrameType
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence
import uuid

import torch


CHECKPOINT_SCHEMA_VERSION = "training-checkpoint/v1"
CONFIG_SCHEMA_VERSION = "cloud-training-config/v1"
DEFAULT_LOG_MODULES = (
    "preflight",
    "data",
    "pretrain",
    "validation",
    "checkpoint",
    "gpu",
    "sft",
    "orchestrator",
)

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "passwd",
    "private_key",
    "secret",
    "token",
}
_SECRET_TEXT_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|cookie|password|passwd|"
    r"private[_-]?key|secret|token)\b(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp that is stable in JSON artifacts."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generate_run_id(prefix: str = "doupo") -> str:
    """Create a sortable, collision-resistant run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def file_sha256(path: Path | str) -> str:
    """Calculate SHA-256 without loading a potentially large file into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash JSON content independently of indentation and dictionary ordering."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    values: dict[str, Any]
    file_sha256: str
    canonical_sha256: str


class ConfigurationError(ValueError):
    """Raised when a versioned run configuration is incomplete or changed."""


def _read_checksum_sidecar(path: Path) -> str:
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file():
        raise ConfigurationError(
            f"missing configuration checksum: {sidecar}; regenerate it after "
            f"reviewing {path.name}"
        )
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
        raise ConfigurationError(f"invalid SHA-256 sidecar format: {sidecar}")
    return fields[0].lower()


def load_versioned_config(
    path: Path | str,
    *,
    verify_sidecar: bool = True,
) -> LoadedConfig:
    """Load a complete, versioned configuration and verify its file digest."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {config_path}")
    try:
        values = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot parse configuration {config_path}: {error}") from error
    if not isinstance(values, dict):
        raise ConfigurationError("configuration root must be a JSON object")
    if values.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigurationError(
            f"schema_version must be {CONFIG_SCHEMA_VERSION!r}, got "
            f"{values.get('schema_version')!r}"
        )
    version = values.get("config_version")
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ConfigurationError("config_version must use semantic versioning, for example 1.0.0")
    required_sections = {
        "project",
        "data",
        "model",
        "pretraining",
        "sft",
        "evaluation",
        "hardware",
        "runtime",
        "logging",
        "safety",
        "artifacts",
    }
    missing = sorted(required_sections.difference(values))
    if missing:
        raise ConfigurationError(f"configuration is missing sections: {', '.join(missing)}")
    digest = file_sha256(config_path)
    if verify_sidecar:
        expected = _read_checksum_sidecar(config_path)
        if digest != expected:
            raise ConfigurationError(
                f"configuration SHA-256 mismatch for {config_path}: expected "
                f"{expected}, calculated {digest}; review the change and update the sidecar"
            )
    return LoadedConfig(
        path=config_path,
        values=values,
        file_sha256=digest,
        canonical_sha256=canonical_json_sha256(values),
    )


def _is_secret_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return (
        normalized in _SECRET_KEYS
        or normalized.endswith("_password")
        or normalized.endswith("_secret")
        or normalized.endswith("_token")
        or normalized.endswith("_api_key")
    )


def redact_text(value: str) -> str:
    """Redact common inline credential forms before they reach a handler."""
    value = _BEARER_PATTERN.sub("Bearer [REDACTED]", value)
    return _SECRET_TEXT_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        value,
    )


def redact_sensitive(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact secret fields while preserving useful log context."""
    if key is not None and _is_secret_key(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_sensitive(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(repr(value))


class JsonLogFormatter(logging.Formatter):
    """One-JSON-object-per-line formatter with run correlation and redaction."""

    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "module": record.name,
            "run_id": self.run_id,
            "message": redact_text(record.getMessage()),
        }
        context = getattr(record, "context", None)
        if context is not None:
            payload["context"] = redact_sensitive(context)
        if record.exc_info:
            payload["exception"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _parse_log_level(value: str) -> tuple[int, bool]:
    normalized = value.strip().upper()
    if normalized in {"OFF", "DISABLED", "NONE"}:
        return logging.CRITICAL + 1, True
    level_names = getattr(logging, "getLevelNamesMapping", None)
    if level_names is not None:
        level = level_names().get(normalized)
    else:  # Python 3.9/3.10 compatibility.
        level = getattr(logging, normalized, None)
    if not isinstance(level, int):
        raise ConfigurationError(f"unknown log level: {value!r}")
    return level, False


def configure_module_loggers(
    log_dir: Path | str,
    run_id: str,
    module_levels: Mapping[str, str],
    *,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
) -> dict[str, logging.Logger]:
    """Create independent rotating JSONL logs for every runtime subsystem."""
    if max_bytes <= 0 or backup_count < 0:
        raise ConfigurationError("log max_bytes must be positive and backup_count non-negative")
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    formatter = JsonLogFormatter(run_id)
    result: dict[str, logging.Logger] = {}
    for module in DEFAULT_LOG_MODULES:
        level, disabled = _parse_log_level(module_levels.get(module, "INFO"))
        logger = logging.getLogger(f"cloud.{module}")
        for existing_handler in logger.handlers:
            existing_handler.close()
        logger.handlers.clear()
        logger.propagate = False
        logger.disabled = disabled
        logger.setLevel(level)
        file_handler = RotatingFileHandler(
            directory / f"{run_id}.{module}.jsonl",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        result[module] = logger
    return result


def close_module_loggers(loggers: Mapping[str, logging.Logger]) -> None:
    """Flush and close module handlers, which is useful between pipeline stages."""
    for logger in loggers.values():
        for handler in logger.handlers:
            try:
                handler.flush()
            finally:
                handler.close()
        logger.handlers.clear()
        logger.disabled = False


def _fsync_directory(path: Path) -> None:
    """Persist directory metadata after os.replace on POSIX filesystems."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write a small artifact via fsync plus an atomic rename."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Atomically write human-readable JSON after redacting sensitive fields."""
    safe_payload = redact_sensitive(payload)
    atomic_write_text(
        path,
        json.dumps(safe_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def capture_rng_state(sampling_generator: torch.Generator) -> dict[str, Any]:
    """Capture every RNG required for deterministic training continuation."""
    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": cuda_states,
        "python_random": random.getstate(),
        "sampling_generator": sampling_generator.get_state(),
    }


def build_checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    best_metric: float,
    history: Sequence[Mapping[str, Any]],
    sampling_generator: torch.Generator,
    config_sha256: str,
    amp_scaler: Any | None = None,
    early_stopping_state: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete checkpoint suitable for an unattended resume."""
    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", config_sha256):
        raise ValueError("config_sha256 must contain exactly 64 hexadecimal characters")
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "config_sha256": config_sha256.lower(),
        "step": int(step),
        "best_metric": float(best_metric),
        "history": [dict(entry) for entry in history],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(sampling_generator),
        "amp_scaler_state_dict": (
            amp_scaler.state_dict() if amp_scaler is not None else None
        ),
        "early_stopping_state": dict(early_stopping_state or {}),
        "extra": dict(extra or {}),
    }
    validate_checkpoint_payload(payload)
    return payload


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    """Reject truncated or incompatible checkpoint payloads before use."""
    required = {
        "schema_version",
        "config_sha256",
        "step",
        "best_metric",
        "history",
        "model_state_dict",
        "optimizer_state_dict",
        "rng_state",
        "amp_scaler_state_dict",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"checkpoint is missing keys: {', '.join(missing)}")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema: {payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("step"), int) or int(payload["step"]) < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    if not isinstance(payload.get("history"), list):
        raise ValueError("checkpoint history must be a list")
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    if not isinstance(payload.get("optimizer_state_dict"), Mapping):
        raise ValueError("checkpoint optimizer_state_dict must be a mapping")
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, Mapping):
        raise ValueError("checkpoint rng_state must be a mapping")
    rng_required = {"torch_cpu", "torch_cuda_all", "sampling_generator"}
    rng_missing = sorted(rng_required.difference(rng_state))
    if rng_missing:
        raise ValueError(f"checkpoint RNG state is missing: {', '.join(rng_missing)}")


def _torch_load(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    """Load trusted local checkpoints across old and new PyTorch APIs."""
    try:
        value = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        value = torch.load(path, map_location=map_location)
    if not isinstance(value, dict):
        raise ValueError("checkpoint root must be a dictionary")
    return value


@dataclass(frozen=True)
class CheckpointSaveResult:
    path: Path
    sha256: str
    size_bytes: int
    step: int
    reload_verified: bool


def atomic_save_checkpoint(
    path: Path | str,
    payload: Mapping[str, Any],
) -> CheckpointSaveResult:
    """Atomically save, checksum, reload, and validate a checkpoint."""
    validate_checkpoint_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()

    digest = file_sha256(destination)
    atomic_write_text(
        f"{destination}.sha256",
        f"{digest}  {destination.name}\n",
    )
    reloaded = _torch_load(destination, "cpu")
    validate_checkpoint_payload(reloaded)
    if int(reloaded["step"]) != int(payload["step"]):
        raise IOError(
            f"checkpoint reload verification failed: expected step {payload['step']}, "
            f"got {reloaded['step']}"
        )
    if file_sha256(destination) != digest:
        raise IOError("checkpoint changed during reload verification")
    return CheckpointSaveResult(
        path=destination,
        sha256=digest,
        size_bytes=destination.stat().st_size,
        step=int(payload["step"]),
        reload_verified=True,
    )


def load_checkpoint(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
    expected_config_sha256: str | None = None,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    """Verify and load a trusted local checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    if verify_checksum:
        sidecar = Path(f"{checkpoint_path}.sha256")
        if not sidecar.is_file():
            raise ValueError(f"checkpoint checksum is missing: {sidecar}")
        fields = sidecar.read_text(encoding="utf-8").strip().split()
        if not fields or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
            raise ValueError(f"checkpoint checksum is malformed: {sidecar}")
        calculated = file_sha256(checkpoint_path)
        if calculated != fields[0].lower():
            raise ValueError(
                f"checkpoint SHA-256 mismatch: expected {fields[0].lower()}, "
                f"calculated {calculated}"
            )
    payload = _torch_load(checkpoint_path, map_location)
    validate_checkpoint_payload(payload)
    if (
        expected_config_sha256 is not None
        and payload["config_sha256"] != expected_config_sha256.lower()
    ):
        raise ValueError(
            "checkpoint configuration does not match this run; use the original "
            "configuration or start a new run"
        )
    return payload


@dataclass(frozen=True)
class ResumeState:
    step: int
    best_metric: float
    history: list[dict[str, Any]]
    early_stopping_state: dict[str, Any]
    extra: dict[str, Any]


def restore_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampling_generator: torch.Generator,
    *,
    amp_scaler: Any | None = None,
    expected_config_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
    restore_cuda_rng: bool = True,
) -> ResumeState:
    """Restore model, optimizer, progress, scaler, and all captured RNG state."""
    payload = load_checkpoint(
        path,
        map_location=map_location,
        expected_config_sha256=expected_config_sha256,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.set_rng_state(payload["rng_state"]["torch_cpu"].cpu())
    sampling_generator.set_state(payload["rng_state"]["sampling_generator"].cpu())
    python_state = payload["rng_state"].get("python_random")
    if python_state is not None:
        random.setstate(python_state)
    cuda_states = payload["rng_state"].get("torch_cuda_all", [])
    if restore_cuda_rng and cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable; resume "
                "on the training GPU or pass restore_cuda_rng=False for inspection only"
            )
        torch.cuda.set_rng_state_all(cuda_states)
    scaler_state = payload.get("amp_scaler_state_dict")
    if scaler_state is not None:
        if amp_scaler is None:
            raise ValueError("checkpoint contains AMP scaler state but no scaler was supplied")
        amp_scaler.load_state_dict(scaler_state)
    return ResumeState(
        step=int(payload["step"]),
        best_metric=float(payload["best_metric"]),
        history=[dict(entry) for entry in payload["history"]],
        early_stopping_state=dict(payload.get("early_stopping_state", {})),
        extra=dict(payload.get("extra", {})),
    )


class NumericalFailure(RuntimeError):
    """Raised before corrupt gradients can be applied to model parameters."""


def assert_finite_tensor(tensor: torch.Tensor, operation: str) -> None:
    """Fail with an actionable label when a loss or activation is non-finite."""
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise NumericalFailure(
            f"non-finite value detected during {operation}; do not update parameters, "
            "save an emergency checkpoint, then inspect learning rate and input data"
        )


def assert_finite_gradients(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> None:
    """Validate all present gradients after backward and before optimizer.step."""
    failures: list[str] = []
    for name, parameter in named_parameters:
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad.detach()).all().item()
        ):
            failures.append(name)
    if failures:
        preview = ", ".join(failures[:5])
        suffix = "..." if len(failures) > 5 else ""
        raise NumericalFailure(
            f"non-finite gradients in {preview}{suffix}; skip optimizer.step and "
            "inspect the last batch, learning rate, and gradient scaling"
        )


@dataclass(frozen=True)
class EarlyStoppingDecision:
    improved: bool
    should_stop: bool
    best_metric: float
    bad_evaluations: int
    best_step: int


class EarlyStopping:
    """Validation-based early stopping whose state can live in checkpoints."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        if patience <= 0:
            raise ValueError("early-stopping patience must be positive")
        if min_delta < 0:
            raise ValueError("early-stopping min_delta cannot be negative")
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = math.inf
        self.bad_evaluations = 0
        self.best_step = -1

    def update(self, metric: float, step: int) -> EarlyStoppingDecision:
        if not math.isfinite(metric):
            raise NumericalFailure("validation metric is NaN or Inf")
        improved = metric < self.best_metric - self.min_delta
        if improved:
            self.best_metric = float(metric)
            self.bad_evaluations = 0
            self.best_step = int(step)
        else:
            self.bad_evaluations += 1
        return EarlyStoppingDecision(
            improved=improved,
            should_stop=self.bad_evaluations >= self.patience,
            best_metric=self.best_metric,
            bad_evaluations=self.bad_evaluations,
            best_step=self.best_step,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_metric": self.best_metric,
            "bad_evaluations": self.bad_evaluations,
            "best_step": self.best_step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.patience = int(state["patience"])
        self.min_delta = float(state["min_delta"])
        self.best_metric = float(state["best_metric"])
        self.bad_evaluations = int(state["bad_evaluations"])
        self.best_step = int(state["best_step"])


class WallClockExceeded(TimeoutError):
    """Raised so the caller can checkpoint before the rental deadline."""


class WallClockLimit:
    """Monotonic maximum-wall-clock guard, independent of system clock changes."""

    def __init__(
        self,
        max_seconds: float,
        *,
        started_monotonic: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_seconds <= 0:
            raise ValueError("max wall-clock seconds must be positive")
        self.max_seconds = float(max_seconds)
        self.clock = clock
        self.started_monotonic = (
            float(started_monotonic)
            if started_monotonic is not None
            else float(clock())
        )

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, float(self.clock()) - self.started_monotonic)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed_seconds)

    def check(self, *, reserve_seconds: float = 0.0) -> None:
        if reserve_seconds < 0:
            raise ValueError("wall-clock reserve cannot be negative")
        if self.elapsed_seconds + reserve_seconds >= self.max_seconds:
            raise WallClockExceeded(
                f"maximum wall clock reached with {reserve_seconds:.1f}s reserve; "
                "save latest checkpoint and stop cleanly"
            )


class RunStateWriter:
    """Write explicit DONE.json or FAILED.json terminal state artifacts."""

    def __init__(
        self,
        run_dir: Path | str,
        run_id: str,
        config_sha256: str,
        *,
        started_at: str | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.config_sha256 = config_sha256
        self.started_at = started_at or utc_now()

    def mark_done(self, details: Mapping[str, Any] | None = None) -> Path:
        path = self.run_dir / "DONE.json"
        atomic_write_json(
            path,
            {
                "schema_version": "training-run-status/v1",
                "status": "DONE",
                "run_id": self.run_id,
                "config_sha256": self.config_sha256,
                "started_at": self.started_at,
                "finished_at": utc_now(),
                "details": dict(details or {}),
            },
        )
        return path

    def mark_failed(
        self,
        error: BaseException | str,
        details: Mapping[str, Any] | None = None,
    ) -> Path:
        path = self.run_dir / "FAILED.json"
        error_type = type(error).__name__ if isinstance(error, BaseException) else "Failure"
        message = str(error)
        atomic_write_json(
            path,
            {
                "schema_version": "training-run-status/v1",
                "status": "FAILED",
                "run_id": self.run_id,
                "config_sha256": self.config_sha256,
                "started_at": self.started_at,
                "finished_at": utc_now(),
                "error_type": error_type,
                "message": message,
                "details": dict(details or {}),
            },
        )
        return path


class EmergencyCheckpointHook:
    """Save an emergency checkpoint when SIGTERM or SIGINT reaches Python."""

    def __init__(
        self,
        path: Path | str,
        payload_factory: Callable[[], Mapping[str, Any]],
        *,
        logger: logging.Logger | None = None,
        exit_after_save: bool = True,
    ):
        self.path = Path(path)
        self.payload_factory = payload_factory
        self.logger = logger
        self.exit_after_save = exit_after_save
        self.received_signal: int | None = None
        self.save_result: CheckpointSaveResult | None = None
        self._saving = False
        self._old_handlers: MutableMapping[int, Any] = {}

    def install(self) -> "EmergencyCheckpointHook":
        if self._old_handlers:
            return self
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)
        return self

    def uninstall(self) -> None:
        for signum, handler in self._old_handlers.items():
            signal.signal(signum, handler)
        self._old_handlers.clear()

    def __enter__(self) -> "EmergencyCheckpointHook":
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.uninstall()

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        del frame
        if self._saving:
            return
        self._saving = True
        self.received_signal = signum
        try:
            self.save_result = atomic_save_checkpoint(
                self.path,
                self.payload_factory(),
            )
            if self.logger:
                self.logger.warning(
                    "emergency checkpoint saved",
                    extra={
                        "context": {
                            "signal": signal.Signals(signum).name,
                            "path": self.path,
                            "step": self.save_result.step,
                            "sha256": self.save_result.sha256,
                        }
                    },
                )
        finally:
            self._saving = False
        if self.exit_after_save:
            raise SystemExit(128 + signum)
