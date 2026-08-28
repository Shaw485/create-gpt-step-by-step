import json
from pathlib import Path
import tempfile
import unittest

from cloud_preflight import (
    Bf16ProbeResult,
    CudaDeviceSnapshot,
    GIB,
    PreflightError,
    run_preflight,
    verify_cuda_requirements,
    verify_release_artifacts,
)
from training_runtime import file_sha256


class FakeCudaBackend:
    def __init__(
        self,
        *,
        available=True,
        count=1,
        name="NVIDIA A10",
        total_memory_bytes=24 * GIB,
        used_memory_bytes=1 * GIB,
        capability=(8, 6),
        bf16=True,
        probe=None,
    ):
        self.available = available
        self.count = count
        self.snapshot = CudaDeviceSnapshot(
            index=0,
            name=name,
            total_memory_bytes=total_memory_bytes,
            used_memory_bytes=used_memory_bytes,
            compute_capability=capability,
        )
        self.bf16 = bf16
        self.probe = probe or Bf16ProbeResult(
            forward_finite=True,
            loss_finite=True,
            gradients_finite=True,
            output_dtype="torch.bfloat16",
            peak_allocated_bytes=256 * 1024 * 1024,
            peak_reserved_bytes=512 * 1024 * 1024,
        )
        self.selected = None

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def select_device(self, index):
        self.selected = index

    def device_snapshot(self, index):
        return self.snapshot

    def is_bf16_supported(self, index):
        return self.bf16

    def run_bf16_probe(self, index, *, matrix_size, batch_size):
        self.probe_arguments = (index, matrix_size, batch_size)
        return self.probe


def hardware_config():
    return {
        "device": "cuda",
        "device_index": 0,
        "required_gpu_count": 1,
        "required_gpu_name_regex": r"\b(?:NVIDIA\s+)?A10\b",
        "min_vram_gib": 22.0,
        "min_compute_capability": [8, 6],
        "require_native_bf16": True,
        "estimated_training_peak_gib": 16.0,
        "max_training_vram_fraction": 0.8,
        "max_existing_vram_fraction": 0.2,
        "max_probe_vram_fraction": 0.1,
        "bf16_probe": {"matrix_size": 512, "batch_size": 16},
    }


class CloudPreflightTests(unittest.TestCase):
    def test_mock_a10_passes_without_real_cuda(self):
        backend = FakeCudaBackend()
        report = verify_cuda_requirements(hardware_config(), backend)
        self.assertEqual(backend.selected, 0)
        self.assertEqual(report["device"]["name"], "NVIDIA A10")
        self.assertLess(report["estimated_training_vram_fraction"], 0.8)
        self.assertEqual(backend.probe_arguments, (0, 512, 16))

    def test_no_cuda_fails_with_action_and_never_falls_back_to_cpu(self):
        with self.assertRaises(PreflightError) as context:
            verify_cuda_requirements(hardware_config(), FakeCudaBackend(available=False))
        self.assertEqual(context.exception.code, "CUDA_UNAVAILABLE")
        self.assertIn("GPU instance", context.exception.action)

        cpu_config = hardware_config()
        cpu_config["device"] = "cpu"
        with self.assertRaises(PreflightError) as context:
            verify_cuda_requirements(cpu_config, FakeCudaBackend())
        self.assertEqual(context.exception.code, "CPU_FALLBACK_FORBIDDEN")

    def test_wrong_card_bf16_and_vram_threshold_each_fail(self):
        cases = [
            (FakeCudaBackend(name="NVIDIA T4"), "WRONG_GPU_MODEL"),
            (FakeCudaBackend(bf16=False), "BF16_UNSUPPORTED"),
            (
                FakeCudaBackend(used_memory_bytes=6 * GIB),
                "GPU_ALREADY_BUSY",
            ),
            (
                FakeCudaBackend(
                    probe=Bf16ProbeResult(
                        True,
                        True,
                        True,
                        "torch.bfloat16",
                        3 * GIB,
                        4 * GIB,
                    )
                ),
                "BF16_PROBE_MEMORY_TOO_HIGH",
            ),
        ]
        for backend, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(PreflightError) as context:
                    verify_cuda_requirements(hardware_config(), backend)
                self.assertEqual(context.exception.code, code)
                self.assertTrue(context.exception.action)

    def test_nonfinite_or_wrong_dtype_probe_fails(self):
        nonfinite = Bf16ProbeResult(
            forward_finite=True,
            loss_finite=False,
            gradients_finite=False,
            output_dtype="torch.bfloat16",
            peak_allocated_bytes=1,
            peak_reserved_bytes=1,
        )
        with self.assertRaises(PreflightError) as context:
            verify_cuda_requirements(
                hardware_config(), FakeCudaBackend(probe=nonfinite)
            )
        self.assertEqual(context.exception.code, "BF16_PROBE_NONFINITE")

        wrong_dtype = Bf16ProbeResult(True, True, True, "torch.float32", 1, 1)
        with self.assertRaises(PreflightError) as context:
            verify_cuda_requirements(
                hardware_config(), FakeCudaBackend(probe=wrong_dtype)
            )
        self.assertEqual(context.exception.code, "BF16_PROBE_WRONG_DTYPE")

    def test_release_manifest_status_and_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "data" / "cloud_v4" / "train.txt"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("reviewed training text", encoding="utf-8")
            manifest = artifact.parent / "corpus_manifest.json"
            manifest_payload = {
                "status": "ready",
                "artifacts": [
                    {
                        "path": "data/cloud_v4/train.txt",
                        "sha256": file_sha256(artifact),
                        "size_bytes": artifact.stat().st_size,
                    }
                ],
            }
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            Path(f"{manifest}.sha256").write_text(
                f"{file_sha256(manifest)}  corpus_manifest.json\n",
                encoding="utf-8",
            )
            config = {
                "release_manifests": [
                    {
                        "path": "data/cloud_v4/corpus_manifest.json",
                        "required_status": "ready",
                        "required_files": ["data/cloud_v4/train.txt"],
                    }
                ]
            }
            report = verify_release_artifacts(root, config)
            self.assertEqual(report[0]["status"], "ready")
            self.assertEqual(
                report[0]["verified_artifacts"][0]["sha256"],
                file_sha256(artifact),
            )

            artifact.write_text("changed after review", encoding="utf-8")
            with self.assertRaises(PreflightError) as context:
                verify_release_artifacts(root, config)
            self.assertEqual(context.exception.code, "ARTIFACT_HASH_MISMATCH")

    def test_not_ready_manifest_is_an_actionable_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"status": "review_required", "artifacts": []}),
                encoding="utf-8",
            )
            Path(f"{manifest}.sha256").write_text(
                file_sha256(manifest), encoding="utf-8"
            )
            with self.assertRaises(PreflightError) as context:
                verify_release_artifacts(
                    root,
                    {
                        "release_manifests": [
                            {
                                "path": "manifest.json",
                                "required_status": "ready",
                                "required_files": ["train.txt"],
                            }
                        ]
                    },
                )
            self.assertEqual(context.exception.code, "ARTIFACT_REVIEW_INCOMPLETE")
            self.assertIn("review-queue", context.exception.action)

    def test_end_to_end_mock_run_writes_report_or_failed_status(self):
        project_root = Path(__file__).resolve().parents[1]
        config_path = project_root / "configs" / "cloud_a10.json"
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "success"
            report = run_preflight(
                config_path,
                backend=FakeCudaBackend(),
                project_root=project_root,
                run_dir=run_dir,
                run_id="mock-success",
                check_artifacts=False,
            )
            self.assertEqual(report["status"], "PASSED")
            self.assertEqual(report["run_id"], "mock-success")
            self.assertTrue((run_dir / "preflight_report.json").is_file())

            failed_dir = Path(directory) / "failed"
            with self.assertRaises(PreflightError):
                run_preflight(
                    config_path,
                    backend=FakeCudaBackend(available=False),
                    project_root=project_root,
                    run_dir=failed_dir,
                    run_id="mock-failed",
                    check_artifacts=False,
                )
            failed = json.loads((failed_dir / "FAILED.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "FAILED")
            self.assertEqual(failed["details"]["preflight"]["code"], "CUDA_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
