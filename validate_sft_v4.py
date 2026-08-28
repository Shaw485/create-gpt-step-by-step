from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from build_sft_v4 import (
    CANDIDATE_DIR,
    CANDIDATE_PATH,
    CORPUS_PATH,
    SftV4ReleaseBlocked,
    atomic_write_text,
    configure_sft_v4_logging,
    quality_gate,
    read_jsonl,
    release_records,
    sha256_file,
)


VALIDATION_REPORT_PATH = CANDIDATE_DIR / "sft_v4_validation.json"


def validate_dataset(
    *,
    dataset_path: Path,
    corpus_path: Path,
    report_path: Path,
    require_release: bool = False,
    export_release: bool = False,
    log_dir: Path = Path("logs"),
) -> dict:
    loggers = configure_sft_v4_logging(log_dir)
    data_logger = loggers["data"]
    validation_logger = loggers["validation"]
    try:
        records = read_jsonl(dataset_path)
        corpus_text = corpus_path.read_text(encoding="utf-8")
        corpus_lines = corpus_text.splitlines()
        corpus_hash = sha256_file(corpus_path)
        data_logger.info(
            "loaded validation input dataset=%s records=%d corpus=%s sha256=%s",
            dataset_path,
            len(records),
            corpus_path,
            corpus_hash,
        )
        report = quality_gate(records, corpus_lines, corpus_hash)
        report.update(
            {
                "dataset_path": str(dataset_path),
                "dataset_sha256": sha256_file(dataset_path),
                "corpus_path": str(corpus_path),
                "corpus_sha256": corpus_hash,
            }
        )
        atomic_write_text(
            report_path,
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        validation_logger.info(
            "validated dataset=%s release_ready=%s failed_gates=%s report=%s",
            dataset_path,
            report["release_ready"],
            report["failed_gates"],
            report_path,
        )
        if require_release and not report["release_ready"]:
            raise SftV4ReleaseBlocked(
                "SFT v4 release blocked by gates: "
                + ", ".join(report["failed_gates"])
            )
        if export_release:
            release_records(records, report, corpus_path)
            validation_logger.info("release export completed")
        return report
    except Exception:
        validation_logger.exception("SFT v4 validation failed")
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an SFT v4 JSONL dataset.")
    parser.add_argument("--dataset", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--report", type=Path, default=VALIDATION_REPORT_PATH)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--require-release",
        action="store_true",
        help="Exit unsuccessfully when any release gate fails.",
    )
    parser.add_argument(
        "--export-release",
        action="store_true",
        help="Validate and export data/cloud_v4; implies --require-release.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_dataset(
        dataset_path=args.dataset,
        corpus_path=args.corpus,
        report_path=args.report,
        require_release=args.require_release or args.export_release,
        export_release=args.export_release,
        log_dir=args.log_dir,
    )
    print(
        json.dumps(
            {
                "release_ready": report["release_ready"],
                "failed_gates": report["failed_gates"],
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
