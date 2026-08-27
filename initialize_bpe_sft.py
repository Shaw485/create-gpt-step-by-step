"""Expand a BPE-pretrained checkpoint with five chat-control tokens."""

from __future__ import annotations

import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

import torch

from evaluate_sft_baseline import expand_pretrained_model, verify_pretrained_weights_copied


PRETRAINED_PATH = Path(
    os.getenv("BPE_SFT_PRETRAINED", "checkpoints/bpe_pretrain_step10000_best.pt")
)
DATA_PATH = Path(
    os.getenv("BPE_SFT_DATA", "data/bpe/sft_balanced_v3_bpe_tensors.pt")
)
OUTPUT_PATH = Path(
    os.getenv("BPE_SFT_INIT", "checkpoints/bpe_sft_init_pre_sft.pt")
)
REPORT_PATH = Path(
    os.getenv("BPE_SFT_INIT_REPORT", "reports/bpe_sft_init_report.json")
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_logger() -> logging.Logger:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bpe.sft.initialize")
    logger.handlers.clear()
    logger.propagate = False
    level = getattr(
        logging, os.getenv("BPE_SFT_INIT_LOG_LEVEL", "INFO").upper(), logging.INFO
    )
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = RotatingFileHandler(
        "logs/bpe_sft_initialize.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if os.getenv("BPE_SFT_INIT_CONSOLE_LOG", "1") == "1":
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger


def main() -> None:
    logger = configure_logger()
    try:
        pretrained = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
        data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
        base_vocab_size = int(pretrained["meta"]["vocab_size"])
        extended_vocab_size = int(data["vocab_size"])
        if base_vocab_size != int(data["base_vocab_size"]):
            raise ValueError("pretrained and SFT base vocabulary sizes differ")
        model = expand_pretrained_model(pretrained, extended_vocab_size)
        verify_pretrained_weights_copied(model, pretrained)
        meta = dict(pretrained["meta"])
        meta.update(
            {
                "stage": "bpe_pre_sft_initialized",
                "vocab_size": extended_vocab_size,
                "base_vocab_size": base_vocab_size,
                "pretrained_checkpoint": str(PRETRAINED_PATH),
                "pretrained_checkpoint_sha256": sha256(PRETRAINED_PATH),
                "sft_data_path": str(DATA_PATH),
                "sft_data_sha256": sha256(DATA_PATH),
                "special_token_ids": data["special_token_ids"],
                "sft_steps": 0,
                "test_records_consumed": 0,
            }
        )
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "meta": meta}, OUTPUT_PATH)
        report = {
            "stage": "bpe_pre_sft_initialized",
            "base_vocab_size": base_vocab_size,
            "extended_vocab_size": extended_vocab_size,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "pretrained_weights_exactly_copied": True,
            "new_special_tokens": list(data["special_token_ids"]),
            "pretrained_checkpoint": str(PRETRAINED_PATH),
            "pretrained_checkpoint_sha256": sha256(PRETRAINED_PATH),
            "output_checkpoint": str(OUTPUT_PATH),
            "output_checkpoint_sha256": sha256(OUTPUT_PATH),
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "initialized base_vocab=%d extended_vocab=%d params=%d output=%s",
            base_vocab_size, extended_vocab_size, report["parameter_count"], OUTPUT_PATH,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception:
        logger.exception("BPE SFT initialization failed")
        raise


if __name__ == "__main__":
    main()
