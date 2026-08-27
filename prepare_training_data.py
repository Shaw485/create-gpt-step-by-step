from pathlib import Path
import hashlib
import json
import logging
import os
import torch
from logging.handlers import RotatingFileHandler

TEXT_PATH = Path(
    os.getenv("TRAIN_TEXT_PATH", "data/clean/doupo_stage3.txt")
)
REPORT_PATH = Path(
    os.getenv("TRAIN_DATA_REPORT", "data/clean/stage3_dataset_report.json")
)
VOCAB_PATH = Path(
    os.getenv("TRAIN_VOCAB_PATH", "data/clean/stage3_vocab.json")
)
TENSOR_PATH = Path(
    os.getenv("TRAIN_TENSOR_PATH", "data/clean/doupo_stage3_tensors.pt")
)
LOG_PATH = Path("logs/prepare_training_data.log")
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.9"))


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def configure_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger("train_data")
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.propagate = False

    io_level = os.getenv("TRAIN_DATA_IO_LOG_LEVEL", "INFO").upper()
    transform_level = os.getenv("TRAIN_DATA_TRANSFORM_LOG_LEVEL", "INFO").upper()
    logging.getLogger("train_data.io").setLevel(io_level)
    logging.getLogger("train_data.transform").setLevel(transform_level)


def build_encoder(text):
    transform_logger = logging.getLogger("train_data.transform")

    chars = sorted(set(text))
    stoi = {ch: index for index, ch in enumerate(chars)}
    itos = {index: ch for index, ch in enumerate(chars)}

    transform_logger.info("vocab_size=%d unique_chars=%d", len(chars), len(chars))

    def encode(input_text):
        return [stoi[ch] for ch in input_text]

    def decode(token_ids):
        return "".join(itos[token] for token in token_ids)

    return encode, decode, stoi, itos, len(chars)


def main():
    configure_logging()
    io_logger = logging.getLogger("train_data.io")
    transform_logger = logging.getLogger("train_data.transform")

    try:
        io_logger.info("reading text path=%s", TEXT_PATH)
        text = TEXT_PATH.read_text(encoding="utf-8")
        text_sha256 = sha256_text(text)
        io_logger.info("characters=%d", len(text))

        encode, decode, stoi, itos, vocab_size = build_encoder(text)
        encode_start = len(stoi)
        decode_start = len(itos)

        encoded = encode(text)
        data = torch.tensor(encoded, dtype=torch.long)

        if encode_start != vocab_size or decode_start != vocab_size:
            raise RuntimeError("encoder internal mismatch")
        if len(stoi) != len(itos):
            raise RuntimeError("vocabulary size mismatch")
        sample_char = next(iter(stoi))
        sample_id = stoi[sample_char]
        if decode([sample_id]) != sample_char:
            raise RuntimeError("encoder decode check failed")

        if not 0 < TRAIN_RATIO < 1:
            raise ValueError("TRAIN_RATIO must be between 0 and 1", TRAIN_RATIO)

        split_index = int(len(data) * TRAIN_RATIO)
        train_data = data[:split_index]
        val_data = data[split_index:]

        transform_logger.debug("split_index=%d train=%d val=%d", split_index, len(train_data), len(val_data))
        if split_index <= 0:
            raise ValueError("split results in empty train_data", split_index)
        if split_index >= len(data):
            raise ValueError("split results in empty val_data", len(data))

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_data": train_data,
                "val_data": val_data,
                "vocab_size": vocab_size,
                "stoi": stoi,
                "itos": itos,
            },
            TENSOR_PATH,
        )

        VOCAB_PATH.write_text(
            json.dumps(
                {
                    "chars": list(stoi.keys()),
                    "vocab_size": vocab_size,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        report = {
            "text_path": str(TEXT_PATH),
            "tensor_path": str(TENSOR_PATH),
            "vocab_path": str(VOCAB_PATH),
            "report_path": str(REPORT_PATH),
            "text_sha256": text_sha256,
            "text_characters": len(text),
            "vocab_size": vocab_size,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "train_ratio": TRAIN_RATIO,
            "total_tokens": len(data),
            "train_token_ratio": len(train_data) / len(data) if len(data) else 0,
            "dtype": str(data.dtype),
            "sample_char": sample_char,
            "sample_id": sample_id,
            "sample_decode_ok": decode([sample_id]) == sample_char,
            "token_file_lines": TENSOR_PATH.stat().st_size,
        }
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        io_logger.info(
            "prepared tensors train=%d val=%d vocab=%d report=%s",
            len(train_data),
            len(val_data),
            vocab_size,
            REPORT_PATH,
        )
    except Exception:
        io_logger.exception("prepare training data failed path=%s", TEXT_PATH)
        raise


if __name__ == "__main__":
    main()
