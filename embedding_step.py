from pathlib import Path
import json
import logging
import os
from logging.handlers import RotatingFileHandler

import torch

TENSOR_PATH = Path(
    os.getenv("TRAIN_TENSOR_PATH", "data/clean/doupo_stage3_tensors.pt")
)
REPORT_PATH = Path(
    os.getenv("EMBEDDING_REPORT_PATH", "data/clean/embedding_report.json")
)
LOG_PATH = Path("logs/embedding_step.log")
BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "64"))


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

    root_logger = logging.getLogger("embedding_step")
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.propagate = False

    io_level = os.getenv("EMBEDDING_IO_LOG_LEVEL", "INFO").upper()
    transform_level = os.getenv("EMBEDDING_TRANSFORM_LOG_LEVEL", "INFO").upper()
    logging.getLogger("embedding_step.io").setLevel(io_level)
    logging.getLogger("embedding_step.transform").setLevel(transform_level)


def decode_from_file(token_ids, payload):
    # helper for readable text preview
    return "".join(payload[token_id] for token_id in token_ids)


def get_batch(data, batch_size, block_size):
    max_start = len(data) - block_size
    if max_start <= 0:
        raise ValueError("data is too short for configured block size", max_start)

    indices = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in indices])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in indices])

    return x, y


def main():
    configure_logging()
    io_logger = logging.getLogger("embedding_step.io")
    transform_logger = logging.getLogger("embedding_step.transform")

    try:
        io_logger.info("loading tensor bundle=%s", TENSOR_PATH)
        payload = torch.load(TENSOR_PATH)
        if isinstance(payload, dict):
            train_data = payload.get("train_data")
            val_data = payload.get("val_data")
            vocab_size = payload.get("vocab_size")
            stoi = payload.get("stoi")
            itos = payload.get("itos")
        else:
            raise ValueError("unexpected tensor file format")

        if train_data is None or val_data is None:
            raise ValueError("missing train_data/val_data in tensor file")
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError("invalid vocab_size", vocab_size)
        if not isinstance(stoi, dict) or not isinstance(itos, dict):
            raise ValueError("missing vocab mapping in tensor file")

        payload_chars = itos

        io_logger.info(
            "loaded train=%d val=%d vocab=%d block=%d batch=%d emb=%d",
            len(train_data),
            len(val_data),
            vocab_size,
            BLOCK_SIZE,
            BATCH_SIZE,
            EMBEDDING_DIM,
        )

        train_x, train_y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
        io_logger.debug("sample batch token shape=%s", tuple(train_x.shape))

        token_embedding = torch.nn.Embedding(vocab_size, EMBEDDING_DIM)
        position_embedding = torch.nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)

        token_vectors = token_embedding(train_x)
        position_ids = torch.arange(BLOCK_SIZE)
        position_vectors = position_embedding(position_ids)
        contextual_vectors = token_vectors + position_vectors

        print("Batch 输入 token 形状：", train_x.shape)
        print("Batch 目标 token 形状：", train_y.shape)
        print("Token Embedding 参数形状：", token_embedding.weight.shape)
        print("Position Embedding 参数形状：", position_embedding.weight.shape)
        print("Token 向量形状：", token_vectors.shape)
        print("位置向量形状：", position_vectors.shape)
        print("加法后上下文向量形状：", contextual_vectors.shape)

        sample_ids = train_x[0].tolist()
        print("样例位置 token：", sample_ids)
        print("样例位置文字：", decode_from_file(sample_ids, payload_chars))
        print("样例 token 1 向量前 5 维：", token_vectors[0, 0, :5].tolist())
        print("样例位置 0 的位置向量前 5 维：", position_vectors[0, :5].tolist())
        print("样例位置 0 的合并向量前 5 维：", contextual_vectors[0, 0, :5].tolist())

        sample_position_preview = contextual_vectors[0, :BLOCK_SIZE, :5]
        preview_sum = float(sample_position_preview.abs().sum().item())

        report = {
            "tensor_path": str(TENSOR_PATH),
            "report_path": str(REPORT_PATH),
            "train_size": int(len(train_data)),
            "val_size": int(len(val_data)),
            "vocab_size": int(vocab_size),
            "block_size": BLOCK_SIZE,
            "batch_size": BATCH_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "batch_input_shape": [int(x) for x in train_x.shape],
            "token_embedding_shape": list(token_embedding.weight.shape),
            "position_embedding_shape": list(position_embedding.weight.shape),
            "token_vectors_shape": list(token_vectors.shape),
            "position_vectors_shape": list(position_vectors.shape),
            "contextual_vectors_shape": list(contextual_vectors.shape),
            "contextual_abs_sum_sample": preview_sum,
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

        transform_logger.info(
            "embedding step done report=%s sample_context_sum=%.6f",
            REPORT_PATH,
            preview_sum,
        )
    except Exception:
        io_logger.exception("embedding step failed input=%s", TENSOR_PATH)
        raise


if __name__ == "__main__":
    main()
