"""Evaluate BPE SFT fairly against the preserved character-token model."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Any

import torch

from bpe_tokenizer import BPETokenizer
from evaluate_sft_hq1000 import score_answer
from train_gpt_stage3 import GPTLanguageModel, set_global_seed


RAW_DATA_PATH = Path("data/sft/sft_balanced_v3.jsonl")
BPE_DATA_PATH = Path("data/bpe/sft_balanced_v3_bpe_tensors.pt")
CHAR_DATA_PATH = Path("data/sft/sft_balanced_v3_tensors.pt")
BPE_TOKENIZER_PATH = Path("data/bpe/tokenizer_v1.json")
PROMPT_PATH = Path("data/prompt10_eval.txt")
REPORT_DIR = Path(os.getenv("BPE_SFT_EVAL_DIR", "reports/milestones/005_bpe_sft"))
MODEL_PATHS = {
    "character_best": Path("checkpoints/sft_balanced_v3_step800_best.pt"),
    "bpe_pre_sft": Path("checkpoints/bpe_sft_init_pre_sft.pt"),
    "bpe_best": Path("checkpoints/bpe_sft_step800_best.pt"),
    "bpe_final": Path("checkpoints/bpe_sft_step800.pt"),
}
SEED = int(os.getenv("BPE_SFT_EVAL_SEED", "42"))
TEMPERATURE = float(os.getenv("BPE_SFT_EVAL_TEMPERATURE", "0.8"))
TOP_K = int(os.getenv("BPE_SFT_EVAL_TOP_K", "20"))
PROMPT_MAX_CHARS = int(os.getenv("BPE_SFT_PROMPT_MAX_CHARS", "30"))
TEST_MAX_CHARS = int(os.getenv("BPE_SFT_TEST_MAX_CHARS", "80"))
DEVICE = torch.device(os.getenv("BPE_SFT_EVAL_DEVICE", "cpu"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_logging() -> dict[str, logging.Logger]:
    Path("logs").mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    result = {}
    for suffix in ("model", "generation", "metrics", "output"):
        logger = logging.getLogger(f"bpe.sft.eval.{suffix}")
        logger.handlers.clear()
        logger.propagate = False
        level = getattr(
            logging,
            os.getenv(f"BPE_SFT_EVAL_{suffix.upper()}_LOG_LEVEL", "INFO").upper(),
            logging.INFO,
        )
        logger.setLevel(level)
        handler = RotatingFileHandler(
            f"logs/bpe_sft_eval_{suffix}.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        if os.getenv("BPE_SFT_EVAL_CONSOLE_LOG", "1") == "1":
            console = logging.StreamHandler()
            console.setLevel(level)
            console.setFormatter(formatter)
            logger.addHandler(console)
        result[suffix] = logger
    return result


def load_model(path: Path) -> tuple[GPTLanguageModel, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    model = GPTLanguageModel(
        vocab_size=int(meta["vocab_size"]),
        embedding_size=int(meta["embedding_dim"]),
        num_heads=int(meta["num_heads"]),
        context_size=int(meta["block_size"]),
        num_layers=int(meta["num_layers"]),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, meta


def prompt_ids(
    question: str,
    payload: dict,
    tokenizer: BPETokenizer | None,
) -> list[int]:
    text_ids = (
        tokenizer.encode(question)
        if tokenizer is not None
        else [payload["stoi"][char] for char in question]
    )
    special = payload["special_token_ids"]
    return [special["<BOS>"], special["<USER>"], *text_ids, special["<ASSISTANT>"]]


@torch.no_grad()
def generate_answer_by_char_limit(
    model: GPTLanguageModel,
    question: str,
    payload: dict,
    tokenizer: BPETokenizer | None,
    max_characters: int,
    seed: int,
) -> dict[str, Any]:
    set_global_seed(seed, deterministic=True)
    current = prompt_ids(question, payload, tokenizer)
    generated: list[int] = []
    special = payload["special_token_ids"]
    forbidden = [
        special["<BOS>"], special["<USER>"],
        special["<ASSISTANT>"], special["<PAD>"],
    ]
    eos_id = special["<EOS>"]
    stopped = False
    safety_tokens = max_characters * 2
    for _ in range(safety_tokens):
        context = current[-model.context_size:]
        tensor = torch.tensor([context], dtype=torch.long, device=DEVICE)
        logits, _ = model(tensor)
        next_logits = logits[:, -1, :] / max(TEMPERATURE, 1e-8)
        next_logits[:, forbidden] = float("-inf")
        k = min(TOP_K, next_logits.shape[-1])
        values, indices = torch.topk(next_logits, k, dim=-1)
        choice = torch.multinomial(torch.softmax(values, dim=-1), 1)
        token_id = int(indices.gather(1, choice)[0, 0])
        if token_id == eos_id:
            stopped = True
            break
        current.append(token_id)
        generated.append(token_id)
        decoded = (
            tokenizer.decode(generated)
            if tokenizer is not None
            else "".join(payload["itos"][token] for token in generated)
        )
        if len(decoded) >= max_characters:
            break
    answer = (
        tokenizer.decode(generated)
        if tokenizer is not None
        else "".join(payload["itos"][token] for token in generated)
    )[:max_characters]
    return {
        "answer": answer,
        "generated_tokens": len(generated),
        "generated_characters": len(answer),
        "stopped_on_eos": stopped,
    }


def summarize(results: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        grouped[result["task_family"]].append(result)

    def metrics(items: list[dict]) -> dict:
        return {
            "count": len(items),
            "exact_match": sum(item["exact_match"] for item in items),
            "contains_gold": sum(item["contains_gold"] for item in items),
            "stopped_on_eos": sum(item["stopped_on_eos"] for item in items),
        }

    return {
        "overall": metrics(results),
        "by_task_family": {
            family: metrics(items) for family, items in sorted(grouped.items())
        },
    }


def escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "↵")


def main() -> None:
    loggers = configure_logging()
    try:
        bpe_payload = torch.load(BPE_DATA_PATH, map_location="cpu", weights_only=False)
        char_payload = torch.load(CHAR_DATA_PATH, map_location="cpu", weights_only=False)
        tokenizer = BPETokenizer.load(BPE_TOKENIZER_PATH)
        raw_records = [
            json.loads(line)
            for line in RAW_DATA_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        test_records = [record for record in raw_records if record["split"] == "test"]
        models = {}
        metadata = {}
        for name, path in MODEL_PATHS.items():
            model, meta = load_model(path)
            models[name] = model
            metadata[name] = {
                "path": str(path), "sha256": sha256(path),
                "sft_steps": int(meta.get("sft_steps", 0)),
                "tokenizer": "character" if name == "character_best" else "bpe",
            }
            loggers["model"].info("loaded name=%s path=%s", name, path)

        prompts = [line for line in PROMPT_PATH.read_text(encoding="utf-8").splitlines() if line]
        comparisons = []
        for index, question in enumerate(prompts[:10]):
            row = {"question": question}
            for name, model in models.items():
                is_char = name == "character_best"
                row[name] = generate_answer_by_char_limit(
                    model, question,
                    char_payload if is_char else bpe_payload,
                    None if is_char else tokenizer,
                    PROMPT_MAX_CHARS, SEED + index,
                )
            comparisons.append(row)

        evaluations = {}
        for name in ("character_best", "bpe_best"):
            is_char = name == "character_best"
            results = []
            for index, record in enumerate(test_records):
                generated = generate_answer_by_char_limit(
                    models[name], record["question"],
                    char_payload if is_char else bpe_payload,
                    None if is_char else tokenizer,
                    TEST_MAX_CHARS, SEED + 10_000 + index,
                )
                results.append({
                    "id": record["id"],
                    "question": record["question"],
                    "gold_answer": record["answer"],
                    "task_family": record["task_family"],
                    **generated,
                    **score_answer(generated["answer"], record["answer"]),
                })
            evaluations[name] = {"metrics": summarize(results), "results": results}
            loggers["metrics"].info(
                "model=%s metrics=%s", name,
                json.dumps(evaluations[name]["metrics"]["overall"], ensure_ascii=False),
            )

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "stage": "bpe_sft_evaluation",
            "fairness": "same questions, seeds, temperature, top-k and character output limits",
            "models": metadata,
            "prompt10": comparisons,
            "held_out_test": evaluations,
        }
        report_path = REPORT_DIR / "bpe_sft_evaluation.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        table_lines = [
            "# 固定10题：字符模型与BPE模型对比", "",
            "每个输出最多30个字符；`↵` 表示换行。", "",
            "| # | 问题 | 字符SFT最佳 | BPE的SFT前 | BPE的SFT最佳 | BPE的SFT最终 |",
            "|---:|---|---|---|---|---|",
        ]
        for index, row in enumerate(comparisons, 1):
            table_lines.append(
                f"| {index} | {escape(row['question'])} | "
                f"{escape(row['character_best']['answer'])} | "
                f"{escape(row['bpe_pre_sft']['answer'])} | "
                f"{escape(row['bpe_best']['answer'])} | "
                f"{escape(row['bpe_final']['answer'])} |"
            )
        table_path = REPORT_DIR / "prompt10_character_vs_bpe.md"
        table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
        loggers["output"].info("wrote report=%s table=%s", report_path, table_path)
    except Exception:
        loggers["output"].exception("BPE SFT evaluation failed")
        raise


if __name__ == "__main__":
    main()
