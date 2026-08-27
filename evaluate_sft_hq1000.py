from collections import Counter
from pathlib import Path
from typing import Any
import json
import logging
import os

import torch
from logging.handlers import RotatingFileHandler

from evaluate_sft_baseline import encode_chat_prompt, generate_answer
from prepare_sft_data import load_jsonl, parse_log_level, sha256_file
from train_gpt_stage3 import GPTLanguageModel, set_global_seed


DATA_PATH = Path("data/sft/sft_hq1000_v2.jsonl")
TENSOR_PATH = Path("data/sft/sft_hq1000_v2_tensors.pt")
PROMPT_PATH = Path("data/prompt10_eval.txt")
PRE_SFT_PATH = Path("checkpoints/archive/sft_stage1_init_pre_sft.pt")
BEST_PATH = Path("checkpoints/sft_hq1000_step800_best.pt")
FINAL_PATH = Path("checkpoints/sft_hq1000_step800.pt")
REPORT_DIR = Path("reports/milestones/003f_sft_hq1000_step800")
REPORT_PATH = REPORT_DIR / "sft_hq1000_evaluation.json"
TABLE_PATH = REPORT_DIR / "prompt10_comparison.md"
SEED = int(os.getenv("SFT_EVAL_SEED", "42"))
MAX_NEW_TOKENS = int(os.getenv("SFT_EVAL_MAX_NEW_TOKENS", "30"))
TEMPERATURE = float(os.getenv("SFT_EVAL_TEMPERATURE", "0.8"))
TOP_K = int(os.getenv("SFT_EVAL_TOP_K", "20"))
DEVICE = torch.device(os.getenv("SFT_EVAL_DEVICE", "cpu"))


def configure_logger(name: str, file_name: str, env_name: str) -> logging.Logger:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(parse_log_level(os.getenv(env_name, "INFO")))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler = RotatingFileHandler(
        Path("logs") / file_name,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if os.getenv("SFT_EVAL_CONSOLE_LOG", "1") == "1":
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    logger.propagate = False
    return logger


def configure_logging() -> dict[str, logging.Logger]:
    return {
        "model": configure_logger(
            "sft.hq.eval.model", "sft_hq_eval_model.log", "SFT_EVAL_MODEL_LOG_LEVEL"
        ),
        "generation": configure_logger(
            "sft.hq.eval.generation",
            "sft_hq_eval_generation.log",
            "SFT_EVAL_GENERATION_LOG_LEVEL",
        ),
        "metrics": configure_logger(
            "sft.hq.eval.metrics",
            "sft_hq_eval_metrics.log",
            "SFT_EVAL_METRICS_LOG_LEVEL",
        ),
        "output": configure_logger(
            "sft.hq.eval.output", "sft_hq_eval_output.log", "SFT_EVAL_OUTPUT_LOG_LEVEL"
        ),
    }


def load_model(path: Path) -> tuple[GPTLanguageModel, dict[str, Any]]:
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


def generate_for_question(
    model: GPTLanguageModel,
    question: str,
    payload: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    prompt_ids = encode_chat_prompt(
        question, payload["stoi"], payload["special_token_ids"]
    )
    answer, generated_ids, stopped_on_eos = generate_answer(
        model=model,
        prompt_ids=prompt_ids,
        itos=payload["itos"],
        special_token_ids=payload["special_token_ids"],
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        seed=seed,
    )
    return {
        "answer": answer,
        "generated_token_count": len(generated_ids),
        "stopped_on_eos": stopped_on_eos,
    }


def score_answer(generated: str, gold: str) -> dict[str, bool]:
    generated_clean = generated.strip()
    gold_clean = gold.strip()
    gold_core = gold_clean.rstrip("。！？!?；;")
    return {
        "exact_match": generated_clean == gold_clean,
        "contains_gold": (
            generated_clean == gold_clean
            if len(gold_core) < 2
            else gold_core in generated_clean
        ),
    }


def evaluate_held_out_test(
    model: GPTLanguageModel,
    test_records: list[dict[str, Any]],
    payload: dict[str, Any],
    logger: logging.Logger,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    results = []
    for index, record in enumerate(test_records):
        generated = generate_for_question(
            model, record["question"], payload, SEED + 10_000 + index
        )
        scores = score_answer(generated["answer"], record["answer"])
        result = {
            "id": record["id"],
            "question": record["question"],
            "gold_answer": record["answer"],
            "generation_method": record["generation_method"],
            "concept_category": record.get("concept_category"),
            **generated,
            **scores,
        }
        results.append(result)
        logger.info(
            "test id=%s tokens=%d eos=%s exact=%s contains=%s",
            record["id"],
            generated["generated_token_count"],
            generated["stopped_on_eos"],
            scores["exact_match"],
            scores["contains_gold"],
        )
    metrics = Counter()
    metrics["count"] = len(results)
    metrics["exact_match"] = sum(item["exact_match"] for item in results)
    metrics["contains_gold"] = sum(item["contains_gold"] for item in results)
    metrics["stopped_on_eos"] = sum(item["stopped_on_eos"] for item in results)
    return dict(metrics), results


def summarize_by_field(
    results: list[dict[str, Any]], field: str
) -> dict[str, dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        key = str(result.get(field) or "not_applicable")
        groups.setdefault(key, []).append(result)
    return {
        key: {
            "count": len(items),
            "exact_match": sum(item["exact_match"] for item in items),
            "contains_gold": sum(item["contains_gold"] for item in items),
            "stopped_on_eos": sum(item["stopped_on_eos"] for item in items),
        }
        for key, items in sorted(groups.items())
    }


def escape_table(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "↵")


def write_prompt_table(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 固定10题：SFT前后输出对比",
        "",
        "每个答案最多生成30个字符；`↵` 代表模型生成的换行符。",
        "",
        "| # | 问题 | SFT前 | 最佳模型（300步） | 最终模型（800步） |",
        "|---:|---|---|---|---|",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | {escape_table(row['question'])} "
            f"| {escape_table(row['pre_sft']['answer'])} "
            f"| {escape_table(row['best_300']['answer'])} "
            f"| {escape_table(row['final_800']['answer'])} |"
        )
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    loggers = configure_logging()
    set_global_seed(SEED, deterministic=True)
    try:
        payload = torch.load(TENSOR_PATH, map_location="cpu", weights_only=False)
        models = {}
        metadata = {}
        for name, path in {
            "pre_sft": PRE_SFT_PATH,
            "best_300": BEST_PATH,
            "final_800": FINAL_PATH,
        }.items():
            model, meta = load_model(path)
            models[name] = model
            metadata[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "sft_steps": int(meta.get("sft_steps", 0)),
            }
            loggers["model"].info(
                "loaded name=%s step=%d path=%s",
                name,
                metadata[name]["sft_steps"],
                path,
            )

        prompts = [
            line.strip()
            for line in PROMPT_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(prompts) != 10:
            raise ValueError(f"expected 10 fixed prompts, found {len(prompts)}")

        comparisons = []
        for index, question in enumerate(prompts):
            row = {"question": question}
            for model_index, (name, model) in enumerate(models.items()):
                row[name] = generate_for_question(
                    model, question, payload, SEED + index + model_index * 1_000
                )
            comparisons.append(row)
            loggers["generation"].info("prompt10 index=%d complete", index + 1)

        all_records = load_jsonl(DATA_PATH)
        test_records = [record for record in all_records if record["split"] == "test"]
        if len(test_records) != 100:
            raise ValueError(f"expected 100 held-out tests, found {len(test_records)}")
        test_metrics, test_results = evaluate_held_out_test(
            models["best_300"], test_records, payload, loggers["generation"]
        )
        test_breakdown = {
            "generation_method": summarize_by_field(
                test_results, "generation_method"
            ),
            "concept_category": summarize_by_field(
                test_results, "concept_category"
            ),
        }
        loggers["metrics"].info(
            "held-out count=%d exact=%d contains=%d eos=%d",
            test_metrics["count"],
            test_metrics["exact_match"],
            test_metrics["contains_gold"],
            test_metrics["stopped_on_eos"],
        )

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "milestone": "M003f",
            "generation_config": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "seed": SEED,
            },
            "models": metadata,
            "prompt10_comparisons": comparisons,
            "held_out_test_model": "best_300",
            "held_out_test_consumed_once": True,
            "held_out_test_metrics": test_metrics,
            "held_out_test_breakdown": test_breakdown,
            "held_out_test_results": test_results,
        }
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_prompt_table(comparisons)
        loggers["output"].info(
            "evaluation saved report=%s table=%s", REPORT_PATH, TABLE_PATH
        )
    except Exception:
        loggers["output"].exception("SFT HQ1000 evaluation failed")
        raise


if __name__ == "__main__":
    main()
