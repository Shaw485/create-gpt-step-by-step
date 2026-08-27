from collections import Counter
from pathlib import Path
from typing import Any
import json
import logging
import os

import torch
from logging.handlers import RotatingFileHandler

from evaluate_sft_baseline import encode_chat_prompt, generate_answer
from evaluate_sft_hq1000 import score_answer
from prepare_sft_data import load_jsonl, parse_log_level, sha256_file
from train_gpt_stage3 import GPTLanguageModel, set_global_seed


DATA_PATH = Path("data/sft/sft_balanced_v3.jsonl")
TENSOR_PATH = Path("data/sft/sft_balanced_v3_tensors.pt")
PROMPT_PATH = Path("data/prompt10_eval.txt")
REPORT_DIR = Path("reports/milestones/003i_sft_balanced_v3_step800")
REPORT_PATH = REPORT_DIR / "sft_balanced_v3_evaluation.json"
TABLE_PATH = REPORT_DIR / "prompt10_balanced_comparison.md"
MODEL_PATHS = {
    "pre_sft": Path("checkpoints/archive/sft_stage1_init_pre_sft.pt"),
    "old_best_300": Path("checkpoints/sft_hq1000_step800_best.pt"),
    "balanced_best_400": Path("checkpoints/sft_balanced_v3_step800_best.pt"),
    "balanced_final_800": Path("checkpoints/sft_balanced_v3_step800.pt"),
}
SEED = int(os.getenv("SFT_BALANCED_EVAL_SEED", "42"))
PROMPT_MAX_NEW_TOKENS = int(
    os.getenv("SFT_BALANCED_PROMPT_MAX_NEW_TOKENS", "30")
)
TEST_MAX_NEW_TOKENS = int(
    os.getenv("SFT_BALANCED_TEST_MAX_NEW_TOKENS", "80")
)
TEMPERATURE = float(os.getenv("SFT_BALANCED_EVAL_TEMPERATURE", "0.8"))
TOP_K = int(os.getenv("SFT_BALANCED_EVAL_TOP_K", "20"))
DEVICE = torch.device(os.getenv("SFT_BALANCED_EVAL_DEVICE", "cpu"))


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
    if os.getenv("SFT_BALANCED_EVAL_CONSOLE_LOG", "1") == "1":
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    logger.propagate = False
    return logger


def configure_logging() -> dict[str, logging.Logger]:
    return {
        "model": configure_logger(
            "sft.balanced.eval.model",
            "sft_balanced_eval_model.log",
            "SFT_BALANCED_EVAL_MODEL_LOG_LEVEL",
        ),
        "generation": configure_logger(
            "sft.balanced.eval.generation",
            "sft_balanced_eval_generation.log",
            "SFT_BALANCED_EVAL_GENERATION_LOG_LEVEL",
        ),
        "metrics": configure_logger(
            "sft.balanced.eval.metrics",
            "sft_balanced_eval_metrics.log",
            "SFT_BALANCED_EVAL_METRICS_LOG_LEVEL",
        ),
        "output": configure_logger(
            "sft.balanced.eval.output",
            "sft_balanced_eval_output.log",
            "SFT_BALANCED_EVAL_OUTPUT_LOG_LEVEL",
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


def generate_question(
    model: GPTLanguageModel,
    question: str,
    payload: dict[str, Any],
    max_new_tokens: int,
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
        max_new_tokens=max_new_tokens,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        seed=seed,
    )
    return {
        "answer": answer,
        "generated_token_count": len(generated_ids),
        "stopped_on_eos": stopped_on_eos,
    }


def summarize_results(
    results: list[dict[str, Any]], field: str | None = None
) -> dict[str, Any]:
    if field is None:
        groups = {"overall": results}
    else:
        grouped = {}
        for result in results:
            grouped.setdefault(str(result.get(field) or "not_applicable"), []).append(
                result
            )
        groups = dict(sorted(grouped.items()))
    return {
        key: {
            "count": len(items),
            "exact_match": sum(item["exact_match"] for item in items),
            "contains_gold": sum(item["contains_gold"] for item in items),
            "stopped_on_eos": sum(item["stopped_on_eos"] for item in items),
        }
        for key, items in groups.items()
    }


def escape_table(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "↵")


def write_comparison_table(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 固定10题：预训练、旧SFT和平衡SFT对比",
        "",
        "每个答案最多生成30个字符；`↵` 表示模型生成的换行。",
        "",
        "| # | 问题 | SFT前 | 旧版最佳300步 | 平衡版最佳400步 | 平衡版最终800步 |",
        "|---:|---|---|---|---|---|",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | {escape_table(row['question'])} "
            f"| {escape_table(row['pre_sft']['answer'])} "
            f"| {escape_table(row['old_best_300']['answer'])} "
            f"| {escape_table(row['balanced_best_400']['answer'])} "
            f"| {escape_table(row['balanced_final_800']['answer'])} |"
        )
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_test_model(
    model_name: str,
    model: GPTLanguageModel,
    test_records: list[dict[str, Any]],
    payload: dict[str, Any],
    logger: logging.Logger,
    seed_offset: int,
) -> dict[str, Any]:
    results = []
    for index, record in enumerate(test_records):
        generated = generate_question(
            model,
            record["question"],
            payload,
            TEST_MAX_NEW_TOKENS,
            SEED + seed_offset + index,
        )
        scores = score_answer(generated["answer"], record["answer"])
        results.append(
            {
                "id": record["id"],
                "question": record["question"],
                "gold_answer": record["answer"],
                "task_family": record["task_family"],
                "generation_method": record["generation_method"],
                **generated,
                **scores,
            }
        )
        logger.info(
            "model=%s test id=%s family=%s exact=%s eos=%s",
            model_name,
            record["id"],
            record["task_family"],
            scores["exact_match"],
            generated["stopped_on_eos"],
        )
    return {
        "metrics": {
            "overall": summarize_results(results)["overall"],
            "by_task_family": summarize_results(results, "task_family"),
            "by_generation_method": summarize_results(results, "generation_method"),
        },
        "results": results,
    }


def main() -> None:
    loggers = configure_logging()
    set_global_seed(SEED, deterministic=True)
    try:
        payload = torch.load(TENSOR_PATH, map_location="cpu", weights_only=False)
        models = {}
        model_metadata = {}
        for name, path in MODEL_PATHS.items():
            model, meta = load_model(path)
            models[name] = model
            model_metadata[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "sft_steps": int(meta.get("sft_steps", 0)),
            }
            loggers["model"].info(
                "loaded name=%s step=%d path=%s",
                name,
                model_metadata[name]["sft_steps"],
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
        for prompt_index, question in enumerate(prompts):
            row = {"question": question}
            for name, model in models.items():
                row[name] = generate_question(
                    model,
                    question,
                    payload,
                    PROMPT_MAX_NEW_TOKENS,
                    SEED + prompt_index,
                )
            comparisons.append(row)
            loggers["generation"].info("fixed prompt=%d complete", prompt_index + 1)

        test_records = [
            record for record in load_jsonl(DATA_PATH) if record["split"] == "test"
        ]
        if len(test_records) != 100:
            raise ValueError(f"expected 100 held-out records, found {len(test_records)}")
        test_evaluations = {}
        for model_name in ("old_best_300", "balanced_best_400"):
            evaluation = evaluate_test_model(
                model_name,
                models[model_name],
                test_records,
                payload,
                loggers["generation"],
                10_000,
            )
            test_evaluations[model_name] = evaluation
            overall = evaluation["metrics"]["overall"]
            loggers["metrics"].info(
                "model=%s count=%d exact=%d contains=%d eos=%d",
                model_name,
                overall["count"],
                overall["exact_match"],
                overall["contains_gold"],
                overall["stopped_on_eos"],
            )

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "milestone": "M003i",
            "generation_config": {
                "fixed_prompt_max_new_tokens": PROMPT_MAX_NEW_TOKENS,
                "held_out_test_max_new_tokens": TEST_MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "seed": SEED,
            },
            "models": model_metadata,
            "prompt10_comparisons": comparisons,
            "held_out_test_models": ["old_best_300", "balanced_best_400"],
            "held_out_test_consumed_after_model_selection": True,
            "held_out_test_evaluations": test_evaluations,
        }
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_comparison_table(comparisons)
        loggers["output"].info(
            "saved report=%s comparison=%s", REPORT_PATH, TABLE_PATH
        )
    except Exception:
        loggers["output"].exception("balanced SFT evaluation failed")
        raise


if __name__ == "__main__":
    main()
