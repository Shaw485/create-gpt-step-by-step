from collections import Counter
from pathlib import Path
from typing import Any
import json
import logging
import os

import torch
from logging.handlers import RotatingFileHandler

from prepare_sft_data import load_jsonl, parse_log_level, sha256_file
from train_gpt_stage3 import GPTLanguageModel, set_global_seed


PRETRAIN_CHECKPOINT = Path(
    "checkpoints/archive/gpt_stage5_pretrain_step10000_best.pt"
)
SFT_DATA_PATH = Path("data/sft/sft_pilot50_v1.jsonl")
SFT_TENSOR_PATH = Path("data/sft/sft_pilot50_v1_tensors.pt")
INIT_CHECKPOINT = Path("checkpoints/archive/sft_stage1_init_pre_sft.pt")
REPORT_PATH = Path(
    "reports/milestones/002b_pre_sft_baseline/pre_sft_baseline.json"
)
SEED = int(os.getenv("SFT_BASELINE_SEED", "42"))
MAX_NEW_TOKENS = int(os.getenv("SFT_BASELINE_MAX_NEW_TOKENS", "30"))
TEMPERATURE = float(os.getenv("SFT_BASELINE_TEMPERATURE", "0.8"))
TOP_K = int(os.getenv("SFT_BASELINE_TOP_K", "20"))
DEVICE = torch.device(os.getenv("SFT_BASELINE_DEVICE", "cpu"))


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
    if os.getenv("SFT_BASELINE_CONSOLE_LOG", "1") == "1":
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    logger.propagate = False
    return logger


def configure_logging() -> dict[str, logging.Logger]:
    return {
        "model": configure_logger(
            "sft_baseline.model",
            "sft_baseline_model.log",
            "SFT_BASELINE_MODEL_LOG_LEVEL",
        ),
        "generation": configure_logger(
            "sft_baseline.generation",
            "sft_baseline_generation.log",
            "SFT_BASELINE_GENERATION_LOG_LEVEL",
        ),
        "output": configure_logger(
            "sft_baseline.output",
            "sft_baseline_output.log",
            "SFT_BASELINE_OUTPUT_LOG_LEVEL",
        ),
    }


def build_model_from_meta(meta: dict[str, Any], vocab_size: int) -> GPTLanguageModel:
    return GPTLanguageModel(
        vocab_size=vocab_size,
        embedding_size=int(meta["embedding_dim"]),
        num_heads=int(meta["num_heads"]),
        context_size=int(meta["block_size"]),
        num_layers=int(meta["num_layers"]),
    )


def expand_pretrained_model(
    checkpoint: dict[str, Any], extended_vocab_size: int
) -> GPTLanguageModel:
    meta = checkpoint["meta"]
    base_vocab_size = int(meta["vocab_size"])
    if extended_vocab_size <= base_vocab_size:
        raise ValueError("extended vocabulary must be larger than base vocabulary")

    model = build_model_from_meta(meta, extended_vocab_size)
    old_state = checkpoint["model_state_dict"]
    new_state = model.state_dict()
    expandable_keys = {
        "token_embedding.weight",
        "head.weight",
        "head.bias",
    }

    for key, old_tensor in old_state.items():
        new_tensor = new_state[key]
        if new_tensor.shape == old_tensor.shape:
            new_tensor.copy_(old_tensor)
        elif key in expandable_keys and new_tensor.shape[0] == extended_vocab_size:
            new_tensor[:base_vocab_size].copy_(old_tensor)
        else:
            raise ValueError(
                f"cannot transfer parameter {key}: {old_tensor.shape} -> {new_tensor.shape}"
            )

    model.load_state_dict(new_state)
    return model


def verify_pretrained_weights_copied(
    model: GPTLanguageModel, checkpoint: dict[str, Any]
) -> None:
    old_state = checkpoint["model_state_dict"]
    new_state = model.state_dict()
    base_vocab_size = int(checkpoint["meta"]["vocab_size"])
    for key, old_tensor in old_state.items():
        new_tensor = new_state[key]
        comparable = (
            new_tensor[:base_vocab_size]
            if key in {"token_embedding.weight", "head.weight", "head.bias"}
            else new_tensor
        )
        if not torch.equal(comparable.cpu(), old_tensor.cpu()):
            raise RuntimeError(f"pretrained parameter copy mismatch: {key}")


def encode_chat_prompt(
    question: str, stoi: dict[str, int], special_token_ids: dict[str, int]
) -> list[int]:
    missing_chars = sorted(set(question) - set(stoi))
    if missing_chars:
        raise ValueError(f"question has out-of-vocabulary chars: {missing_chars}")
    return [
        special_token_ids["<BOS>"],
        special_token_ids["<USER>"],
        *[stoi[char] for char in question],
        special_token_ids["<ASSISTANT>"],
    ]


@torch.no_grad()
def generate_answer(
    model: GPTLanguageModel,
    prompt_ids: list[int],
    itos: dict[int, str],
    special_token_ids: dict[str, int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> tuple[str, list[int], bool]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    torch.manual_seed(seed)
    model.eval()
    current_ids = prompt_ids.copy()
    generated_ids: list[int] = []
    eos_id = special_token_ids["<EOS>"]
    forbidden_ids = [
        special_token_ids["<BOS>"],
        special_token_ids["<USER>"],
        special_token_ids["<ASSISTANT>"],
        special_token_ids["<PAD>"],
    ]
    stopped_on_eos = False

    for _ in range(max_new_tokens):
        context = current_ids[-model.context_size :]
        input_tensor = torch.tensor([context], dtype=torch.long, device=DEVICE)
        logits, _ = model(input_tensor)
        next_logits = logits[:, -1, :] / temperature
        next_logits[:, forbidden_ids] = float("-inf")

        effective_top_k = min(top_k, next_logits.shape[-1])
        if effective_top_k > 0:
            values, indices = torch.topk(next_logits, effective_top_k, dim=-1)
            probabilities = torch.softmax(values, dim=-1)
            sampled_index = torch.multinomial(probabilities, num_samples=1)
            next_id = int(indices.gather(1, sampled_index)[0, 0])
        else:
            probabilities = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probabilities, num_samples=1)[0, 0])

        if next_id == eos_id:
            stopped_on_eos = True
            break
        current_ids.append(next_id)
        generated_ids.append(next_id)

    answer = "".join(itos[token_id] for token_id in generated_ids)
    return answer, generated_ids, stopped_on_eos


def save_initialized_checkpoint(
    model: GPTLanguageModel,
    pretrained_checkpoint: dict[str, Any],
    extended_vocab_size: int,
    special_token_ids: dict[str, int],
) -> None:
    INIT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(pretrained_checkpoint["meta"])
    meta.update(
        {
            "vocab_size": extended_vocab_size,
            "base_vocab_size": pretrained_checkpoint["meta"]["vocab_size"],
            "special_token_ids": special_token_ids,
            "sft_steps": 0,
            "stage": "pre_sft_initialized",
            "source_checkpoint": str(PRETRAIN_CHECKPOINT),
            "source_checkpoint_sha256": sha256_file(PRETRAIN_CHECKPOINT),
        }
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "meta": meta,
            "loss_history": [],
        },
        INIT_CHECKPOINT,
    )


def main() -> None:
    loggers = configure_logging()
    set_global_seed(SEED, deterministic=True)
    try:
        sft_payload = torch.load(SFT_TENSOR_PATH, map_location="cpu", weights_only=False)
        pretrained = torch.load(
            PRETRAIN_CHECKPOINT, map_location="cpu", weights_only=False
        )
        base_vocab_size = int(sft_payload["base_vocab_size"])
        extended_vocab_size = int(sft_payload["vocab_size"])
        if base_vocab_size != int(pretrained["meta"]["vocab_size"]):
            raise ValueError("SFT and pretrained base vocabularies do not match")

        model = expand_pretrained_model(pretrained, extended_vocab_size).to(DEVICE)
        verify_pretrained_weights_copied(model, pretrained)
        save_initialized_checkpoint(
            model,
            pretrained,
            extended_vocab_size,
            sft_payload["special_token_ids"],
        )
        loggers["model"].info(
            "initialized pre-SFT model base_vocab=%d extended_vocab=%d params=%d device=%s",
            base_vocab_size,
            extended_vocab_size,
            sum(parameter.numel() for parameter in model.parameters()),
            DEVICE,
        )

        test_records = [
            record for record in load_jsonl(SFT_DATA_PATH) if record["split"] == "test"
        ]
        results = []
        for index, record in enumerate(test_records):
            prompt_ids = encode_chat_prompt(
                record["question"],
                sft_payload["stoi"],
                sft_payload["special_token_ids"],
            )
            answer, generated_ids, stopped_on_eos = generate_answer(
                model=model,
                prompt_ids=prompt_ids,
                itos=sft_payload["itos"],
                special_token_ids=sft_payload["special_token_ids"],
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                seed=SEED + index,
            )
            gold = record["answer"]
            results.append(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "gold_answer": gold,
                    "generated_answer": answer,
                    "generated_token_ids": generated_ids,
                    "generated_token_count": len(generated_ids),
                    "stopped_on_eos": stopped_on_eos,
                    "exact_match": answer.strip() == gold.strip(),
                    "contains_gold": gold.rstrip("。") in answer,
                }
            )
            loggers["generation"].info(
                "baseline generated id=%s tokens=%d eos=%s",
                record["id"],
                len(generated_ids),
                stopped_on_eos,
            )

        metrics = Counter()
        metrics["exact_match"] = sum(result["exact_match"] for result in results)
        metrics["contains_gold"] = sum(result["contains_gold"] for result in results)
        metrics["stopped_on_eos"] = sum(result["stopped_on_eos"] for result in results)
        report = {
            "milestone": "M002b",
            "stage": "pre_sft_baseline",
            "pretrained_step": pretrained["meta"]["step"],
            "sft_steps": 0,
            "parameter_updates": 0,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "base_vocab_size": base_vocab_size,
            "extended_vocab_size": extended_vocab_size,
            "generation_config": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "seed": SEED,
            },
            "metrics": dict(metrics),
            "test_count": len(results),
            "results": results,
            "source_checkpoint": str(PRETRAIN_CHECKPOINT),
            "source_checkpoint_sha256": sha256_file(PRETRAIN_CHECKPOINT),
            "initialized_checkpoint": str(INIT_CHECKPOINT),
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        loggers["output"].info(
            "baseline saved report=%s exact=%d/%d contains=%d/%d eos=%d/%d",
            REPORT_PATH,
            metrics["exact_match"],
            len(results),
            metrics["contains_gold"],
            len(results),
            metrics["stopped_on_eos"],
            len(results),
        )
    except Exception:
        loggers["output"].exception("pre-SFT baseline failed")
        raise


if __name__ == "__main__":
    main()
