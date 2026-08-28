"""Evaluate the validation-selected v4 pretraining checkpoint.

This script intentionally evaluates only the checkpoint selected by validation
loss.  It writes an immutable-style JSON report and fixed-prompt samples so the
pretraining result can later be compared with SFT without changing the protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from bpe_tokenizer import BPETokenizer
from gpt_model_v4 import GPTConfig, GPTLanguageModelV4
from train_pretrain_v4 import (
    bits_per_character,
    evaluate,
    generate_sample,
    load_config,
    load_tensor,
    select_device,
)
from training_runtime import (
    atomic_write_json,
    canonical_json_sha256,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    load_checkpoint,
)


def _read_prompts(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [prompt for prompt in prompts if prompt][:10]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local_m4_8m.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompts", type=Path, default=Path("data/prompt10_eval.txt"))
    parser.add_argument("--sample-max-characters", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Keep the test split sealed and evaluate only fixed generation prompts.",
    )
    parser.add_argument(
        "--selection-rule",
        default="minimum validation loss; test split was not used for selection",
        help="Human-readable rule that selected the supplied checkpoint.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(config["run_dir"])
    checkpoint_path = args.checkpoint or run_dir / "best.pt"
    output_path = args.output or run_dir / "selected_model_evaluation.json"
    run_id = generate_run_id("local-v4-selected-eval")
    loggers = configure_module_loggers(
        run_dir / "logs",
        run_id,
        config["logging"]["module_levels"],
        max_bytes=int(config["logging"]["max_bytes"]),
        backup_count=int(config["logging"]["backup_count"]),
        console=bool(config["logging"]["console"]),
    )

    device = select_device(config["device"])
    data_dir = Path(config["data_dir"])
    tokenizer = BPETokenizer.load(data_dir / "tokenizer.json")
    test_data = None if args.skip_test else load_tensor(data_dir / "test_tokens.pt")
    manifest = json.loads((data_dir / "token_manifest.json").read_text(encoding="utf-8"))
    model_config = GPTConfig(vocab_size=tokenizer.vocab_size, **config["model"])
    model = GPTLanguageModelV4(model_config)

    config_hash = canonical_json_sha256(config)
    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location="cpu",
        expected_config_sha256=config_hash,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    loggers["checkpoint"].info(
        "selected checkpoint loaded and checksum verified",
        extra={
            "context": {
                "path": str(checkpoint_path),
                "step": checkpoint["step"],
                "sha256": file_sha256(checkpoint_path),
            }
        },
    )

    settings: dict[str, Any] = dict(config["training"])
    generation: dict[str, Any] = dict(config["generation"])
    if args.sample_max_characters is not None:
        settings["sample_max_characters"] = args.sample_max_characters
    if args.temperature is not None:
        generation["temperature"] = args.temperature
    if args.top_k is not None:
        generation["top_k"] = args.top_k
    if args.max_new_tokens is not None:
        generation["max_new_tokens"] = args.max_new_tokens
    test_generator = torch.Generator().manual_seed(int(config["seed"]) + 4001)
    test_loss = (
        evaluate(model, test_data, settings, test_generator, device)
        if test_data is not None
        else None
    )
    test_bpc = (
        bits_per_character(
            test_loss,
            int(manifest["splits"]["test"]["tokens"]),
            int(manifest["splits"]["test"]["characters"]),
        )
        if test_loss is not None
        else None
    )
    loggers["validation"].info(
        "validation-selected checkpoint evaluation complete",
        extra={
            "context": {
                "step": checkpoint["step"],
                "test_evaluated": test_data is not None,
                "test_loss": test_loss,
                "test_bits_per_character": test_bpc,
            }
        },
    )

    sample_generator = torch.Generator().manual_seed(int(config["seed"]) + 4002)
    samples = []
    for prompt in _read_prompts(args.prompts):
        continuation = generate_sample(
            model,
            tokenizer,
            prompt,
            settings,
            generation,
            sample_generator,
            device,
        )
        samples.append({"prompt": prompt, "continuation": continuation})

    matching_history = [
        entry
        for entry in checkpoint.get("history", [])
        if int(entry.get("step", -1)) == int(checkpoint["step"])
    ]
    checkpoint_validation = matching_history[-1] if matching_history else {}
    report = {
        "schema_version": "selected-pretrain-evaluation-v4/v1",
        "selection_rule": args.selection_rule,
        "run_id": run_id,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_validation_loss": checkpoint_validation.get("val_loss"),
        "checkpoint_validation_bits_per_character": checkpoint_validation.get(
            "val_bits_per_character"
        ),
        "test_loss": test_loss,
        "test_bits_per_character": test_bpc,
        "test_evaluated": test_data is not None,
        "test_batches": int(settings["eval_batches"]) if test_data is not None else 0,
        "test_tokens_per_batch": (
            int(settings["micro_batch_size"]) * model_config.block_size
            if test_data is not None
            else 0
        ),
        "parameter_count": model.parameter_count(),
        "tokenizer_sha256": file_sha256(data_dir / "tokenizer.json"),
        "generation": generation,
        "sample_max_characters": int(settings["sample_max_characters"]),
        "samples": samples,
    }
    atomic_write_json(output_path, report)
    loggers["orchestrator"].info(
        "selected-model evaluation artifact written",
        extra={"context": {"output": str(output_path), "test_loss": test_loss}},
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
