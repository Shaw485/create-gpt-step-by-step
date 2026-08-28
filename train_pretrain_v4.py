"""Run resumable local v4 pretraining on the 8.1M-parameter GPT."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch

from bpe_tokenizer import BPETokenizer
from gpt_model_v4 import GPTConfig, GPTLanguageModelV4
from training_runtime import (
    EarlyStopping,
    EmergencyCheckpointHook,
    RunStateWriter,
    assert_finite_gradients,
    assert_finite_tensor,
    atomic_save_checkpoint,
    atomic_write_json,
    build_checkpoint_payload,
    canonical_json_sha256,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    load_checkpoint,
    restore_checkpoint,
)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "local-pretrain-v4/v1":
        raise ValueError("unsupported local pretraining config schema")
    return config


def select_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError(
                "config requests MPS but it is unavailable; run outside a restricted "
                "sandbox or explicitly choose CPU for a much slower diagnostic run"
            )
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported local device: {requested}")


def load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 1:
        raise ValueError(f"expected one-dimensional int64 tensor: {path}")
    return value


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size:
        raise ValueError("token split is shorter than block_size")
    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,),
        generator=generator,
    )
    inputs = torch.stack([data[index : index + block_size] for index in starts])
    targets = torch.stack(
        [data[index + 1 : index + block_size + 1] for index in starts]
    )
    return inputs.to(device), targets.to(device)


def learning_rate(step: int, settings: dict[str, Any]) -> float:
    maximum = float(settings["learning_rate"])
    minimum = float(settings["minimum_learning_rate"])
    warmup = int(settings["warmup_steps"])
    maximum_steps = int(settings["max_steps"])
    schedule_start = int(settings.get("schedule_start_step", 0))
    schedule_step = max(0, step - schedule_start)
    schedule_length = max(1, maximum_steps - schedule_start)
    if schedule_step < warmup:
        return maximum * (schedule_step + 1) / max(1, warmup)
    progress = min(1.0, (schedule_step - warmup) / max(1, schedule_length - warmup))
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: GPTLanguageModelV4,
    data: torch.Tensor,
    settings: dict[str, Any],
    generator: torch.Generator,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(int(settings["eval_batches"])):
        inputs, targets = get_batch(
            data,
            int(settings["micro_batch_size"]),
            model.config.block_size,
            generator,
            device,
        )
        _, loss = model(inputs, targets)
        assert loss is not None
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_sample(
    model: GPTLanguageModelV4,
    tokenizer: BPETokenizer,
    prompt: str,
    settings: dict[str, Any],
    generation: dict[str, Any],
    generator: torch.Generator,
    device: torch.device,
) -> str:
    ids = tokenizer.encode(prompt)
    generated = []
    eos_id = tokenizer.special_to_id["<EOS>"]
    model.eval()
    for _ in range(int(generation["max_new_tokens"])):
        context = torch.tensor(
            [ids[-model.config.block_size :]], dtype=torch.long, device=device
        )
        logits, _ = model(context)
        scores = logits[0, -1].float().cpu() / max(
            float(generation["temperature"]), 1e-6
        )
        top_k = min(int(generation["top_k"]), scores.numel())
        values, indices = torch.topk(scores, top_k)
        selected = torch.multinomial(
            torch.softmax(values, dim=-1), 1, generator=generator
        )
        next_id = int(indices[selected].item())
        if next_id == eos_id:
            break
        ids.append(next_id)
        generated.append(next_id)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if len(text) >= int(settings["sample_max_characters"]):
            break
    model.train()
    return tokenizer.decode(generated, skip_special_tokens=True)[
        : int(settings["sample_max_characters"])
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local_m4_8m.json"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint cannot be used together")

    config = load_config(args.config)
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.run_dir is not None:
        config["run_dir"] = str(args.run_dir)
    if args.smoke:
        config["training"].update(
            {
                "max_steps": 3,
                "eval_interval": 1,
                "eval_batches": 2,
                "checkpoint_interval": 1,
                "sample_interval": 3,
            }
        )
    config_hash = canonical_json_sha256(config)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = generate_run_id("local-v4-smoke" if args.smoke else "local-v4")
    loggers = configure_module_loggers(
        run_dir / "logs",
        run_id,
        config["logging"]["module_levels"],
        max_bytes=int(config["logging"]["max_bytes"]),
        backup_count=int(config["logging"]["backup_count"]),
        console=bool(config["logging"]["console"]),
    )
    state_writer = RunStateWriter(run_dir, run_id, config_hash)
    atomic_write_json(run_dir / "effective_config.json", config)
    device = select_device(config["device"])
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    batch_generator = torch.Generator().manual_seed(seed + 1)
    eval_generator = torch.Generator().manual_seed(seed + 2)
    sample_generator = torch.Generator().manual_seed(seed + 3)
    data_dir = Path(config["data_dir"])
    settings = config["training"]
    started = time.monotonic()

    train_data = load_tensor(data_dir / "train_tokens.pt")
    val_data = load_tensor(data_dir / "val_tokens.pt")
    test_data = load_tensor(data_dir / "test_tokens.pt")
    tokenizer = BPETokenizer.load(data_dir / "tokenizer.json")
    manifest = json.loads((data_dir / "token_manifest.json").read_text(encoding="utf-8"))
    if manifest["tokenizer_sha256"] != file_sha256(data_dir / "tokenizer.json"):
        raise ValueError("tokenizer checksum does not match token manifest")

    model_config = GPTConfig(vocab_size=tokenizer.vocab_size, **config["model"])
    model = GPTLanguageModelV4(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        betas=tuple(settings["betas"]),
        weight_decay=float(settings["weight_decay"]),
    )
    early_stopping = EarlyStopping(
        int(settings["early_stopping_patience"]),
        float(settings["early_stopping_min_delta"]),
    )
    history: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    start_step = 1
    best_loss = float("inf")
    best_step = -1
    elapsed_offset = 0.0
    initial_checkpoint: dict[str, Any] | None = None
    if args.resume:
        resumed = restore_checkpoint(
            args.resume,
            model,
            optimizer,
            batch_generator,
            expected_config_sha256=config_hash,
            map_location=device,
            restore_cuda_rng=False,
        )
        start_step = resumed.step + 1
        best_loss = resumed.best_metric
        best_step = int(resumed.early_stopping_state.get("best_step", resumed.step))
        history = resumed.history
        samples = list(resumed.extra.get("samples", []))
        if "eval_generator_state" in resumed.extra:
            eval_generator.set_state(resumed.extra["eval_generator_state"].cpu())
        if "sample_generator_state" in resumed.extra:
            sample_generator.set_state(resumed.extra["sample_generator_state"].cpu())
        if resumed.early_stopping_state:
            early_stopping.load_state_dict(resumed.early_stopping_state)
        elapsed_offset = max(
            (float(entry.get("elapsed_seconds", 0.0)) for entry in history),
            default=0.0,
        )
    elif args.init_checkpoint:
        checkpoint = load_checkpoint(args.init_checkpoint, map_location="cpu")
        checkpoint_config = checkpoint.get("extra", {}).get("model_config")
        if checkpoint_config != model_config.to_dict():
            raise ValueError(
                "initial checkpoint model configuration does not match the current model"
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        start_step = int(checkpoint["step"]) + 1
        best_loss = float(checkpoint["best_metric"])
        best_step = int(checkpoint["step"])
        history = [dict(entry) for entry in checkpoint["history"]]
        samples = list(checkpoint.get("extra", {}).get("samples", []))
        elapsed_offset = max(
            (float(entry.get("elapsed_seconds", 0.0)) for entry in history),
            default=0.0,
        )
        early_stopping.best_metric = best_loss
        early_stopping.best_step = best_step
        initial_checkpoint = {
            "path": str(args.init_checkpoint),
            "step": int(checkpoint["step"]),
            "sha256": file_sha256(args.init_checkpoint),
            "optimizer_policy": "fresh AdamW state for the lower-learning-rate phase",
        }
        loggers["checkpoint"].info(
            "model initialized from validation-selected checkpoint",
            extra={"context": initial_checkpoint},
        )

    current_step = max(0, start_step - 1)

    def elapsed_seconds() -> float:
        return elapsed_offset + time.monotonic() - started

    def payload() -> dict[str, Any]:
        return build_checkpoint_payload(
            model,
            optimizer,
            step=current_step,
            best_metric=best_loss,
            history=history,
            sampling_generator=batch_generator,
            config_sha256=config_hash,
            early_stopping_state=early_stopping.state_dict(),
            extra={
                "samples": samples,
                "model_config": model_config.to_dict(),
                "parameter_count": model.parameter_count(),
                "token_manifest_sha256": file_sha256(data_dir / "token_manifest.json"),
                "eval_generator_state": eval_generator.get_state(),
                "sample_generator_state": sample_generator.get_state(),
                "initial_checkpoint": initial_checkpoint,
            },
        )

    emergency = EmergencyCheckpointHook(
        run_dir / "emergency.pt",
        payload,
        logger=loggers["checkpoint"],
    )
    prompts_path = Path(config.get("sample_prompts_path", "data/prompt10_eval.txt"))
    prompts = [
        line.strip()
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: int(config.get("sample_prompt_limit", 10))]
    loggers["data"].info(
        "formal tensors loaded",
        extra={
            "context": {
                "device": str(device),
                "parameters": model.parameter_count(),
                "vocab_size": tokenizer.vocab_size,
                "train_tokens": len(train_data),
                "val_tokens": len(val_data),
                "test_tokens": len(test_data),
                "block_size": model_config.block_size,
                "micro_batch": settings["micro_batch_size"],
                "gradient_accumulation": settings["gradient_accumulation_steps"],
            }
        },
    )

    try:
        with emergency:
            for step in range(start_step, int(settings["max_steps"]) + 1):
                current_step = step
                lr = learning_rate(step, settings)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                for _ in range(int(settings["gradient_accumulation_steps"])):
                    inputs, targets = get_batch(
                        train_data,
                        int(settings["micro_batch_size"]),
                        model_config.block_size,
                        batch_generator,
                        device,
                    )
                    _, loss = model(inputs, targets)
                    assert loss is not None
                    assert_finite_tensor(loss, "pretraining loss")
                    (loss / int(settings["gradient_accumulation_steps"])).backward()
                    accumulated_loss += float(loss.detach().cpu())
                assert_finite_gradients(model.named_parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(settings["gradient_clip_norm"])
                )
                optimizer.step()
                batch_loss = accumulated_loss / int(settings["gradient_accumulation_steps"])

                if step % int(settings["log_interval"]) == 0 or step <= 3:
                    loggers["pretrain"].info(
                        "optimizer step complete",
                        extra={
                            "context": {
                                "step": step,
                                "max_steps": settings["max_steps"],
                                "loss": batch_loss,
                                "learning_rate": lr,
                                "gradient_norm": float(grad_norm.detach().cpu()),
                                "elapsed_seconds": elapsed_seconds(),
                            }
                        },
                    )

                should_stop = False
                if step % int(settings["eval_interval"]) == 0 or step == int(settings["max_steps"]):
                    train_loss = evaluate(model, train_data, settings, eval_generator, device)
                    val_loss = evaluate(model, val_data, settings, eval_generator, device)
                    decision = early_stopping.update(val_loss, step)
                    best_loss = decision.best_metric
                    best_step = decision.best_step
                    entry = {
                        "step": step,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": lr,
                        "elapsed_seconds": elapsed_seconds(),
                    }
                    history.append(entry)
                    loggers["validation"].info(
                        "validation complete",
                        extra={"context": {**entry, "best_val_loss": best_loss}},
                    )
                    if decision.improved:
                        atomic_save_checkpoint(run_dir / "best.pt", payload())
                    should_stop = decision.should_stop

                if step % int(settings["sample_interval"]) == 0 or step == int(settings["max_steps"]):
                    step_samples = []
                    for prompt in prompts:
                        continuation = generate_sample(
                            model,
                            tokenizer,
                            prompt,
                            settings,
                            config["generation"],
                            sample_generator,
                            device,
                        )
                        step_samples.append({"prompt": prompt, "continuation": continuation})
                    samples.append({"step": step, "samples": step_samples})
                    atomic_write_json(run_dir / "samples.json", {"history": samples})

                if step % int(settings["checkpoint_interval"]) == 0 or step == int(settings["max_steps"]):
                    result = atomic_save_checkpoint(run_dir / "latest.pt", payload())
                    loggers["checkpoint"].info(
                        "latest checkpoint saved",
                        extra={"context": {"step": step, "sha256": result.sha256}},
                    )
                if should_stop:
                    loggers["orchestrator"].info(
                        "early stopping requested",
                        extra={"context": {"step": step, "best_val_loss": best_loss}},
                    )
                    break

        test_loss = evaluate(model, test_data, settings, eval_generator, device)
        report = {
            "status": "complete",
            "run_id": run_id,
            "config_sha256": config_hash,
            "final_step": current_step,
            "parameter_count": model.parameter_count(),
            "device": str(device),
            "best_validation_loss": best_loss,
            "best_step": best_step,
            "test_loss": test_loss,
            "stage_elapsed_seconds": time.monotonic() - started,
            "elapsed_seconds": elapsed_seconds(),
            "history": history,
            "samples_path": str(run_dir / "samples.json"),
            "latest_checkpoint": str(run_dir / "latest.pt"),
            "best_checkpoint": str(
                run_dir / "best.pt" if (run_dir / "best.pt").is_file() else args.init_checkpoint
            ),
            "initial_checkpoint": initial_checkpoint,
        }
        atomic_write_json(run_dir / "report.json", report)
        state_writer.mark_done(report)
        loggers["orchestrator"].info(
            "training complete", extra={"context": report}
        )
    except BaseException as error:
        state_writer.mark_failed(error, {"step": current_step})
        loggers["orchestrator"].exception(
            "training failed", extra={"context": {"step": current_step}}
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
