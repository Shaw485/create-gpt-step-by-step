"""Sample an SFT v7 checkpoint through the isolated public-token contract.

This entry point never accepts a training corpus or a blind-evaluation split.
It uses the public tensor artifact only for tokenizer/model provenance and
supports either one or more single-turn prompts or one multi-turn conversation.
Prompt and generated text are written only to the requested sample artifacts;
structured runtime logs contain counts, hashes, and lengths but no text bodies.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch

from bpe_tokenizer import BPETokenizer
from prepare_bpe_sft_v7 import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SPECIAL_TOKEN_IDS,
    EXPECTED_TOKENIZER_SHA256,
    EXPECTED_VOCAB_SIZE,
    REQUIRED_BASE_CHECKPOINT,
)
from train_pretrain_v4 import load_config
from train_sft_v4 import build_model, select_device
from training_runtime import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_sha256,
    close_module_loggers,
    configure_module_loggers,
    file_sha256,
    generate_run_id,
    load_checkpoint,
    resolve_module_log_levels,
)


DEFAULT_CONFIG = Path("configs/formal_pretrain_14m_bpe3000.json")
DEFAULT_PUBLIC_TENSORS = Path("data/sft/v7/public_diagnostic_tensors.pt")
DEFAULT_CHECKPOINT = Path("runs/sft_v7_vertical/latest.pt")
DEFAULT_BASELINE_CHECKPOINT = Path(str(REQUIRED_BASE_CHECKPOINT["path"]))
DEFAULT_OUTPUT_JSON = Path(
    "reports/milestones/020_sft_v7_vertical/custom_samples.json"
)
DEFAULT_OUTPUT_MARKDOWN = Path(
    "reports/milestones/020_sft_v7_vertical/custom_samples.md"
)
DEFAULT_LOG_DIR = Path("logs/sft_v7_sample")

PUBLIC_SCHEMA = "sft-v7-public-tensors/v1"
EXPECTED_STAGE = "sft_v7_vertical"
MAX_SEQUENCE_LENGTH = 512
PUBLIC_RECORD_COUNT = 600
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_FIELD = re.compile(r"(?:^|[^a-z])(sealed|test)(?:$|[^a-z])")
_PLAINTEXT_TENSOR_FIELDS = {
    "answer",
    "content",
    "evidence",
    "messages",
    "prompt",
    "question",
    "reference_answer",
    "source_path",
    "text",
}


class SFTV7SamplingError(ValueError):
    """A public-safe sampling failure with an actionable, log-safe code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> None:
    raise SFTV7SamplingError(code, message)


def _contains_forbidden_field(value: str) -> bool:
    return bool(_FORBIDDEN_FIELD.search(value.lower()))


def reject_forbidden_public_fields(value: Any, *, location: str = "payload") -> None:
    """Reject fields or path components that reconnect a public run to blind data."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if _contains_forbidden_field(key_text):
                _fail("forbidden_public_field", f"{location} has a forbidden field")
            if "path" in key_text.lower() and isinstance(nested, (str, Path)):
                components = [
                    part
                    for part in re.split(r"[/\\]+", str(nested).lower())
                    if part
                ]
                if any(_contains_forbidden_field(part) for part in components):
                    _fail("forbidden_public_path", f"{location} has a forbidden path")
            reject_forbidden_public_fields(nested, location=f"{location}.{key_text}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            reject_forbidden_public_fields(nested, location=f"{location}[{index}]")
        return
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and "path" in location.lower():
        components = [part for part in re.split(r"[/\\]+", value.lower()) if part]
        if any(_contains_forbidden_field(part) for part in components):
            _fail("forbidden_public_path", f"{location} has a forbidden path")


def _torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        value = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch compatibility
        value = torch.load(path, map_location=map_location)
    if not isinstance(value, dict):
        _fail("invalid_artifact", "artifact root must be a dictionary")
    return value


def _reject_plaintext_metadata(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).strip().lower() in _PLAINTEXT_TENSOR_FIELDS:
                _fail(
                    "public_tensor_plaintext_field",
                    "public tensor metadata contains a plaintext field",
                )
            _reject_plaintext_metadata(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_plaintext_metadata(nested)


def validate_public_payload(
    payload: Mapping[str, Any],
    *,
    expected_count: int | None = PUBLIC_RECORD_COUNT,
) -> None:
    """Validate the public-only tensor schema without opening any source path."""

    reject_forbidden_public_fields(payload)
    required = {
        "schema_version",
        "public_records",
        "vocab_size",
        "stoi",
        "itos",
        "special_token_ids",
        "ignore_index",
        "tokenizer_path",
        "tokenizer_sha256",
        "bpe_token_manifest_path",
        "bpe_token_manifest_sha256",
        "sft_dataset_manifest_sha256",
        "source_jsonl_paths",
        "source_jsonl_sha256",
        "required_base_checkpoint",
        "artifact_binding_sha256",
    }
    missing = sorted(required.difference(payload))
    if missing:
        _fail("public_payload_missing_fields", f"public payload misses {len(missing)} fields")
    if payload["schema_version"] != PUBLIC_SCHEMA:
        _fail("public_schema_mismatch", "public tensor schema is not SFT v7")
    record_keys = {str(key) for key in payload if str(key).endswith("_records")}
    if record_keys != {"public_records"}:
        _fail("public_payload_not_isolated", "public payload contains a non-public record set")
    if int(payload["vocab_size"]) != EXPECTED_VOCAB_SIZE:
        _fail("public_vocab_mismatch", "public payload vocabulary is not frozen")
    actual_special = {
        str(token): int(token_id)
        for token, token_id in dict(payload["special_token_ids"]).items()
    }
    if actual_special != EXPECTED_SPECIAL_TOKEN_IDS:
        _fail("public_special_ids_mismatch", "public payload special-token IDs changed")
    if int(payload["ignore_index"]) != -100:
        _fail("public_ignore_index_mismatch", "public payload ignore index must be -100")
    if str(payload["tokenizer_sha256"]) != EXPECTED_TOKENIZER_SHA256:
        _fail("public_tokenizer_sha_mismatch", "public payload tokenizer SHA changed")
    if str(payload["bpe_token_manifest_sha256"]) != EXPECTED_MANIFEST_SHA256:
        _fail("public_manifest_sha_mismatch", "public payload manifest SHA changed")
    if not _HEX_SHA256.fullmatch(str(payload["sft_dataset_manifest_sha256"])):
        _fail("dataset_manifest_sha_invalid", "SFT dataset manifest SHA is invalid")
    source_paths = payload["source_jsonl_paths"]
    source_hashes = payload["source_jsonl_sha256"]
    if not isinstance(source_paths, Mapping) or set(source_paths) != {"public_diagnostic"}:
        _fail("public_source_scope_mismatch", "public payload source scope is not isolated")
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != {"public_diagnostic"}:
        _fail("public_source_hash_scope_mismatch", "public source hashes are not isolated")
    if Path(str(source_paths["public_diagnostic"])).name != "public_diagnostic.jsonl":
        _fail("public_source_name_mismatch", "public source has an unexpected filename")
    if not _HEX_SHA256.fullmatch(str(source_hashes["public_diagnostic"])):
        _fail("public_source_sha_invalid", "public source SHA is invalid")

    required_base = payload["required_base_checkpoint"]
    if not isinstance(required_base, Mapping):
        _fail("base_provenance_missing", "public payload lacks base provenance")
    expected_base = dict(REQUIRED_BASE_CHECKPOINT)
    expected_base["binding_sha256"] = canonical_json_sha256(REQUIRED_BASE_CHECKPOINT)
    if dict(required_base) != expected_base:
        _fail("base_provenance_mismatch", "public payload base provenance changed")
    binding = {
        "schema_version": payload["schema_version"],
        "source_jsonl_sha256": dict(source_hashes),
        "tokenizer_sha256": payload["tokenizer_sha256"],
        "bpe_token_manifest_sha256": payload["bpe_token_manifest_sha256"],
        "sft_dataset_manifest_sha256": payload["sft_dataset_manifest_sha256"],
        "required_base_checkpoint": dict(required_base),
    }
    if payload["artifact_binding_sha256"] != canonical_json_sha256(binding):
        _fail("public_artifact_binding_mismatch", "public artifact binding is invalid")

    records = payload["public_records"]
    if not isinstance(records, list) or not records:
        _fail("public_records_empty", "public payload contains no records")
    if expected_count is not None and len(records) != expected_count:
        _fail("public_record_count_mismatch", "public payload is not the frozen 600 records")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            _fail("invalid_public_tensor_record", "public tensor record is not an object")
        if record.get("split") != "public_diagnostic":
            _fail("public_tensor_split_mismatch", "public tensor record has a wrong split")
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            _fail("public_tensor_id_invalid", "public tensor record ID is empty or duplicated")
        seen.add(identifier)
        allowed_record_fields = {
            "id",
            "primary_dimension",
            "task_family",
            "split",
            "input_ids",
            "labels",
            "assistant_spans",
            "assistant_turns",
            "sequence_length",
            "evaluation",
        }
        if set(record).difference(allowed_record_fields):
            _fail("public_tensor_unexpected_field", "public tensor record has an unexpected field")
        input_ids = record.get("input_ids")
        labels = record.get("labels")
        if (
            not isinstance(input_ids, torch.Tensor)
            or not isinstance(labels, torch.Tensor)
            or input_ids.dtype != torch.long
            or labels.dtype != torch.long
            or input_ids.ndim != 1
            or labels.ndim != 1
            or len(input_ids) != len(labels)
            or not 1 <= len(input_ids) < MAX_SEQUENCE_LENGTH
        ):
            _fail("public_tensor_shape_invalid", "public tensor record shape is invalid")
        if int((labels != -100).sum()) <= 0:
            _fail("public_tensor_supervision_missing", "public tensor record has no supervision")
        if bool((input_ids < 0).any()) or bool((input_ids >= EXPECTED_VOCAB_SIZE).any()):
            _fail("public_tensor_token_id_invalid", "public tensor input ID is invalid")
        supervised_labels = labels[labels != -100]
        if bool((supervised_labels < 0).any()) or bool(
            (supervised_labels >= EXPECTED_VOCAB_SIZE).any()
        ):
            _fail("public_tensor_label_invalid", "public tensor target ID is invalid")
        if int(record.get("sequence_length", -1)) != len(input_ids) + 1:
            _fail("public_tensor_length_mismatch", "public tensor sequence length is invalid")
        if int(record.get("assistant_turns", 0)) <= 0:
            _fail("public_tensor_turns_invalid", "public tensor assistant turns are invalid")
        if "evaluation" not in record:
            _fail("public_evaluation_missing", "public tensor record lacks scoring metadata")
        _reject_plaintext_metadata(record["evaluation"])


def load_public_payload(path: Path) -> dict[str, Any]:
    if path.name != "public_diagnostic_tensors.pt":
        _fail("public_tensor_name_mismatch", "public tensor artifact has an unexpected name")
    reject_forbidden_public_fields({"public_tensor_path": path})
    payload = _torch_load(path)
    validate_public_payload(payload, expected_count=PUBLIC_RECORD_COUNT)
    return payload


def load_bound_tokenizer(payload: Mapping[str, Any]) -> BPETokenizer:
    """Load and recompute both tokenizer and manifest identities."""

    tokenizer_path = Path(str(payload["tokenizer_path"]))
    manifest_path = Path(str(payload["bpe_token_manifest_path"]))
    reject_forbidden_public_fields(
        {"tokenizer_path": tokenizer_path, "manifest_path": manifest_path}
    )
    if file_sha256(tokenizer_path) != str(payload["tokenizer_sha256"]):
        _fail("tokenizer_file_sha_mismatch", "tokenizer file does not match public payload")
    if file_sha256(manifest_path) != str(payload["bpe_token_manifest_sha256"]):
        _fail("manifest_file_sha_mismatch", "manifest file does not match public payload")
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != int(payload["vocab_size"]):
        _fail("tokenizer_vocab_mismatch", "loaded tokenizer vocabulary changed")
    if tokenizer.special_to_id != EXPECTED_SPECIAL_TOKEN_IDS:
        _fail("tokenizer_special_ids_mismatch", "loaded tokenizer special IDs changed")
    expected_stoi = {token: index for index, token in enumerate(tokenizer.tokens)}
    actual_stoi = {str(token): int(index) for token, index in dict(payload["stoi"]).items()}
    actual_itos = {int(index): str(token) for index, token in dict(payload["itos"]).items()}
    if actual_stoi != expected_stoi or actual_itos != dict(enumerate(tokenizer.tokens)):
        _fail("tokenizer_tables_mismatch", "public token lookup tables changed")
    return tokenizer


def validate_checkpoint_provenance(
    checkpoint: Mapping[str, Any],
    public_payload: Mapping[str, Any],
    *,
    require_formal_counts: bool = True,
) -> dict[str, Any]:
    """Bind an SFT checkpoint to the frozen Step5750/BPE3000 training lineage."""

    if checkpoint.get("schema_version") != "training-checkpoint/v1":
        _fail("checkpoint_schema_mismatch", "checkpoint is not a training checkpoint")
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        _fail("checkpoint_model_missing", "checkpoint has no model state")
    if int(checkpoint.get("step", -1)) < 0:
        _fail("checkpoint_step_invalid", "checkpoint step is invalid")
    if not _HEX_SHA256.fullmatch(str(checkpoint.get("config_sha256", ""))):
        _fail("checkpoint_signature_invalid", "checkpoint training signature is invalid")
    extra = checkpoint.get("extra")
    if not isinstance(extra, Mapping):
        _fail("checkpoint_provenance_missing", "checkpoint lacks provenance")
    expected = {
        "stage": EXPECTED_STAGE,
        "base_checkpoint_path": REQUIRED_BASE_CHECKPOINT["path"],
        "base_checkpoint_sha256": REQUIRED_BASE_CHECKPOINT["sha256"],
        "base_checkpoint_step": REQUIRED_BASE_CHECKPOINT["step"],
        "base_config_canonical_sha256": REQUIRED_BASE_CHECKPOINT[
            "config_canonical_sha256"
        ],
        "base_token_manifest_sha256": REQUIRED_BASE_CHECKPOINT[
            "token_manifest_sha256"
        ],
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "sft_dataset_manifest_sha256": str(
            public_payload["sft_dataset_manifest_sha256"]
        ),
    }
    for key, value in expected.items():
        if extra.get(key) != value:
            _fail("checkpoint_provenance_mismatch", f"checkpoint provenance mismatch: {key}")
    if Path(str(extra.get("sft_tensor_path", ""))).name != "train_val_tensors.pt":
        _fail("checkpoint_training_artifact_invalid", "checkpoint training artifact name changed")
    if not _HEX_SHA256.fullmatch(str(extra.get("sft_tensor_sha256", ""))):
        _fail("checkpoint_training_sha_invalid", "checkpoint training artifact SHA is invalid")
    if int(extra.get("public_records_consumed", -1)) != 0:
        _fail("checkpoint_public_leakage", "checkpoint reports public records consumed")
    if int(extra.get("sealed_records_consumed", -1)) != 0:
        _fail("checkpoint_blind_leakage", "checkpoint reports blind records consumed")
    summary = extra.get("payload_summary")
    if not isinstance(summary, Mapping):
        _fail("checkpoint_payload_summary_missing", "checkpoint lacks training summary")
    counts = summary.get("split_counts")
    if not isinstance(counts, Mapping):
        _fail("checkpoint_split_counts_missing", "checkpoint lacks training split counts")
    normalized_counts = {str(key): int(value) for key, value in counts.items()}
    if require_formal_counts and normalized_counts != {"train": 8000, "val": 800}:
        _fail("checkpoint_split_counts_mismatch", "checkpoint is not formal 8000/800 SFT v7")
    return {
        "step": int(checkpoint["step"]),
        "training_tensor_sha256": str(extra["sft_tensor_sha256"]),
        "training_split_counts": normalized_counts,
        "base_checkpoint_sha256": str(extra["base_checkpoint_sha256"]),
    }


def validate_model_config(config: Mapping[str, Any]) -> None:
    expected_config_sha = str(REQUIRED_BASE_CHECKPOINT["config_canonical_sha256"])
    if canonical_json_sha256(config) != expected_config_sha:
        _fail("config_sha_mismatch", "configuration is not frozen Step 5750")
    model = config.get("model")
    if not isinstance(model, Mapping):
        _fail("model_config_missing", "configuration lacks a model section")
    expected = {
        "block_size": 512,
        "embedding_size": 320,
        "num_layers": 10,
        "num_heads": 8,
        "ffn_multiplier": 4,
        "dropout": 0.1,
        "tie_embeddings": True,
    }
    if dict(model) != expected:
        _fail("model_config_mismatch", "configuration is not the frozen 14.9M model")


def validate_pretrain_baseline_checkpoint(
    checkpoint: Mapping[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    """Require the exact pure-pretraining Step5750 artifact."""

    if file_sha256(checkpoint_path) != REQUIRED_BASE_CHECKPOINT["sha256"]:
        _fail("baseline_checkpoint_sha_mismatch", "baseline is not the frozen file")
    if int(checkpoint.get("step", -1)) != REQUIRED_BASE_CHECKPOINT["step"]:
        _fail("baseline_checkpoint_step_mismatch", "baseline is not Step 5750")
    if checkpoint.get("config_sha256") != REQUIRED_BASE_CHECKPOINT[
        "config_canonical_sha256"
    ]:
        _fail("baseline_checkpoint_config_mismatch", "baseline config hash changed")
    extra = checkpoint.get("extra")
    if not isinstance(extra, Mapping):
        _fail("baseline_checkpoint_provenance_missing", "baseline provenance is missing")
    if extra.get("initial_checkpoint") is not None:
        _fail("baseline_checkpoint_not_pure", "baseline is not pure pretraining")
    if int(extra.get("parameter_count", -1)) != REQUIRED_BASE_CHECKPOINT[
        "parameter_count"
    ]:
        _fail("baseline_parameter_count_mismatch", "baseline parameter count changed")
    if extra.get("token_manifest_sha256") != REQUIRED_BASE_CHECKPOINT[
        "token_manifest_sha256"
    ]:
        _fail("baseline_token_manifest_mismatch", "baseline token manifest changed")
    expected_model = {
        "vocab_size": 7465,
        "block_size": 512,
        "embedding_size": 320,
        "num_layers": 10,
        "num_heads": 8,
        "ffn_multiplier": 4,
        "dropout": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initialization_std": 0.02,
        "tie_embeddings": True,
    }
    if extra.get("model_config") != expected_model:
        _fail("baseline_model_config_mismatch", "baseline model architecture changed")
    sft_markers = {
        "payload_summary",
        "sampler_state",
        "sft_dataset_manifest_sha256",
        "sft_tensor_path",
        "sft_tensor_sha256",
        "base_checkpoint_sha256",
    }
    if extra.get("stage") == EXPECTED_STAGE or sft_markers & set(extra):
        _fail("baseline_contains_sft_marker", "baseline contains SFT provenance")
    return {
        "step": int(checkpoint["step"]),
        "base_checkpoint_sha256": REQUIRED_BASE_CHECKPOINT["sha256"],
        "checkpoint_mode": "pretrain-baseline",
    }


def load_model_bundle(
    config_path: Path,
    checkpoint_path: Path,
    public_payload: Mapping[str, Any],
    device: torch.device,
    checkpoint_mode: str = "sft-v7",
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    validate_model_config(config)
    try:
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    except (OSError, ValueError) as error:
        raise SFTV7SamplingError(
            "checkpoint_integrity_failure", "checkpoint or checksum sidecar is invalid"
        ) from error
    if checkpoint_mode == "pretrain-baseline":
        provenance = validate_pretrain_baseline_checkpoint(checkpoint, checkpoint_path)
    elif checkpoint_mode == "sft-v7":
        provenance = validate_checkpoint_provenance(checkpoint, public_payload)
        provenance["checkpoint_mode"] = "sft-v7"
    else:  # defensive for programmatic callers
        _fail("checkpoint_mode_invalid", "unknown checkpoint mode")
    model = build_model(config, int(public_payload["vocab_size"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint, provenance


def build_conversation_prompt_ids(
    tokenizer: BPETokenizer,
    messages: Sequence[Mapping[str, Any]],
    special_token_ids: Mapping[str, int],
    *,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> list[int]:
    """Serialize ``user, assistant, ..., user`` and open the next assistant turn."""

    if not messages or len(messages) % 2 == 0:
        _fail("invalid_prompt_turn_count", "conversation must end with a user turn")
    sequence = [int(special_token_ids["<BOS>"])]
    for index, message in enumerate(messages):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if not isinstance(message, Mapping) or message.get("role") != expected_role:
            _fail("invalid_prompt_role_order", "conversation roles do not alternate")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            _fail("empty_prompt_content", "conversation contains an empty message")
        try:
            content_ids = tokenizer.encode(content)
        except ValueError as error:
            raise SFTV7SamplingError(
                "unencodable_prompt",
                "conversation contains characters outside the frozen vocabulary",
            ) from error
        if expected_role == "user":
            sequence.append(int(special_token_ids["<USER>"]))
            sequence.extend(content_ids)
        else:
            sequence.append(int(special_token_ids["<ASSISTANT>"]))
            sequence.extend(content_ids)
            sequence.append(int(special_token_ids["<EOS>"]))
    sequence.append(int(special_token_ids["<ASSISTANT>"]))
    if len(sequence) > max_sequence_length:
        _fail("prompt_too_long", "conversation exceeds the 512-token context")
    return sequence


def forbidden_generation_token_ids(special_token_ids: Mapping[str, int]) -> tuple[int, ...]:
    """Mask every control token except EOS; UNK is a control token here."""

    return tuple(
        int(special_token_ids[token])
        for token in ("<UNK>", "<BOS>", "<USER>", "<ASSISTANT>", "<PAD>")
    )


def mask_generation_scores(
    scores: torch.Tensor,
    special_token_ids: Mapping[str, int],
) -> torch.Tensor:
    masked = scores.clone()
    masked[list(forbidden_generation_token_ids(special_token_ids))] = float("-inf")
    return masked


def _sample_next_token_id(
    scores: torch.Tensor,
    special_token_ids: Mapping[str, int],
    *,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
) -> int:
    """Apply the frozen CPU sampling path used by single and batched generation."""

    cpu_scores = scores.detach().float().cpu() / temperature
    cpu_scores = mask_generation_scores(cpu_scores, special_token_ids)
    if top_k > 0:
        values, indices = torch.topk(cpu_scores, min(top_k, cpu_scores.numel()))
        probabilities = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probabilities, 1, generator=generator)
        return int(indices[choice].item())
    probabilities = torch.softmax(cpu_scores, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def _generation_result(
    generated_ids: Sequence[int],
    tokenizer: BPETokenizer,
    *,
    stopped_on_eos: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    ids = [int(token_id) for token_id in generated_ids]
    return {
        "generated_text": tokenizer.decode(ids, skip_special_tokens=True),
        "generated_token_ids": ids,
        "generated_tokens": len(ids),
        "stopped_on_eos": bool(stopped_on_eos),
        "truncated": not stopped_on_eos and len(ids) == max_new_tokens,
    }


class _InferenceDecodeSession:
    """Share the safe KV-cache policy across single and batched decoding."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        use_kv_cache: bool,
        padding_token_id: int,
    ):
        self.model = model
        self.device = device
        self.padding_token_id = int(padding_token_id)
        inference_forward = getattr(model, "forward_inference", None)
        self.inference_forward = (
            inference_forward
            if use_kv_cache and callable(inference_forward)
            else None
        )
        self.supports_left_padding = bool(
            getattr(model, "supports_left_padded_inference", False)
        )
        self.cache: Any | None = None
        self.cached_context_lengths: tuple[int, ...] | None = None

    def _legacy_scores(self, contexts: Sequence[Sequence[int]]) -> torch.Tensor:
        """Preserve the original unpadded path for unsupported model classes."""

        lengths = {len(context) for context in contexts}
        if len(lengths) == 1:
            inputs = torch.tensor(contexts, dtype=torch.long, device=self.device)
            logits, _ = self.model(inputs)
            return logits[:, -1]
        per_row_scores: list[torch.Tensor] = []
        for context in contexts:
            inputs = torch.tensor([context], dtype=torch.long, device=self.device)
            logits, _ = self.model(inputs)
            per_row_scores.append(logits[0, -1])
        return torch.stack(per_row_scores, dim=0)

    def next_token_scores(self, contexts: Sequence[Sequence[int]]) -> torch.Tensor:
        if not contexts:
            _fail("empty_generation_batch", "generation context batch is empty")
        context_lengths = tuple(len(context) for context in contexts)
        if any(length <= 0 for length in context_lengths):
            _fail(
                "invalid_generation_contexts",
                "generation contexts must have positive lengths",
            )
        cache_sequence_length = getattr(self.cache, "sequence_length", None)
        cache_has_physical_room = (
            cache_sequence_length is None
            or int(cache_sequence_length) < int(self.model.config.block_size)
        )
        extend_cache = (
            self.inference_forward is not None
            and self.cache is not None
            and cache_has_physical_room
            and self.cached_context_lengths is not None
            and len(context_lengths) == len(self.cached_context_lengths)
            and all(
                current_length == cached_length + 1
                for current_length, cached_length in zip(
                    context_lengths,
                    self.cached_context_lengths,
                )
            )
        )
        if extend_cache:
            model_inputs = [[int(context[-1])] for context in contexts]
            inputs = torch.tensor(model_inputs, dtype=torch.long, device=self.device)
            logits, self.cache = self.inference_forward(inputs, self.cache)
        else:
            # This branch also implements the exact sliding-window policy.  Once
            # a full window shifts, prior representations have stale reset
            # positions, so the whole cropped window is rebuilt from position 0.
            self.cache = None
            self.cached_context_lengths = None
            same_length = len(set(context_lengths)) == 1
            if self.inference_forward is None or (
                not same_length and not self.supports_left_padding
            ):
                return self._legacy_scores(contexts)
            if same_length:
                model_inputs = [
                    [int(token_id) for token_id in context]
                    for context in contexts
                ]
                attention_mask = None
            else:
                maximum_length = max(context_lengths)
                model_inputs = []
                valid_rows = []
                for context, context_length in zip(contexts, context_lengths):
                    padding_length = maximum_length - context_length
                    model_inputs.append(
                        [self.padding_token_id] * padding_length
                        + [int(token_id) for token_id in context]
                    )
                    valid_rows.append(
                        [False] * padding_length + [True] * context_length
                    )
                attention_mask = torch.tensor(
                    valid_rows,
                    dtype=torch.bool,
                    device=self.device,
                )
            inputs = torch.tensor(model_inputs, dtype=torch.long, device=self.device)
            if attention_mask is None:
                logits, self.cache = self.inference_forward(inputs, None)
            else:
                logits, self.cache = self.inference_forward(
                    inputs,
                    None,
                    attention_mask=attention_mask,
                )
        self.cached_context_lengths = context_lengths
        return logits[:, -1]

    def retain_rows(self, rows: Sequence[int]) -> None:
        """Drop EOS rows from cached tensors while preserving active-row order."""

        if self.cache is None:
            return
        select_rows = getattr(self.cache, "select_rows", None)
        if not callable(select_rows):
            # A third-party model may offer forward_inference without a cache
            # row selector.  Rebuilding is slower but preserves exact semantics.
            self.cache = None
            self.cached_context_lengths = None
            return
        self.cache = select_rows(rows)
        if self.cached_context_lengths is not None:
            self.cached_context_lengths = tuple(
                self.cached_context_lengths[index] for index in rows
            )


@torch.no_grad()
def generate_response(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    tokenizer: BPETokenizer,
    special_token_ids: Mapping[str, int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    device: torch.device,
    use_kv_cache: bool = True,
) -> dict[str, Any]:
    """Generate deterministically for a fixed seed and sampling configuration."""

    if max_new_tokens <= 0 or temperature <= 0 or top_k < 0:
        _fail("invalid_generation_arguments", "generation arguments are invalid")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    current_ids = [int(token_id) for token_id in prompt_ids]
    generated_ids: list[int] = []
    eos_id = int(special_token_ids["<EOS>"])
    was_training = bool(model.training)
    model.eval()
    stopped_on_eos = False
    decode_session = _InferenceDecodeSession(
        model,
        device,
        use_kv_cache=use_kv_cache,
        padding_token_id=int(special_token_ids["<PAD>"]),
    )
    try:
        for _ in range(max_new_tokens):
            block_size = int(model.config.block_size)
            context = current_ids[-block_size:]
            scores = decode_session.next_token_scores([context])[0]
            next_id = _sample_next_token_id(
                scores,
                special_token_ids,
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            if next_id == eos_id:
                stopped_on_eos = True
                break
            generated_ids.append(next_id)
            current_ids.append(next_id)
    finally:
        model.train(was_training)
    return _generation_result(
        generated_ids,
        tokenizer,
        stopped_on_eos=stopped_on_eos,
        max_new_tokens=max_new_tokens,
    )


@torch.no_grad()
def _generate_responses_batch(
    model: torch.nn.Module,
    prompt_ids_batch: Sequence[Sequence[int]],
    tokenizer: BPETokenizer,
    special_token_ids: Mapping[str, int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seeds: Sequence[int],
    device: torch.device,
    use_kv_cache: bool = True,
    require_same_length: bool = False,
) -> list[dict[str, Any]]:
    """Generate one batch with an independent CPU RNG per case.

    Every active case grows by one token per forward pass. Cases that emit EOS
    leave the active batch. GPT v4 prompts may start at different lengths and
    use masked left padding; unsupported test/third-party models safely fall
    back to independent unpadded forwards.
    """

    if max_new_tokens <= 0 or temperature <= 0 or top_k < 0:
        _fail("invalid_generation_arguments", "generation arguments are invalid")
    if not prompt_ids_batch or len(prompt_ids_batch) != len(seeds):
        _fail("invalid_generation_batch", "generation batch or seed count is invalid")
    prompt_lengths = {len(prompt_ids) for prompt_ids in prompt_ids_batch}
    if require_same_length and len(prompt_lengths) != 1:
        _fail("mixed_generation_lengths", "same-length generation batch contains mixed prompts")

    current_ids = [
        [int(token_id) for token_id in prompt_ids]
        for prompt_ids in prompt_ids_batch
    ]
    generated_ids: list[list[int]] = [[] for _ in prompt_ids_batch]
    stopped_on_eos = [False for _ in prompt_ids_batch]
    generators = [
        torch.Generator(device="cpu").manual_seed(int(seed)) for seed in seeds
    ]
    active = list(range(len(prompt_ids_batch)))
    eos_id = int(special_token_ids["<EOS>"])
    was_training = bool(model.training)
    model.eval()
    decode_session = _InferenceDecodeSession(
        model,
        device,
        use_kv_cache=use_kv_cache,
        padding_token_id=int(special_token_ids["<PAD>"]),
    )
    try:
        for _ in range(max_new_tokens):
            if not active:
                break
            block_size = int(model.config.block_size)
            contexts = [current_ids[index][-block_size:] for index in active]
            scores = decode_session.next_token_scores(contexts)
            next_active: list[int] = []
            retained_cache_rows: list[int] = []
            for row_index, case_index in enumerate(active):
                next_id = _sample_next_token_id(
                    scores[row_index],
                    special_token_ids,
                    temperature=temperature,
                    top_k=top_k,
                    generator=generators[case_index],
                )
                if next_id == eos_id:
                    stopped_on_eos[case_index] = True
                    continue
                generated_ids[case_index].append(next_id)
                current_ids[case_index].append(next_id)
                next_active.append(case_index)
                retained_cache_rows.append(row_index)
            if next_active and len(next_active) != len(active):
                decode_session.retain_rows(retained_cache_rows)
            active = next_active
    finally:
        model.train(was_training)

    return [
        _generation_result(
            generated_ids[index],
            tokenizer,
            stopped_on_eos=stopped_on_eos[index],
            max_new_tokens=max_new_tokens,
        )
        for index in range(len(prompt_ids_batch))
    ]


def generate_responses_same_length_batch(
    model: torch.nn.Module,
    prompt_ids_batch: Sequence[Sequence[int]],
    tokenizer: BPETokenizer,
    special_token_ids: Mapping[str, int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seeds: Sequence[int],
    device: torch.device,
    use_kv_cache: bool = True,
) -> list[dict[str, Any]]:
    """Retain the strict same-length public contract used by existing callers."""

    return _generate_responses_batch(
        model,
        prompt_ids_batch,
        tokenizer,
        special_token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seeds=seeds,
        device=device,
        use_kv_cache=use_kv_cache,
        require_same_length=True,
    )


def generate_responses_batched(
    model: torch.nn.Module,
    prompt_ids_batch: Sequence[Sequence[int]],
    tokenizer: BPETokenizer,
    special_token_ids: Mapping[str, int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seeds: Sequence[int],
    device: torch.device,
    generation_batch_size: int,
    use_kv_cache: bool = True,
) -> list[dict[str, Any]]:
    """Pack nearby prompt lengths into heterogeneous left-padded batches."""

    if generation_batch_size <= 0:
        _fail("invalid_generation_batch_size", "generation batch size must be positive")
    if len(prompt_ids_batch) != len(seeds):
        _fail("invalid_generation_batch", "generation prompt and seed counts differ")
    if not prompt_ids_batch:
        return []
    # Sorting minimizes padding and groups rows that will reach the sliding
    # boundary at similar times. Results are restored to caller order below.
    packed_indices = sorted(
        range(len(prompt_ids_batch)),
        key=lambda index: (len(prompt_ids_batch[index]), index),
    )
    ordered_results: list[dict[str, Any] | None] = [None] * len(prompt_ids_batch)
    for start in range(0, len(packed_indices), generation_batch_size):
        chunk_indices = packed_indices[start : start + generation_batch_size]
        chunk_results = _generate_responses_batch(
            model,
            [prompt_ids_batch[index] for index in chunk_indices],
            tokenizer,
            special_token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seeds=[seeds[index] for index in chunk_indices],
            device=device,
            use_kv_cache=use_kv_cache,
            require_same_length=False,
        )
        for index, result in zip(chunk_indices, chunk_results):
            ordered_results[index] = result
    if any(result is None for result in ordered_results):  # defensive invariant
        _fail("generation_result_missing", "batched generation lost a result")
    return [result for result in ordered_results if result is not None]


def generate_responses_length_bucketed(
    model: torch.nn.Module,
    prompt_ids_batch: Sequence[Sequence[int]],
    tokenizer: BPETokenizer,
    special_token_ids: Mapping[str, int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seeds: Sequence[int],
    device: torch.device,
    generation_batch_size: int,
    use_kv_cache: bool = True,
) -> list[dict[str, Any]]:
    """Compatibility name for length-packed, cross-length batch generation."""

    return generate_responses_batched(
        model,
        prompt_ids_batch,
        tokenizer,
        special_token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seeds=seeds,
        device=device,
        generation_batch_size=generation_batch_size,
        use_kv_cache=use_kv_cache,
    )


def load_prompt_conversations(args: argparse.Namespace) -> list[list[dict[str, str]]]:
    if args.prompt:
        return [[{"role": "user", "content": text}] for text in args.prompt]
    try:
        document = json.loads(args.messages_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SFTV7SamplingError(
            "invalid_messages_json", "multi-turn prompt file is invalid JSON"
        ) from error
    messages = document.get("messages") if isinstance(document, Mapping) else document
    if not isinstance(messages, list):
        _fail("invalid_messages_document", "multi-turn prompt must contain a message list")
    reject_forbidden_public_fields(document, location="messages_document")
    return [deepcopy(messages)]


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def render_markdown(report: Mapping[str, Any]) -> str:
    rows = [
        "# SFT v7 自定义采样",
        "",
        f"Checkpoint step：`{report['checkpoint_step']}`",
        "",
        "| # | 输入轮数 | 最后问题 | 模型输出 | EOS | 截断 |",
        "|---:|---:|---|---|---|---|",
    ]
    for index, sample in enumerate(report["samples"], 1):
        last_question = sample["messages"][-1]["content"]
        rows.append(
            f"| {index} | {len(sample['messages'])} | {_escape_markdown(last_question)} | "
            f"{_escape_markdown(sample['generated_text'])} | "
            f"{'是' if sample['stopped_on_eos'] else '否'} | "
            f"{'是' if sample['truncated'] else '否'} |"
        )
    return "\n".join(rows) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--public-tensors", type=Path, default=DEFAULT_PUBLIC_TENSORS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--checkpoint-mode",
        choices=("sft-v7", "pretrain-baseline"),
        default="sft-v7",
    )
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", action="append")
    prompts.add_argument("--messages-json", type=Path)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0 or args.temperature <= 0 or args.top_k < 0:
        _fail("invalid_arguments", "sampling length, temperature, or top-k is invalid")
    reject_forbidden_public_fields(
        {
            "public_tensor_path": args.public_tensors,
            "checkpoint_path": args.checkpoint,
            "messages_path": args.messages_json,
        },
        location="arguments",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.checkpoint_mode == "pretrain-baseline" and args.checkpoint == DEFAULT_CHECKPOINT:
        args.checkpoint = DEFAULT_BASELINE_CHECKPOINT
    validate_args(args)
    run_id = generate_run_id("sft-v7-sample")
    config = load_config(args.config)
    loggers = configure_module_loggers(
        args.log_dir,
        run_id,
        resolve_module_log_levels(
            {"generation": "INFO", "evaluation": "INFO", "checkpoint": "INFO"}
        ),
        max_bytes=int(config["logging"]["max_bytes"]),
        backup_count=int(config["logging"]["backup_count"]),
        console=bool(config["logging"]["console"]),
    )
    try:
        payload = load_public_payload(args.public_tensors)
        tokenizer = load_bound_tokenizer(payload)
        device = select_device(args.device)
        model, checkpoint, provenance = load_model_bundle(
            args.config, args.checkpoint, payload, device, args.checkpoint_mode
        )
        checkpoint_sha256 = file_sha256(args.checkpoint)
        loggers["checkpoint"].info(
            "checkpoint validated step=%d sha256=%s base_sha256=%s device=%s",
            provenance["step"],
            checkpoint_sha256,
            provenance["base_checkpoint_sha256"],
            device,
        )
        conversations = load_prompt_conversations(args)
        samples: list[dict[str, Any]] = []
        for index, messages in enumerate(conversations):
            prompt_ids = build_conversation_prompt_ids(
                tokenizer, messages, payload["special_token_ids"]
            )
            generated = generate_response(
                model,
                prompt_ids,
                tokenizer,
                payload["special_token_ids"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed + index,
                device=device,
            )
            samples.append({"messages": deepcopy(messages), **generated})
            loggers["generation"].info(
                "sample generated index=%d input_turns=%d prompt_tokens=%d output_tokens=%d "
                "eos=%s truncated=%s",
                index,
                len(messages),
                len(prompt_ids),
                generated["generated_tokens"],
                generated["stopped_on_eos"],
                generated["truncated"],
            )
        report = {
            "schema_version": "sft-v7-custom-samples/v1",
            "status": "complete",
            "run_id": run_id,
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": int(checkpoint["step"]),
            "checkpoint_mode": args.checkpoint_mode,
            "public_tensor_path": str(args.public_tensors),
            "public_tensor_sha256": file_sha256(args.public_tensors),
            "tokenizer_sha256": payload["tokenizer_sha256"],
            "bpe_token_manifest_sha256": payload["bpe_token_manifest_sha256"],
            "sft_dataset_manifest_sha256": payload[
                "sft_dataset_manifest_sha256"
            ],
            "device": str(device),
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "seed": args.seed,
                "masked_special_tokens": [
                    "<UNK>",
                    "<BOS>",
                    "<USER>",
                    "<ASSISTANT>",
                    "<PAD>",
                ],
                "eos_allowed": True,
            },
            "samples": samples,
        }
        atomic_write_json(args.output_json, report)
        atomic_write_text(args.output_markdown, render_markdown(report))
        loggers["evaluation"].info(
            "sample artifacts written count=%d eos=%d truncated=%d json=%s markdown=%s",
            len(samples),
            sum(bool(sample["stopped_on_eos"]) for sample in samples),
            sum(bool(sample["truncated"]) for sample in samples),
            args.output_json,
            args.output_markdown,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "checkpoint_step": int(checkpoint["step"]),
                    "samples": len(samples),
                    "output_json": str(args.output_json),
                    "output_markdown": str(args.output_markdown),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        loggers["evaluation"].error(
            "sampling failed error_code=%s error_type=%s",
            getattr(error, "code", "unexpected_failure"),
            type(error).__name__,
        )
        raise
    finally:
        close_module_loggers(loggers)


if __name__ == "__main__":
    raise SystemExit(main())
