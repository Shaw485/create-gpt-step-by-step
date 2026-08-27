import json
import logging
from logging.handlers import RotatingFileHandler
import os
import time
from pathlib import Path

import torch


TENSOR_PATH = os.getenv("TRAIN_TENSOR_PATH", "data/clean/doupo_stage3_tensors.pt")
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
CHECKPOINT_PREFIX = os.getenv("CHECKPOINT_PREFIX", "gpt_stage3")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "")
BEST_CHECKPOINT_PATH = os.getenv("BEST_CHECKPOINT_PATH", "")

BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "64"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "64"))
NUM_HEADS = int(os.getenv("NUM_HEADS", "4"))
NUM_LAYERS = int(os.getenv("NUM_LAYERS", "2"))

MAX_STEPS = int(os.getenv("MAX_STEPS", "300"))
EVAL_INTERVAL = int(os.getenv("EVAL_INTERVAL", "100"))
EVAL_ITERS = int(os.getenv("EVAL_ITERS", "20"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "3e-4"))
GRAD_CLIP = float(os.getenv("GRAD_CLIP", "1.0"))
RESUME_TRAINING = os.getenv("RESUME_TRAINING", "0") == "1"
RESUME_FROM = os.getenv("RESUME_FROM", "")
GEN_STEPS = int(os.getenv("GEN_STEPS", "0"))
TOP_K = int(os.getenv("TOP_K", "0"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))
GEN_START_TEXT = os.getenv("GEN_START_TEXT", "小")
SEED = int(os.getenv("SEED", "42"))
DETERMINISTIC = os.getenv("DETERMINISTIC", "1") == "1"

PROMPT_EVAL_FILE = os.getenv("PROMPT_EVAL_FILE", "")
PROMPT_EVAL_EVERY = int(os.getenv("PROMPT_EVAL_EVERY", "0"))
PROMPT_EVAL_MAX_NEW_TOKENS = int(os.getenv("PROMPT_EVAL_MAX_NEW_TOKENS", "30"))
PROMPT_EVAL_TEMPERATURE = float(os.getenv("PROMPT_EVAL_TEMPERATURE", "1.0"))
PROMPT_EVAL_TOP_K = int(os.getenv("PROMPT_EVAL_TOP_K", "40"))
PROMPT_EVAL_OUT_PATH = os.getenv("PROMPT_EVAL_OUT_PATH", "checkpoints/prompt_eval_history.json")

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def configure_logging() -> None:
    """Set up separate loggers with rotation.

    This keeps data-loading, training, and checkpoint logs independent.
    """
    Path("logs").mkdir(exist_ok=True, parents=True)

    logger_names = ("train.data", "train.train", "train.ckpt")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    root.addHandler(console_handler)

    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        file_handler = RotatingFileHandler(
            f"logs/{logger_name.replace('.', '_')}.log",
            maxBytes=1_000_000,
            backupCount=3,
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        )
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.propagate = False


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set deterministic-ish RNG behaviour for repeatable runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False


class AttentionHead(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, context_size: int):
        super().__init__()
        self.output_size = output_size
        self.key = torch.nn.Linear(input_size, output_size, bias=False)
        self.query = torch.nn.Linear(input_size, output_size, bias=False)
        self.value = torch.nn.Linear(input_size, output_size, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_size, context_size)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        scores = queries @ keys.transpose(-2, -1)
        scores *= self.output_size ** -0.5

        sequence_length = x.shape[1]
        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ values


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, context_size: int):
        super().__init__()
        if embedding_size % num_heads != 0:
            raise ValueError("embedding_size must be divisible by num_heads")

        self.head_size = embedding_size // num_heads
        self.heads = torch.nn.ModuleList(
            [
                AttentionHead(embedding_size, self.head_size, context_size)
                for _ in range(num_heads)
            ]
        )
        self.proj = torch.nn.Linear(embedding_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(x) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        return self.proj(concat)


class FeedForward(torch.nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        hidden = 4 * embedding_size
        self.network = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, context_size: int):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(embedding_size)
        self.attention = MultiHeadAttention(embedding_size, num_heads, context_size)
        self.ln2 = torch.nn.LayerNorm(embedding_size)
        self.ff = FeedForward(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, num_heads: int,
                 context_size: int, num_layers: int):
        super().__init__()
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.token_embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.pos_embedding = torch.nn.Embedding(context_size, embedding_size)
        self.blocks = torch.nn.Sequential(
            *[
                TransformerBlock(embedding_size, num_heads, context_size)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = torch.nn.LayerNorm(embedding_size)
        self.head = torch.nn.Linear(embedding_size, vocab_size)

    def forward(
        self,
        token_indices: torch.Tensor,
        target_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length = token_indices.shape
        if sequence_length > self.context_size:
            raise ValueError("sequence length exceeds context window")

        x = self.token_embedding(token_indices)
        pos = torch.arange(sequence_length, device=token_indices.device)
        x = x + self.pos_embedding(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if target_indices is not None:
            logits_flat = logits.reshape(batch_size * sequence_length, self.vocab_size)
            targets_flat = target_indices.reshape(batch_size * sequence_length)
            loss = torch.nn.functional.cross_entropy(
                logits_flat, targets_flat
            )

        return logits, loss


def pick_checkpoint_path(prefix: str) -> Path:
    """Resolve checkpoint path from explicit env/path defaults."""
    if CHECKPOINT_PATH:
        return Path(CHECKPOINT_PATH)
    return CHECKPOINT_DIR / f"{prefix}.pt"


def resolve_resume_path() -> Path:
    """Choose checkpoint path for resume or fallback to default."""
    if RESUME_FROM:
        return Path(RESUME_FROM)
    if CHECKPOINT_PATH:
        return Path(CHECKPOINT_PATH)
    return CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}.pt"


def pick_best_checkpoint_path(prefix: str) -> Path:
    """Resolve the independently stored best-validation checkpoint path."""
    if BEST_CHECKPOINT_PATH:
        return Path(BEST_CHECKPOINT_PATH)
    return CHECKPOINT_DIR / f"{prefix}_best.pt"


def build_checkpoint_payload(
    model: torch.nn.Module,
    step: int,
    vocab_size: int,
    history: list[dict],
    best_val_loss: float,
) -> dict:
    """Build one consistent payload for latest and best checkpoints."""
    return {
        "model_state_dict": model.state_dict(),
        "meta": {
            "step": step,
            "vocab_size": vocab_size,
            "block_size": BLOCK_SIZE,
            "batch_size": BATCH_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "deterministic": DETERMINISTIC,
            "best_val_loss": best_val_loss,
        },
        "loss_history": history,
    }


def save_best_checkpoint_if_improved(
    model: torch.nn.Module,
    checkpoint_path: Path,
    step: int,
    val_loss: float,
    best_val_loss: float,
    vocab_size: int,
    history: list[dict],
    logger: logging.Logger,
) -> tuple[float, bool]:
    """Save model weights only when validation loss reaches a new minimum."""
    if val_loss >= best_val_loss:
        logger.info(
            "best checkpoint unchanged step=%d val_loss=%.6f best_val_loss=%.6f",
            step,
            val_loss,
            best_val_loss,
        )
        return best_val_loss, False

    payload = build_checkpoint_payload(
        model=model,
        step=step,
        vocab_size=vocab_size,
        history=history,
        best_val_loss=val_loss,
    )
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, checkpoint_path)
    except Exception:
        logger.exception(
            "best checkpoint save failed path=%s step=%d val_loss=%.6f",
            checkpoint_path,
            step,
            val_loss,
        )
        raise

    logger.info(
        "best checkpoint saved path=%s step=%d val_loss=%.6f previous=%.6f",
        checkpoint_path,
        step,
        val_loss,
        best_val_loss,
    )
    return val_loss, True


def get_batch(data: torch.Tensor, batch_size: int, block_size: int):
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError("dataset too short for current block size")

    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
) -> float:
    losses = []
    model.eval()
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, block_size)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def decode_ids(ids, itos):
    return "".join(itos.get(int(token_id), "?") for token_id in ids)


def load_prompt_inputs(prompt_path: str) -> list[str]:
    if not prompt_path:
        return []
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {prompt_path}")
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return lines[:10]


def encode_prompt(prompt: str, stoi: dict[str, int]) -> list[int]:
    ids: list[int] = []
    missing: list[str] = []
    for char in prompt:
        token_id = stoi.get(char)
        if token_id is None:
            missing.append(char)
        else:
            ids.append(int(token_id))
    if missing:
        raise ValueError(
            f"prompt has chars not in vocab: {missing[:10]}{'...' if len(missing)>10 else ''}"
        )
    return ids


@torch.no_grad()
def generate_from_prompt(
    model: GPTLanguageModel,
    prompt_ids: list[int],
    itos: dict[int, str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    block_size: int,
) -> str:
    current_ids = prompt_ids.copy()
    if max_new_tokens <= 0 or not current_ids:
        return ""

    model.eval()
    for _ in range(max_new_tokens):
        context = current_ids[-block_size:]
        context_tensor = torch.tensor(
            [context],
            dtype=torch.long,
            device=model.token_embedding.weight.device,
        )
        logits, _ = model(context_tensor)
        logits = logits[:, -1, :] / max(1e-8, temperature)

        if top_k > 0:
            values, indices = torch.topk(logits, top_k, dim=-1)
            probs = torch.softmax(values, dim=-1)
            next_index = torch.multinomial(probs, num_samples=1)
            next_id = indices.gather(1, next_index)[0, 0].item()
        else:
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)[0, 0].item()

        current_ids.append(int(next_id))

    model.train()
    prompt_len = len(prompt_ids)
    return decode_ids(current_ids[prompt_len:], itos)


def run_prompt_eval(
    step: int,
    model: GPTLanguageModel,
    prompts: list[str],
    stoi: dict[str, int],
    itos: dict[int, str],
    block_size: int,
    logger: logging.Logger,
) -> list[dict]:
    if not prompts:
        return []

    records: list[dict] = []
    for index, prompt in enumerate(prompts, start=1):
        prompt_ids = encode_prompt(prompt, stoi)
        continuation = generate_from_prompt(
            model=model,
            prompt_ids=prompt_ids,
            itos=itos,
            max_new_tokens=PROMPT_EVAL_MAX_NEW_TOKENS,
            temperature=PROMPT_EVAL_TEMPERATURE,
            top_k=PROMPT_EVAL_TOP_K,
            block_size=block_size,
        )
        full_text = prompt + continuation
        records.append(
            {
                "prompt_index": index,
                "prompt": prompt,
                "continuation": continuation,
                "full": full_text,
            }
        )
        logger.info(
            "prompt_eval step=%d idx=%d prompt=%s continuation=%s",
            step,
            index,
            prompt,
            continuation,
        )
    return records


@torch.no_grad()
def generate_text(
    model: GPTLanguageModel,
    start_token_id: int,
    itos: dict[int, str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    block_size: int,
) -> str:
    current_ids = [start_token_id]
    if max_new_tokens <= 0:
        return decode_ids(current_ids, itos)

    model.eval()
    for _ in range(max_new_tokens):
        context = current_ids[-block_size:]
        context_tensor = torch.tensor(
            [context],
            dtype=torch.long,
            device=model.token_embedding.weight.device,
        )
        logits, _ = model(context_tensor)
        logits = logits[:, -1, :] / max(1e-8, temperature)

        if top_k > 0:
            values, indices = torch.topk(logits, top_k, dim=-1)
            probs = torch.softmax(values, dim=-1)
            next_index = torch.multinomial(probs, num_samples=1)
            next_id = indices.gather(1, next_index)[0, 0].item()
        else:
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)[0, 0].item()

        current_ids.append(int(next_id))

    model.train()
    return decode_ids(current_ids, itos)


def main() -> None:
    configure_logging()
    data_logger = logging.getLogger("train.data")
    train_logger = logging.getLogger("train.train")
    ckpt_logger = logging.getLogger("train.ckpt")

    set_global_seed(SEED, deterministic=DETERMINISTIC)

    if BATCH_SIZE <= 0 or BLOCK_SIZE <= 0:
        raise ValueError("BATCH_SIZE and BLOCK_SIZE must be positive")
    if EMBEDDING_DIM <= 0:
        raise ValueError("EMBEDDING_DIM must be positive")

    payload = torch.load(TENSOR_PATH, weights_only=False)
    train_data = payload["train_data"]
    val_data = payload["val_data"]
    vocab_size = payload["vocab_size"]
    stoi = payload["stoi"]
    itos = payload["itos"]

    if not isinstance(stoi, dict) or not isinstance(itos, dict):
        raise TypeError("stoi and itos must be dicts from prepare_training_data")

    prompt_inputs = load_prompt_inputs(PROMPT_EVAL_FILE) if PROMPT_EVAL_FILE else []
    if prompt_inputs:
        data_logger.info("prompt_eval prompts=%d path=%s every=%d", len(prompt_inputs), PROMPT_EVAL_FILE, PROMPT_EVAL_EVERY)

    prompt_records: list[dict] = []

    data_logger.info("tensor loaded path=%s", TENSOR_PATH)
    data_logger.info(
        "seed=%d deterministic=%s train=%d val=%d vocab=%d batch=%d block=%d emb=%d heads=%s layers=%d",
        SEED,
        DETERMINISTIC,
        len(train_data),
        len(val_data),
        vocab_size,
        BATCH_SIZE,
        BLOCK_SIZE,
        EMBEDDING_DIM,
        NUM_HEADS,
        NUM_LAYERS,
    )
    data_logger.info("dtype train=%s val=%s device=%s", train_data.dtype, val_data.dtype, DEVICE)

    train_data = train_data.to(DEVICE)
    val_data = val_data.to(DEVICE)

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        embedding_size=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        context_size=BLOCK_SIZE,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    start_step = 0
    if RESUME_TRAINING:
        resume_path = resolve_resume_path()
        data_logger.info("resuming from=%s", resume_path)
        ckpt_resume = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt_resume["model_state_dict"])
        start_step = int(ckpt_resume["meta"]["step"]) + 1
        data_logger.info(
            "resume step=%d embedding=%d heads=%d layers=%d",
            start_step,
            int(ckpt_resume["meta"]["embedding_dim"]),
            int(ckpt_resume["meta"]["num_heads"]),
            int(ckpt_resume["meta"]["num_layers"]),
        )

    total_params = sum(param.numel() for param in model.parameters())
    data_logger.info("model params=%d", total_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = pick_checkpoint_path(CHECKPOINT_PREFIX)
    best_ckpt_path = pick_best_checkpoint_path(CHECKPOINT_PREFIX)
    history = []
    if RESUME_TRAINING and ckpt_path.exists():
        existing_ckpt = torch.load(ckpt_path, map_location=DEVICE)
        history = existing_ckpt.get("loss_history", [])

    best_val_loss = float("inf")
    if best_ckpt_path.exists():
        existing_best = torch.load(best_ckpt_path, map_location=DEVICE)
        best_val_loss = float(
            existing_best.get("meta", {}).get("best_val_loss", float("inf"))
        )
        ckpt_logger.info(
            "best checkpoint loaded path=%s best_val_loss=%.6f",
            best_ckpt_path,
            best_val_loss,
        )
    elif RESUME_TRAINING and history:
        latest_entry = history[-1]
        best_val_loss, _ = save_best_checkpoint_if_improved(
            model=model,
            checkpoint_path=best_ckpt_path,
            step=start_step - 1,
            val_loss=float(latest_entry["val_loss"]),
            best_val_loss=float("inf"),
            vocab_size=vocab_size,
            history=history,
            logger=ckpt_logger,
        )

    for step in range(start_step, MAX_STEPS + 1):
        x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        if step % EVAL_INTERVAL == 0:
            train_loss = estimate_loss(
                model,
                train_data,
                BATCH_SIZE,
                BLOCK_SIZE,
                EVAL_ITERS,
            )
            val_loss = estimate_loss(
                model,
                val_data,
                BATCH_SIZE,
                BLOCK_SIZE,
                EVAL_ITERS,
            )
            entry = {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time": time.time(),
            }
            history.append(entry)
            train_logger.info(
                "step=%d train_loss=%.6f val_loss=%.6f",
                step,
                train_loss,
                val_loss,
            )
            best_val_loss, _ = save_best_checkpoint_if_improved(
                model=model,
                checkpoint_path=best_ckpt_path,
                step=step,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                vocab_size=vocab_size,
                history=history,
                logger=ckpt_logger,
            )

        if PROMPT_EVAL_EVERY > 0 and step % PROMPT_EVAL_EVERY == 0 and prompt_inputs:
            try:
                eval_entries = run_prompt_eval(
                    step=step,
                    model=model,
                    prompts=prompt_inputs,
                    stoi=stoi,
                    itos=itos,
                    block_size=BLOCK_SIZE,
                    logger=train_logger,
                )
                prompt_records.append(
                    {"step": step, "time": time.time(), "results": eval_entries}
                )
            except Exception as exc:
                data_logger.error("prompt_eval failed at step=%d error=%s", step, exc)

    sample_ids, _ = get_batch(train_data, 1, BLOCK_SIZE)
    sample_ids = sample_ids[0][:20].tolist()

    torch.save(
        build_checkpoint_payload(
            model=model,
            step=step,
            vocab_size=vocab_size,
            history=history,
            best_val_loss=best_val_loss,
        ),
        ckpt_path,
    )

    with open(CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    if prompt_records:
        with open(PROMPT_EVAL_OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(prompt_records, f, ensure_ascii=False, indent=2)

    ckpt_logger.info("checkpoint saved=%s", ckpt_path)
    train_logger.info("final sample ids=%s", sample_ids)
    train_logger.info("final sample text=%s", decode_ids(sample_ids, itos))
    train_logger.info("final loss=%s", loss.item())

    if GEN_STEPS > 0:
        start_id = payload["stoi"].get(GEN_START_TEXT, list(payload["stoi"].values())[0])
        prompt_text = GEN_START_TEXT
        generated = generate_text(
            model,
            int(start_id),
            itos,
            GEN_STEPS,
            TEMPERATURE,
            TOP_K,
            BLOCK_SIZE,
        )
        train_logger.info("prompt=%s", prompt_text)
        train_logger.info("generated=%s", generated)


if __name__ == "__main__":
    main()
