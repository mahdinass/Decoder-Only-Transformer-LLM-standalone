"""Seeded experiments with data fingerprints and resumable optimizer/RNG state."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import torch
from .checkpoint import read_checkpoint, save_checkpoint
from .data import CharacterDataset, TranslationLineDataset, parse_chat_pairs
from .evaluation import evaluate_loss
from .model import LLMConfig, build_model, compute_loss
from .tokenizer import CharTokenizer


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    seed: int = 1337
    val_fraction: float = 0.1
    val_batches: int = 10
    eval_every: int = 100
    save_every: int = 100
    deterministic: bool = True

    def __post_init__(self):
        for key in ("steps", "batch_size", "val_batches", "eval_every", "save_every"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be in (0, 1)")
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0
                or not math.isfinite(self.grad_clip) or self.grad_clip <= 0):
            raise ValueError("optimizer values must be finite; learning_rate/grad_clip positive, decay nonnegative")


def set_seed(seed=1337, deterministic=True):
    # Set before CUDA initialization. Unsupported deterministic operations fail loudly.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def environment():
    root = Path(__file__).resolve().parent.parent
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                             stderr=subprocess.DEVNULL))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "platform": platform.platform(), "cuda_runtime": torch.version.cuda,
            "git_revision": revision, "git_dirty": dirty,
            "cpu_threads": torch.get_num_threads()}


def _prepare(train_text, validation_text, config, model_values):
    pairs = parse_chat_pairs(train_text)
    if pairs:
        # Deduplicate before splitting so identical pairs do not leak across partitions.
        train = list(dict.fromkeys(pairs))
        if validation_text is None:
            split = int(len(train) * (1 - config.val_fraction))
            train, val = train[:split], train[split:]
        else:
            val = list(dict.fromkeys(parse_chat_pairs(validation_text)))
        if not train or not val:
            raise ValueError("paired training requires nonempty train and validation partitions")
        if set(train) & set(val):
            raise ValueError("identical pairs overlap train and validation partitions")
        tokenizer = CharTokenizer("\n" + "".join(p + t for p, t in train))
        sep = tokenizer.encode("\n")[0]
        dataset_cls = TranslationLineDataset
        mode = "pairs"
    else:
        if validation_text is None:
            split = int(len(train_text) * (1 - config.val_fraction))
            train, val = train_text[:split], train_text[split:]
        else:
            train, val = train_text, validation_text
        tokenizer = CharTokenizer(train)
        sep = -1
        dataset_cls = CharacterDataset
        mode = "text"
    values = dict(model_values)
    supplied_vocab = values.pop("vocab_size", tokenizer.vocab_size)
    supplied_sep = values.pop("sep_token_id", sep)
    if supplied_vocab != tokenizer.vocab_size or supplied_sep != sep:
        raise ValueError("vocab_size and sep_token_id must match the training data")
    model_config = LLMConfig(vocab_size=tokenizer.vocab_size, sep_token_id=sep, **values)
    generator = torch.Generator().manual_seed(config.seed + 1)
    train_dataset = dataset_cls(train, tokenizer, model_config.context_length,
                                batch_size=config.batch_size,
                                num_samples=config.steps * config.batch_size, generator=generator)

    def validation():
        # Identical validation windows at each evaluation; independent of training RNG.
        return dataset_cls(val, tokenizer, model_config.context_length,
                           batch_size=config.batch_size,
                           num_samples=config.val_batches * config.batch_size,
                           generator=torch.Generator().manual_seed(config.seed + 2))

    # Validate OOV and minimum corpus length before doing any optimization.
    validation()
    split_hashes = {"train": hashlib.sha256(repr(train).encode()).hexdigest(),
                    "validation": hashlib.sha256(repr(val).encode()).hexdigest()}
    return tokenizer, model_config, train_dataset, validation, mode, split_hashes


def run_training(data_path, output_path, *, model_values=None, config=None,
                 validation_path=None, device="cpu", resume=None, stop_after=None):
    """Train to config.steps, or stop at an absolute step for a bounded segment.

    Resume requires unchanged data, split, model, training configuration, runtime
    versions, device type, and thread count. Exact replay across hardware is not promised.
    """
    config = config or TrainConfig()
    if stop_after is not None and not 1 <= stop_after <= config.steps:
        raise ValueError("stop_after must be between 1 and steps")
    raw = Path(data_path).read_bytes()
    validation_raw = Path(validation_path).read_bytes() if validation_path else None
    set_seed(config.seed, config.deterministic)
    tokenizer, model_config, dataset, validation, mode, split_hashes = _prepare(
        raw.decode("utf-8"), validation_raw.decode("utf-8") if validation_raw is not None else None,
        config, model_values or {"context_length": 64, "d_model": 128, "num_layers": 2,
                                  "num_heads": 4, "d_ff": 512, "dropout": 0.1})
    runtime = environment()
    identity = {"data_sha256": hashlib.sha256(raw).hexdigest(),
                "validation_sha256": hashlib.sha256(validation_raw).hexdigest() if validation_raw is not None else None,
                "split_sha256": split_hashes, "mode": mode,
                "train_config": asdict(config), "model_config": asdict(model_config),
                "torch": runtime["torch"], "python": runtime["python"],
                "device_type": torch.device(device).type, "cpu_threads": torch.get_num_threads()}
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    history, step = [], 0
    if resume:
        payload = read_checkpoint(resume)
        if payload["metadata"].get("identity") != identity:
            raise ValueError("resume configuration, data or runtime differs from checkpoint")
        model.load_state_dict(payload["model_state_dict"])
        if payload["tokenizer_chars"] != tokenizer.chars:
            raise ValueError("resume vocabulary differs")
        state = payload["training_state"]
        optimizer.load_state_dict(state["optimizer"])
        step, history = state["step"], state["history"]
        dataset.generator.set_state(state["data_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.device(device).type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
    limit = stop_after if stop_after is not None else config.steps
    if step >= limit:
        raise ValueError("checkpoint has already reached the requested step")
    batches = iter(dataset)
    model.train()
    metadata = {"identity": identity, "environment": runtime,
                "validation_policy": "fixed seeded random windows; train-only character vocabulary"}
    for step in range(step + 1, limit + 1):
        batch = next(batches)
        x, y = batch["input_ids"].to(device), batch["target_ids"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model(x)["logits"], y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        record = {"step": step, "train_loss": loss.item()}
        if step % config.eval_every == 0 or step == limit:
            record["validation"] = evaluate_loss(model, validation(), device=device)
            print(json.dumps(record), file=sys.stderr)
        history.append(record)
        if step % config.save_every == 0 or step == limit:
            state = {"step": step, "optimizer": optimizer.state_dict(), "history": history,
                     "torch_rng": torch.get_rng_state(), "data_rng": dataset.generator.get_state(),
                     "python_rng": random.getstate(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else []}
            save_checkpoint(output_path, model, tokenizer, training_state=state, metadata=metadata)
    report = {"step": step, **metadata, "history": history}
    Path(str(output_path) + ".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
