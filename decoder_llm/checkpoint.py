"""Versioned tensor/primitive checkpoints; no automatic pickle fallback."""
from dataclasses import asdict
from pathlib import Path
import os
import pickle
import tempfile
import torch
from .model import LLMConfig, build_model
from .tokenizer import CharTokenizer

FORMAT_VERSION = 1


def save_checkpoint(path, model, tokenizer, *, training_state=None, metadata=None):
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("model and tokenizer vocabulary sizes differ")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": FORMAT_VERSION, "config": asdict(model.config),
               "tokenizer_chars": tokenizer.chars, "model_state_dict": model.state_dict(),
               "training_state": training_state or {}, "metadata": metadata or {}}
    # Replace atomically in the same directory, so interruption preserves the old file.
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_checkpoint(path):
    # An explicit restricted loader is used even on versions where it is not default.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError("restricted checkpoint load failed; use convert-legacy only for your own trusted files") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint; legacy object checkpoints require explicit conversion")
    for name in ("config", "tokenizer_chars", "model_state_dict", "training_state", "metadata"):
        if name not in payload:
            raise ValueError(f"checkpoint is missing {name}")
    return payload


def load_checkpoint(path, map_location="cpu"):
    payload = read_checkpoint(path)
    config = LLMConfig(**payload["config"])
    tokenizer = CharTokenizer.from_chars(payload["tokenizer_chars"])
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("checkpoint vocabulary does not match model configuration")
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(map_location)
    return model, tokenizer, config


def convert_legacy(source, destination, *, trust_pickle=False):
    """Explicit, local-only migration for checkpoints the caller created and trusts."""
    if not trust_pickle:
        raise ValueError("legacy conversion requires trust_pickle=True; pickle can execute code")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    config = payload["config"]
    config = LLMConfig(**(config if isinstance(config, dict) else asdict(config)))
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"])
    tokenizer = CharTokenizer.from_chars(payload["tokenizer_chars"])
    save_checkpoint(destination, model, tokenizer, metadata={"converted_legacy": True})
