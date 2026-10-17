"""Token-weighted next-character evaluation and historical paired-text metrics."""
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from .checkpoint import load_checkpoint
from .data import parse_chat_pairs
from .generation import generate
from .metrics import bleu_char, rouge_l_char


@torch.inference_mode()
def evaluate_loss(model, batches, *, device=None, max_batches=None):
    """Weight cross entropy by supervised tokens, excluding ignored padding."""
    device = device or next(model.parameters()).device
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for step, batch in enumerate(batches):
            if max_batches is not None and step >= max_batches:
                break
            target = batch["target_ids"].to(device)
            logits = model(batch["input_ids"].to(device))["logits"]
            total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                                     ignore_index=-100, reduction="sum").item()
            count += (target != -100).sum().item()
    finally:
        model.train(training)
    if count == 0:
        raise ValueError("evaluation contains no supervised tokens")
    loss = total / count
    return {"loss": loss, "perplexity": math.exp(loss) if loss < 709 else float("inf"),
            "tokens": count}


def text_eval_batches(text, tokenizer, context_length):
    """Score every next-token target exactly once, including the final short chunk.

    Context resets between nonoverlapping chunks; the report documents this policy.
    """
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(ids) < 2:
        raise ValueError("evaluation text needs at least two characters")
    for start in range(0, len(ids) - 1, context_length):
        end = min(start + context_length, len(ids) - 1)
        yield {"input_ids": ids[start:end].unsqueeze(0),
               "target_ids": ids[start + 1:end + 1].unsqueeze(0)}


def evaluate_model_on_dataset(checkpoint, dataset_path, continuation=80,
                              num_eval_samples=5, device=None, print_samples=0):
    """Retain character BLEU/ROUGE-L, reporting skipped OOV prompts explicitly.

    Decode a fixed continuation budget without consulting the reference length.
    """
    if continuation < 1 or num_eval_samples < 1:
        raise ValueError("continuation and num_eval_samples must be positive")
    model, tokenizer, _ = load_checkpoint(checkpoint, device or "cpu")
    pairs = parse_chat_pairs(Path(dataset_path).read_text(encoding="utf-8"))[:num_eval_samples]
    if not pairs:
        raise ValueError("dataset contains no prompt/target pairs")
    bleu, rouge, skipped = [], [], 0
    for prompt, reference in pairs:
        try:
            ids = tokenizer.encode(prompt)
        except ValueError:
            skipped += 1
            continue
        tokens = generate(model, torch.tensor([ids]), max_new_tokens=continuation,
                          strategy="beam", temperature=0.8, top_k=40)
        prefix_length = len(ids) + int(model.config.sep_token_id >= 0)
        hypothesis = tokenizer.decode(tokens[0, prefix_length:].tolist())
        bleu.append(bleu_char(reference, hypothesis))
        rouge.append(rouge_l_char(reference, hypothesis))
        if print_samples and len(bleu) <= print_samples:
            print(f"Prompt: {prompt}\nTarget: {reference}\nGenerated: {hypothesis}")
    if not bleu:
        raise ValueError(f"all {skipped} prompts contain out-of-vocabulary characters")
    return {"bleu": sum(bleu) / len(bleu), "rouge_l": sum(rouge) / len(rouge),
            "evaluated": len(bleu), "skipped": skipped}
