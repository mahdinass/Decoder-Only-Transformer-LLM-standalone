"""Autoregressive decoding, including learned-position cache rollover."""
import math
import torch
from torch.nn import functional as F


def _penalize_repetition(logits, tokens, penalty):
    if penalty == 1.0:
        return logits
    previous = logits.gather(1, tokens)
    adjusted = torch.where(previous < 0, previous * penalty, previous / penalty)
    return logits.scatter(1, tokens, adjusted)


def _filter(logits, temperature, top_k, top_p):
    logits = logits / temperature
    if top_k is not None:
        threshold = logits.topk(min(top_k, logits.shape[-1])).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    if top_p < 1:
        values, indices = logits.sort(descending=True)
        remove = values.softmax(-1).cumsum(-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        values = values.masked_fill(remove, -torch.inf)
        logits = torch.full_like(logits, -torch.inf).scatter(-1, indices, values)
    return logits


@torch.inference_mode()
def generate(model, prompt_ids, max_new_tokens=100, temperature=0.9, top_k=None,
             device=None, do_sample=False, beam_size=5, length_penalty=1.2,
             *, strategy=None, top_p=1.0, use_cache=True, eos_token_id=None,
             generator=None, append_separator=True, repetition_penalty=1.0):
    """Return prompt plus continuation; preserve the caller's train/eval mode.

    Legacy calls default to beam search (or sampling with do_sample=True).
    New callers should pass strategy='greedy', 'sample', or 'beam' explicitly.
    At the context boundary, rebuild the complete window: learned absolute
    positions make simply evicting cached keys semantically incorrect.
    Beam search retains the original full-prefix algorithm without a KV cache.
    A repetition_penalty greater than 1 discourages characters already present
    in the prompt or continuation. The default of 1 leaves logits unchanged.
    """
    strategy = strategy or ("sample" if do_sample else "beam")
    if strategy not in ("greedy", "sample", "beam"):
        raise ValueError("strategy must be greedy, sample, or beam")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be finite and positive; top_p must be in (0, 1]")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be a positive integer")
    if not isinstance(beam_size, int) or beam_size < 1 or not math.isfinite(length_penalty):
        raise ValueError("beam_size must be positive and length_penalty finite")
    if not math.isfinite(repetition_penalty) or repetition_penalty < 1:
        raise ValueError("repetition_penalty must be finite and at least 1")
    if not isinstance(prompt_ids, torch.Tensor) or prompt_ids.dtype != torch.long:
        raise ValueError("prompt_ids must be a torch.long tensor")
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] < 1 or prompt_ids.shape[1] < 1:
        raise ValueError("prompt_ids must contain nonempty rank-2 token sequences")
    if torch.any((prompt_ids < 0) | (prompt_ids >= model.config.vocab_size)):
        raise ValueError("prompt_ids contain a token outside the vocabulary")
    if eos_token_id is not None and not 0 <= eos_token_id < model.config.vocab_size:
        raise ValueError("eos_token_id is outside the vocabulary")
    if strategy == "beam" and prompt_ids.shape[0] != 1:
        raise ValueError("beam search supports batch size 1")
    if device is not None:
        model.to(device)
    device = next(model.parameters()).device
    tokens = prompt_ids.to(device).clone()
    if append_separator and model.config.sep_token_id >= 0:
        sep = torch.full((tokens.shape[0], 1), model.config.sep_token_id,
                         dtype=torch.long, device=device)
        tokens = torch.cat((tokens, sep), dim=1)
    training = model.training
    model.eval()
    try:
        if strategy == "beam":
            return _beam(model, tokens, max_new_tokens, temperature, top_k, top_p,
                         beam_size, length_penalty, eos_token_id, repetition_penalty)
        past = None
        finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            if use_cache and past is not None and past[0][0].shape[2] < model.config.context_length:
                output = model(tokens[:, -1:], past_kv=past)
            else:
                output = model(tokens[:, -model.config.context_length:])
            past = output["past_kv"] if use_cache else None
            logits = _penalize_repetition(output["logits"][:, -1], tokens, repetition_penalty)
            logits = _filter(logits, temperature, top_k, top_p)
            next_token = (torch.multinomial(logits.softmax(-1), 1, generator=generator)
                          if strategy == "sample" else logits.argmax(-1, keepdim=True))
            if eos_token_id is not None:
                next_token[finished] = eos_token_id
                finished |= next_token[:, 0] == eos_token_id
            tokens = torch.cat((tokens, next_token), dim=1)
            if finished.all():
                break
        return tokens
    finally:
        model.train(training)


def _beam(model, prompt, count, temperature, top_k, top_p, width, penalty, eos, repetition_penalty):
    beams = [(prompt, 0.0, False)]
    initial = prompt.shape[1]

    def score(beam):
        return beam[1] / max(1, beam[0].shape[1] - initial) ** penalty

    for _ in range(count):
        candidates = []
        for tokens, logprob, done in beams:
            if done:
                candidates.append((tokens, logprob, done))
                continue
            logits = model(tokens[:, -model.config.context_length:])["logits"][:, -1]
            logits = _penalize_repetition(logits, tokens, repetition_penalty)
            probs = _filter(logits, temperature, top_k, top_p).log_softmax(-1)[0]
            values, indices = probs.topk(min(width, probs.numel()))
            for value, index in zip(values.tolist(), indices.tolist()):
                if math.isfinite(value):
                    token = torch.tensor([[index]], device=tokens.device)
                    candidates.append((torch.cat((tokens, token), 1), logprob + value, index == eos))
        beams = sorted(candidates, key=score, reverse=True)[:width]
        if all(b[2] for b in beams):
            break
    return max(beams, key=score)[0]
