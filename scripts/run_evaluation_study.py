"""Train and evaluate fixed three-seed grammar and Tiny Shakespeare studies.

External text is fetched into ignored runs/, verified by SHA-256, never bundled.
No checkpoint selection or hyperparameter search uses test metrics.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from decoder_llm import CharTokenizer, LLMConfig, build_model, generate
from decoder_llm.benchmark import benchmark_cache
from decoder_llm.checkpoint import load_checkpoint
from decoder_llm.evaluation import text_eval_batches
from decoder_llm.experiment import TrainConfig, environment, run_training, set_seed
from scripts.run_reference_experiment import MODEL, corpus

TEXT_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
            "370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt")
TEXT_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def fetch_text(path):
    path = Path(path)
    if not path.exists():
        with urllib.request.urlopen(TEXT_URL, timeout=30) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("unexpected upstream data size")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEXT_SHA256:
        raise ValueError("Tiny Shakespeare SHA-256 mismatch")
    return raw.decode("utf-8")


def prediction_report(loss_sum, correct, count):
    if count < 1:
        raise ValueError("evaluation needs at least one target")
    loss = loss_sum / count
    return {"loss": loss, "perplexity": math.exp(loss), "bits_per_character": loss / math.log(2),
            "accuracy": correct / count, "tokens": count}


def fit_baselines(train_ids, vocab_size, alpha=1.0):
    """Fit training-only Laplace-smoothed unigram and first-order Markov models."""
    if alpha <= 0 or vocab_size < 1 or len(train_ids) < 2:
        raise ValueError("positive smoothing/vocabulary and at least two tokens required")
    ids = torch.tensor(train_ids, dtype=torch.long)
    unigram = torch.bincount(ids, minlength=vocab_size).double() + alpha
    unigram = (unigram / unigram.sum()).expand(vocab_size, -1)
    bigram = torch.bincount(ids[:-1] * vocab_size + ids[1:],
                           minlength=vocab_size ** 2).reshape(vocab_size, vocab_size).double() + alpha
    bigram = bigram / bigram.sum(1, keepdim=True)
    return {"uniform": torch.full((vocab_size, vocab_size), 1 / vocab_size, dtype=torch.float64),
            "unigram": unigram, "bigram": bigram}


def score_baseline(probabilities, ids):
    if len(ids) < 2:
        raise ValueError("evaluation needs at least two tokens")
    tokens = torch.tensor(ids, dtype=torch.long)
    rows = probabilities[tokens[:-1]]
    loss = -rows.gather(1, tokens[1:, None]).log().sum().item()
    correct = (rows.argmax(1) == tokens[1:]).sum().item()
    return prediction_report(loss, correct, len(ids) - 1)


@torch.inference_mode()
def score_model(model, text, tokenizer):
    training = model.training
    loss, correct, count = 0.0, 0, 0
    model.eval()
    try:
        for batch in text_eval_batches(text, tokenizer, model.config.context_length):
            target = batch["target_ids"]
            logits = model(batch["input_ids"])["logits"]
            loss += F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum").item()
            correct += (logits.argmax(-1) == target).sum().item()
            count += target.numel()
    finally:
        model.train(training)
    return prediction_report(loss, correct, count)


def grammar_samples(model, tokenizer, splits, seed):
    valid = set(sum(splits.values(), []))
    train = set(splits["train"])
    prompts = [f"the {color} {subject} " for color in ("red", "blue", "green", "gold")
               for subject in ("robot", "pilot", "artist", "baker")]
    strategies = {}
    for strategy in ("greedy", "sample"):
        examples = []
        for index, prompt in enumerate(prompts):
            tokens = generate(model, torch.tensor([tokenizer.encode(prompt)]), max_new_tokens=64,
                              strategy=strategy, temperature=0.8, top_k=10, top_p=0.95,
                              append_separator=False, eos_token_id=tokenizer.encode("\n")[0],
                              generator=torch.Generator().manual_seed(seed * 100 + index))
            sentence = tokenizer.decode(tokens[0].tolist())
            examples.append({"prompt": prompt, "text": sentence, "complete": sentence.endswith("\n"),
                             "valid_grammar": sentence in valid,
                             "valid_unseen_sentence": sentence in valid and sentence not in train})
        strategies[strategy] = {"count": len(examples),
            "valid_count": sum(e["valid_grammar"] for e in examples),
            "complete_count": sum(e["complete"] for e in examples),
            "valid_unseen_count": sum(e["valid_unseen_sentence"] for e in examples), "examples": examples}
    return strategies


def summarize(runs):
    result = {}
    for metric in ("loss", "perplexity", "bits_per_character", "accuracy"):
        values = [run["test"][metric] for run in runs]
        result[metric] = {"mean": statistics.mean(values),
                          "stdev": statistics.stdev(values) if len(values) > 1 else None,
                          "values": values}
    return result


def run(output, *, seeds=(17, 29, 43), grammar_steps=300, text_steps=1000,
        datasets=("grammar", "shakespeare"), cache_repeats=5):
    if not seeds or len(set(seeds)) != len(seeds) or min(grammar_steps, text_steps, cache_repeats) < 1:
        raise ValueError("distinct nonempty seeds and positive step/repeat counts required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    sources = sorted((ROOT / "decoder_llm").rglob("*.py")) + [Path(__file__).resolve(), ROOT / "scripts/run_reference_experiment.py"]
    report = {"study": "character-evaluation-v1", "environment": environment(),
              "source_sha256": [{"path": p.relative_to(ROOT).as_posix(),
                                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
              "protocol": {"seeds": list(seeds), "model_values": MODEL, "smoothing_alpha": 1.0,
                "test_policy": "all next-character targets once; Transformer context resets every 64 targets",
                "selection": "fixed final step; test metrics never select checkpoints or settings",
                "accuracy": "teacher-forced next-character top-1; ties choose lowest token ID",
                "limitations": ["One small model family, three seeds and one CPU runtime.",
                    "Contiguous Shakespeare splits can share vocabulary, characters and motifs.",
                    "Generation grammar validity is a narrow mechanical check, not semantic quality.",
                    "Cache timings are one-process samples, not independent hardware trials."]},
              "datasets": {}, "cache_sweep": []}
    for name in datasets:
        folder = output / name
        folder.mkdir(exist_ok=True)
        if name == "grammar":
            splits = corpus()
            texts = {key: "".join(lines) for key, lines in splits.items()}
            source = {"kind": "generated", "split_seed": 20261017,
                      "split_policy": "complete unique sentences; fixed shuffled 80/10/10 split"}
            steps = grammar_steps
        elif name == "shakespeare":
            text = fetch_text(output / "tinyshakespeare.txt")
            a, b = int(len(text) * 0.9), int(len(text) * 0.95)
            texts = {"train": text[:a], "validation": text[a:b], "test": text[b:]}
            source = {"url": TEXT_URL, "sha256": TEXT_SHA256,
                      "split_policy": "contiguous character offsets: first 90% train, next 5% validation, final 5% test"}
            steps = text_steps
        else:
            raise ValueError(f"unknown dataset: {name}")
        for split, text in texts.items():
            (folder / f"{split}.txt").write_bytes(text.encode("utf-8"))
        tokenizer = CharTokenizer(texts["train"])
        train_ids, test_ids = tokenizer.encode(texts["train"]), tokenizer.encode(texts["test"])
        tokenizer.encode(texts["validation"])
        baseline = {key: score_baseline(probs, test_ids) for key, probs in
                    fit_baselines(train_ids, tokenizer.vocab_size).items()}
        baseline["uniform"]["accuracy"] = None  # Uniform ties have no informative top-1 prediction.
        record = {"source": source, "splits": {key: {"characters": len(text),
                  "sha256": hashlib.sha256(text.encode()).hexdigest()} for key, text in texts.items()},
                  "vocabulary_size": tokenizer.vocab_size, "baselines": baseline, "runs": []}
        for seed in seeds:
            print(f"{name}: seed={seed}, steps={steps}", flush=True)
            set_seed(seed)
            model = build_model(LLMConfig(vocab_size=tokenizer.vocab_size, **MODEL))
            untrained = score_model(model, texts["test"], tokenizer)
            config = TrainConfig(steps=steps, batch_size=16, learning_rate=0.001, seed=seed,
                                 val_batches=4, eval_every=100, save_every=100)
            checkpoint = folder / f"seed-{seed}.pt"
            start = time.perf_counter()
            history = run_training(folder / "train.txt", checkpoint, validation_path=folder / "validation.txt",
                                   model_values=MODEL, config=config)
            elapsed = time.perf_counter() - start
            model, tok, _ = load_checkpoint(checkpoint)
            assert tok.chars == tokenizer.chars
            result = {"seed": seed, "train_config": asdict(config), "training_seconds": elapsed,
                      "parameter_count": sum(p.numel() for p in model.parameters()),
                      "untrained_test": untrained, "test": score_model(model, texts["test"], tok),
                      "validation_curve": [r for r in history["history"] if "validation" in r]}
            if name == "grammar":
                result["generation"] = grammar_samples(model, tok, splits, seed)
                if seed == seeds[0]:
                    for batch_size in (1, 4):
                        for prompt_length in (8, 32):
                            ids = torch.tensor([train_ids[:prompt_length]] * batch_size)
                            for count in (16, 48, 80):
                                report["cache_sweep"].append(benchmark_cache(model, ids, max_new_tokens=count,
                                                                            repeats=cache_repeats, warmup=2))
            else:
                result["samples"] = []
                for index, prompt in enumerate(("First Citizen:\n", "ROMEO:\n", "To be, or not to be")):
                    ids = torch.tensor([tok.encode(prompt)])
                    for strategy in ("greedy", "sample"):
                        tokens = generate(model, ids, max_new_tokens=160, strategy=strategy,
                                          temperature=0.8, top_k=10, top_p=0.95, append_separator=False,
                                          generator=torch.Generator().manual_seed(seed * 100 + index))
                        result["samples"].append({"prompt": prompt, "strategy": strategy,
                                                   "text": tok.decode(tokens[0].tolist())})
            record["runs"].append(result)
        record["summary"] = summarize(record["runs"])
        record["loss_reduction_vs_bigram_percent"] = 100 * (1 - record["summary"]["loss"]["mean"] / baseline["bigram"]["loss"])
        report["datasets"][name] = record
        (output / "study.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation-study"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--grammar-steps", type=int, default=300)
    parser.add_argument("--text-steps", type=int, default=1000)
    parser.add_argument("--datasets", nargs="+", choices=["grammar", "shakespeare"], default=["grammar", "shakespeare"])
    parser.add_argument("--cache-repeats", type=int, default=5)
    args = parser.parse_args()
    run(args.output, seeds=args.seeds, grammar_steps=args.grammar_steps, text_steps=args.text_steps,
        datasets=args.datasets, cache_repeats=args.cache_repeats)
