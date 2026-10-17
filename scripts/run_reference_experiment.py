"""Reproduce a controlled synthetic-language experiment without external datasets.

The train/validation/test split is deterministic and disjoint by complete sentence.
This measures learning a small generated grammar, not natural-language quality.
Run from the repository root using the installed package environment.
"""
import argparse
from dataclasses import asdict
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import statistics
import torch
from decoder_llm import CharTokenizer, LLMConfig, build_model, generate
from decoder_llm.benchmark import benchmark_cache
from decoder_llm.checkpoint import load_checkpoint
from decoder_llm.evaluation import evaluate_loss, text_eval_batches
from decoder_llm.experiment import TrainConfig, environment, run_training, set_seed


MODEL = dict(context_length=64, d_model=64, num_layers=2, num_heads=4,
             d_ff=128, dropout=0.1)


def corpus():
    """Return fixed sentence splits; no external text or data is incorporated."""
    subjects = ("robot", "pilot", "artist", "baker")
    colors = ("red", "blue", "green", "gold")
    actions = ("finds", "moves", "counts", "makes")
    objects = ("box", "book", "cup", "lamp", "key", "ball")
    sentences = [f"the {color} {subject} {action} the {obj}.\n"
                 for subject, color, action, obj in product(subjects, colors, actions, objects)]
    random.Random(20261017).shuffle(sentences)
    a, b = int(len(sentences) * 0.8), int(len(sentences) * 0.9)
    splits = {"train": sentences[:a], "validation": sentences[a:b], "test": sentences[b:]}
    assert len(set(sentences)) == len(sentences)
    assert not (set(splits["train"]) & set(splits["test"]))
    assert set("".join(splits["test"] + splits["validation"])) <= set("".join(splits["train"]))
    return splits


def source_fingerprint():
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / "decoder_llm").glob("*.py")) + [Path(__file__).resolve()]
    return [{"path": path.relative_to(root).as_posix(),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]


def run(output, steps, seeds):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    splits = corpus()
    for name, lines in splits.items():
        # Canonical bytes avoid Windows CRLF conversion changing the vocabulary.
        (output / f"{name}.txt").write_bytes("".join(lines).encode("utf-8"))
    train = (output / "train.txt").read_bytes().decode("utf-8")
    test = (output / "test.txt").read_bytes().decode("utf-8")
    tokenizer = CharTokenizer(train)
    results = []
    for seed in seeds:
        print(f"Running reference experiment: seed={seed}, steps={steps}", flush=True)
        set_seed(seed)
        baseline = build_model(LLMConfig(vocab_size=tokenizer.vocab_size, **MODEL))
        untrained = evaluate_loss(baseline, text_eval_batches(test, tokenizer, MODEL["context_length"]))
        config = TrainConfig(steps=steps, batch_size=16, learning_rate=0.001, seed=seed,
                             val_batches=4, eval_every=100, save_every=100)
        checkpoint = output / f"seed-{seed}.pt"
        history = run_training(output / "train.txt", checkpoint, validation_path=output / "validation.txt",
                               model_values=MODEL, config=config)
        model, tok, _ = load_checkpoint(checkpoint)
        if tok.chars != tokenizer.chars:
            raise RuntimeError("baseline and trained model vocabularies differ")
        final = evaluate_loss(model, text_eval_batches(test, tok, MODEL["context_length"]))
        prompt = "the red robot "
        ids = torch.tensor([tok.encode(prompt)])
        sample = tok.decode(generate(model, ids, max_new_tokens=64, strategy="greedy")[0].tolist())
        result = {"seed": seed, "train_config": asdict(config), "untrained_test": untrained,
                  "trained_test": final, "final_validation": history["history"][-1]["validation"],
                  "greedy_sample": sample}
        if seed == seeds[0]:
            result["cache_benchmarks"] = {
                "within_context": benchmark_cache(model, ids, max_new_tokens=24, repeats=5, warmup=2),
                "with_rollover": benchmark_cache(model, ids, max_new_tokens=80, repeats=5, warmup=2)}
        results.append(result)
    report = {"experiment": "synthetic-grammar-v1", "environment": environment(),
              "source_sha256": source_fingerprint(), "model_config": asdict(model.config),
              "split_seed": 20261017, "split_policy": "complete unique sentences; fixed 80/10/10 ordered after seeded shuffle",
              "splits": {name: {"sentences": len(lines),
                                  "sha256": hashlib.sha256("".join(lines).encode()).hexdigest()}
                         for name, lines in splits.items()},
              "evaluation_policy": "all test next-character targets once, nonoverlapping context chunks",
              "test_selection": "final fixed training step; no checkpoint selection using test loss",
              "runs": results,
              "summary": {"trained_test_loss_mean": statistics.mean(r["trained_test"]["loss"] for r in results),
                          "trained_test_loss_stdev": statistics.stdev(r["trained_test"]["loss"] for r in results) if len(results) > 1 else None,
                          "untrained_test_loss_mean": statistics.mean(r["untrained_test"]["loss"] for r in results)},
              "limitations": ["Controlled synthetic grammar, not translation or natural-language quality.",
                              "One CPU/hardware/software environment; no GPU validation.",
                              "Few seeds, small model and short budget; timing confidence intervals not estimated."]}
    path = output / "reference.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Reference report saved: {path}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/reference")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("steps must be positive and seeds must be distinct")
    run(args.output, args.steps, args.seeds)
