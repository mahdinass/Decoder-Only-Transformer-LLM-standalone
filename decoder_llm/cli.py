"""Command-line entry points for local training, generation and measurements."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from .benchmark import benchmark_cache
from .checkpoint import convert_legacy, load_checkpoint
from .evaluation import evaluate_loss, evaluate_model_on_dataset, text_eval_batches
from .experiment import TrainConfig, run_training, set_seed
from .generation import generate


def _write_report(report, path=None):
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def parser():
    root = argparse.ArgumentParser(prog="decoder-llm", description="Character-level decoder Transformer experiments")
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train on plain text or tagged prompt/target pairs")
    train.add_argument("--dataset", "-d", required=True)
    train.add_argument("--validation", help="Optional separate held-out UTF-8 file")
    train.add_argument("--output", "-o", default="runs/default/checkpoint.pt")
    train.add_argument("--model-config", help="JSON object with LLMConfig fields; vocabulary inferred from train data")
    for name in ("context_length", "d_model", "num_layers", "num_heads", "d_ff"):
        train.add_argument("--" + name.replace("_", "-"), type=int)
    train.add_argument("--dropout", type=float)
    for name in ("steps", "batch_size", "seed", "val_batches", "eval_every", "save_every"):
        train.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(TrainConfig(), name))
    for name in ("learning_rate", "weight_decay", "grad_clip", "val_fraction"):
        train.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(TrainConfig(), name))
    train.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--resume", help="Resume a matching versioned checkpoint")
    train.add_argument("--stop-after", type=int, help="Stop and save at this absolute step (total --steps unchanged)")

    gen = commands.add_parser("generate", help="Generate a continuation")
    gen.add_argument("checkpoint")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-new-tokens", type=int, default=100)
    gen.add_argument("--strategy", choices=["greedy", "sample", "beam"], default="greedy")
    gen.add_argument("--temperature", type=float, default=0.9)
    gen.add_argument("--top-k", type=int)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--beam-size", type=int, default=5)
    gen.add_argument("--length-penalty", type=float, default=1.2)
    gen.add_argument("--repetition-penalty", type=float, default=1.0,
                     help="Discourage previously seen characters with a value greater than 1")
    gen.add_argument("--eos-token-id", type=int)
    gen.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    gen.add_argument("--append-separator", action=argparse.BooleanOptionalAction, default=True)
    gen.add_argument("--seed", type=int, default=1337)

    evaluate = commands.add_parser("evaluate", help="Measure held-out next-character loss or paired-text metrics")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("dataset")
    evaluate.add_argument("--mode", choices=["text", "pairs"], default="text")
    evaluate.add_argument("--continuation", type=int, default=80)
    evaluate.add_argument("--num-eval-samples", type=int, default=100)
    evaluate.add_argument("--report", help="Also save the JSON report")

    bench = commands.add_parser("benchmark", help="Measure cached versus uncached greedy decoding")
    bench.add_argument("checkpoint")
    bench.add_argument("--prompt", required=True)
    bench.add_argument("--max-new-tokens", type=int, default=32)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--warmup", type=int, default=1)
    bench.add_argument("--report")

    convert = commands.add_parser("convert-legacy", help="Migrate an explicitly trusted local pickle checkpoint")
    convert.add_argument("source")
    convert.add_argument("destination")
    convert.add_argument("--trust-pickle", action="store_true", required=True,
                         help="Acknowledge that loading this file may execute code")
    for command in (train, gen, evaluate, bench):
        command.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
        command.add_argument("--threads", type=int, default=1, help="CPU intra-op threads (recorded in reports)")
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "convert-legacy":
            convert_legacy(args.source, args.destination, trust_pickle=args.trust_pickle)
            return
        if args.threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)
        if args.command == "train":
            values = {"context_length": 64, "d_model": 128, "num_layers": 2,
                      "num_heads": 4, "d_ff": 512, "dropout": 0.1}
            if args.model_config:
                values.update(json.loads(Path(args.model_config).read_text(encoding="utf-8")))
            for key in ("context_length", "d_model", "num_layers", "num_heads", "d_ff", "dropout"):
                if getattr(args, key) is not None:
                    values[key] = getattr(args, key)
            config = TrainConfig(**{key: getattr(args, key) for key in asdict(TrainConfig())})
            report = run_training(args.dataset, args.output, model_values=values, config=config,
                                  validation_path=args.validation, device=args.device,
                                  resume=args.resume, stop_after=args.stop_after)
            _write_report({"step": report["step"], "checkpoint": args.output,
                           "metrics": report["history"][-1]})
            return
        model, tokenizer, _ = load_checkpoint(args.checkpoint, args.device)
        if args.command in ("generate", "benchmark"):
            if not args.prompt:
                raise ValueError("prompt must be nonempty")
            prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)
        if args.command == "generate":
            set_seed(args.seed)
            tokens = generate(model, prompt, max_new_tokens=args.max_new_tokens,
                              strategy=args.strategy, temperature=args.temperature,
                              top_k=args.top_k, top_p=args.top_p, beam_size=args.beam_size,
                              length_penalty=args.length_penalty, use_cache=args.cache,
                              eos_token_id=args.eos_token_id, append_separator=args.append_separator,
                              repetition_penalty=args.repetition_penalty)
            print(tokenizer.decode(tokens[0].tolist()))
        elif args.command == "benchmark":
            _write_report(benchmark_cache(model, prompt, max_new_tokens=args.max_new_tokens,
                                         repeats=args.repeats, warmup=args.warmup), args.report)
        elif args.mode == "pairs":
            _write_report(evaluate_model_on_dataset(args.checkpoint, args.dataset,
                                                   continuation=args.continuation,
                                                   num_eval_samples=args.num_eval_samples,
                                                   device=args.device), args.report)
        else:
            text = Path(args.dataset).read_text(encoding="utf-8")
            metrics = evaluate_loss(model, text_eval_batches(text, tokenizer, model.config.context_length))
            metrics["context_policy"] = "nonoverlapping chunks; context resets; every target scored once"
            _write_report(metrics, args.report)
    except (ValueError, OSError, RuntimeError, TypeError, KeyError) as exc:
        root.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
