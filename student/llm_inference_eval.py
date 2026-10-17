"""Compatibility CLI for historical character BLEU/ROUGE-L evaluation."""
import argparse
import json
from decoder_llm.checkpoint import load_checkpoint
from decoder_llm.data import parse_chat_pairs
from decoder_llm.metrics import bleu_char, rouge_l_char, _ngram_counts
from decoder_llm.evaluation import evaluate_model_on_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--continuation", type=int, default=80)
    parser.add_argument("--num_eval_samples", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    for dataset in args.datasets:
        metrics = evaluate_model_on_dataset(args.checkpoint, dataset, args.continuation,
                                            args.num_eval_samples)
        print(json.dumps({"dataset": dataset, **metrics}))


if __name__ == "__main__":
    main()
