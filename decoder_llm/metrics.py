"""Character metrics with source attribution retained in PROVENANCE.md.

These are sentence-level character approximations, not standardized corpus BLEU.
"""
from typing import Dict, List, Tuple
import torch

def _ngram_counts(sequence: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i : i + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def bleu_char(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Simple character-level BLEU with uniform n-gram weights."""
    if max_n < 1:
        raise ValueError("max_n must be positive")
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not hyp_chars:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_counts = _ngram_counts(ref_chars, n)
        hyp_counts = _ngram_counts(hyp_chars, n)
        overlap = 0
        total = max(len(hyp_chars) - n + 1, 0)
        if total == 0:
            precisions.append(0.0)
            continue
        for ngram, hcount in hyp_counts.items():
            overlap += min(hcount, ref_counts.get(ngram, 0))
        precisions.append(overlap / total)

    eps = 1e-9
    geom_mean = 1.0
    for p in precisions:
        geom_mean *= max(p, eps)
    geom_mean = geom_mean ** (1.0 / max_n)

    bp = 1.0
    if len(hyp_chars) < len(ref_chars):
        bp = torch.exp(torch.tensor(1.0 - len(ref_chars) / max(len(hyp_chars), 1))).item()
    return float(bp * geom_mean)


def rouge_l_char(reference: str, hypothesis: str) -> float:
    """Character-level ROUGE-L (F1) using LCS length."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars or not hyp_chars:
        return 0.0

    n, m = len(ref_chars), len(hyp_chars)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[n][m]
    recall = lcs_len / n
    precision = lcs_len / m
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)
