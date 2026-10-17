import math
import pytest
import torch
from decoder_llm import CharTokenizer
from decoder_llm.evaluation import evaluate_loss, text_eval_batches
from scripts.run_evaluation_study import fetch_text, fit_baselines, score_baseline, score_model


def test_smoothed_bigram_matches_hand_calculation():
    models = fit_baselines([0, 1, 0, 1], 2)
    torch.testing.assert_close(models["bigram"], torch.tensor([[0.25, 0.75], [2 / 3, 1 / 3]], dtype=torch.float64))
    for matrix in models.values():
        torch.testing.assert_close(matrix.sum(1), torch.ones(2, dtype=torch.float64))
    report = score_baseline(models["bigram"], [0, 1, 0])
    assert report["tokens"] == 2 and report["accuracy"] == 1
    assert report["loss"] == pytest.approx((-math.log(0.75) - math.log(2 / 3)) / 2)
    assert report["bits_per_character"] == pytest.approx(report["loss"] / math.log(2))


def test_unseen_bigram_is_finite_and_scoring_does_not_refit():
    models = fit_baselines([0, 0, 0], 2)
    before = models["bigram"].clone()
    report = score_baseline(models["bigram"], [1, 1, 1])
    assert math.isfinite(report["loss"]) and report["tokens"] == 2
    torch.testing.assert_close(before, models["bigram"], atol=0, rtol=0)
    with pytest.raises(ValueError):
        fit_baselines([0], 2)
    with pytest.raises(ValueError):
        score_baseline(models["bigram"], [0])


def test_diagnostic_loss_matches_existing_evaluator_and_restores_mode(model):
    tokenizer = CharTokenizer.from_chars(list("abcdefg"))
    text = "abcdefgabcdefgabc"
    expected = evaluate_loss(model, text_eval_batches(text, tokenizer, 8))
    actual = score_model(model, text, tokenizer)
    assert model.training and model.token_embed.weight.grad is None
    assert actual["tokens"] == len(text) - 1 == expected["tokens"]
    assert actual["loss"] == pytest.approx(expected["loss"])
    assert actual["perplexity"] == pytest.approx(math.exp(actual["loss"]))
    assert 0 <= actual["accuracy"] <= 1


def test_external_corpus_requires_exact_checksum(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("changed or incomplete download", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch_text(path)
