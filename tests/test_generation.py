import pytest
import torch
from decoder_llm import generate
from decoder_llm.benchmark import benchmark_cache


@pytest.mark.parametrize("prompt_length,count", [(3, 4), (7, 7), (12, 4)])
def test_cached_greedy_matches_uncached_across_rollover(model, prompt_length, count):
    ids = torch.arange(prompt_length).remainder(7).unsqueeze(0).repeat(2, 1)
    cached = generate(model, ids, count, strategy="greedy", use_cache=True)
    plain = generate(model, ids, count, strategy="greedy", use_cache=False)
    assert torch.equal(cached, plain)
    assert model.training
    assert cached.shape == (2, prompt_length + count)


def test_seeded_sampling_and_top_k(model):
    ids = torch.tensor([[1, 2]])
    def sample():
        return generate(model, ids, 6, strategy="sample", top_p=0.8, top_k=3,
                        generator=torch.Generator().manual_seed(2))
    assert torch.equal(sample(), sample())
    greedy = generate(model, ids, 6, strategy="greedy")
    assert torch.equal(greedy, generate(model, ids, 6, strategy="sample", top_k=1))


def test_batched_eos_and_separator(model):
    model.config.sep_token_id = 3
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    result = generate(model, torch.tensor([[1], [2]]), 5, strategy="greedy", eos_token_id=0)
    assert result.tolist() == [[1, 3, 0], [2, 3, 0]]


def test_legacy_beam_search_and_zero_tokens(model):
    ids = torch.tensor([[1, 2]])
    # Beam width larger than vocabulary used to fail with torch.topk.
    assert generate(model, ids, 2, beam_size=10, top_k=1).shape == (1, 4)
    assert torch.equal(generate(model, ids, 0), ids)
    with pytest.raises(ValueError, match="batch size"):
        generate(model, ids.repeat(2, 1))


@pytest.mark.parametrize("args", [{"temperature": 0}, {"top_p": 0}, {"top_k": 0},
                                  {"max_new_tokens": -1}, {"strategy": "bad"},
                                  {"beam_size": 0}, {"eos_token_id": 7}])
def test_invalid_generation(model, args):
    with pytest.raises(ValueError):
        generate(model, torch.tensor([[1]]), **args)


def test_benchmark_records_measurements_and_equivalence(model):
    report = benchmark_cache(model, torch.tensor([[1, 2]]), max_new_tokens=3, repeats=2, warmup=0)
    assert report["outputs_equal"]
    assert report["includes_prefill"]
    assert len(report["cached"]["seconds"]) == 2
    assert report["cached"]["median_seconds"] > 0
    assert report["cached"]["peak_cuda_allocated_bytes"] is None
    assert report["model_config"]["d_model"] == model.config.d_model
    assert report["parameter_count"] == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("strategy", ["greedy", "sample", "beam"])
@pytest.mark.parametrize("scores", [(4.0, 3.0), (-1.0, -1.5)])
def test_repetition_penalty_changes_decoding_for_both_logit_signs(model, strategy, scores, monkeypatch):
    def fixed_logits(ids, past_kv=None):
        logits = torch.full((*ids.shape, model.config.vocab_size), -20.0, device=ids.device)
        logits[..., 0], logits[..., 1] = scores
        return {"logits": logits, "past_kv": None}

    monkeypatch.setattr(model, "forward", fixed_logits)
    prompt = torch.tensor([[0]])
    options = dict(max_new_tokens=1, strategy=strategy, top_k=1, beam_size=1, use_cache=False)
    assert generate(model, prompt, **options)[0, -1].item() == 0
    assert generate(model, prompt, repetition_penalty=2.0, **options)[0, -1].item() == 1
    assert model.training


@pytest.mark.parametrize("prompt", [torch.tensor([[1.0]]), torch.tensor([[-1]]), torch.tensor([[7]])])
def test_invalid_prompt_rejected_even_when_generation_is_empty(model, prompt):
    with pytest.raises(ValueError, match="torch.long|outside the vocabulary"):
        generate(model, prompt, max_new_tokens=0)


@pytest.mark.parametrize("penalty", [0.5, float("nan"), float("inf")])
def test_invalid_repetition_penalty(model, penalty):
    with pytest.raises(ValueError, match="repetition_penalty"):
        generate(model, torch.tensor([[1]]), repetition_penalty=penalty)
