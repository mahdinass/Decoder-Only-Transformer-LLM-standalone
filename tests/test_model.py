import pytest
import torch
from decoder_llm import LLMConfig, compute_loss
from decoder_llm.model import RMSNorm


@pytest.mark.parametrize("values", [{"d_model": 7, "num_heads": 2}, {"num_layers": 0},
                                    {"vocab_size": -1}, {"dropout": 1.0},
                                    {"dropout": float("nan")}, {"sep_token_id": -2},
                                    {"context_length": True}, {"num_heads": 0}])
def test_invalid_config(values):
    with pytest.raises(ValueError):
        LLMConfig(**values)


def test_future_tokens_do_not_affect_prefix(model):
    model.eval()
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5]])
    original = model(tokens)["logits"]
    tokens[:, 3:] = 6
    torch.testing.assert_close(original[:, :3], model(tokens)["logits"][:, :3])


def test_cached_chunk_matches_full_prefix(model):
    model.eval()
    ids = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 5, 4, 3, 2, 1]])
    full = model(ids)["logits"]
    prefix = model(ids[:, :3])
    suffix = model(ids[:, 3:], past_kv=prefix["past_kv"])
    torch.testing.assert_close(suffix["logits"], full[:, 3:], atol=1e-6, rtol=1e-5)
    assert all(k.shape == (2, 4, 6, 4) for k, _ in suffix["past_kv"])


def test_cache_overflow_and_invalid_layers(model):
    ids = torch.zeros(1, 8, dtype=torch.long)
    past = model(ids)["past_kv"]
    with pytest.raises(ValueError, match="exceeds"):
        model(ids[:, :1], past_kv=past)
    with pytest.raises(ValueError, match="per layer"):
        model(ids, past_kv=[])
    with pytest.raises(ValueError, match="shape"):
        model(ids[:, :1], past_kv=[(torch.zeros(1), torch.zeros(1))] * 2)
    prefix = model(ids[:, :2])["past_kv"]
    prefix[1] = tuple(x[:, :, :1] for x in prefix[1])
    with pytest.raises(ValueError, match="lengths"):
        model(ids[:, :1], past_kv=prefix)


@pytest.mark.parametrize("ids", [torch.empty(1, 0, dtype=torch.long), torch.zeros(2),
                                 torch.zeros(1, 2, dtype=torch.int32)])
def test_invalid_inputs(model, ids):
    with pytest.raises(ValueError):
        model(ids)


def test_tied_weights_receive_gradients(model):
    assert model.lm_head.weight is model.token_embed.weight
    ids = torch.tensor([[1, 2, 3]])
    compute_loss(model(ids)["logits"], torch.tensor([[2, 3, 4]])).backward()
    assert torch.isfinite(model.token_embed.weight.grad).all()
    assert model.blocks[0].attn.qkv.weight.grad.abs().sum() > 0


def test_rmsnorm_half_precision_avoids_square_overflow():
    value = torch.full((2, 4), 1000.0, dtype=torch.float16)
    assert torch.isfinite(RMSNorm(4)(value)).all()
    torch.testing.assert_close(RMSNorm(4)(value), torch.ones(2, 4), atol=1e-3, rtol=1e-3)


def test_masked_loss_and_noncontiguous_tensors():
    logits = torch.randn(2, 4, 7).transpose(0, 1)
    target = torch.tensor([[1, 2, -100, 3], [4, 5, 6, -100]]).transpose(0, 1)
    expected = torch.nn.functional.cross_entropy(logits[target != -100], target[target != -100])
    torch.testing.assert_close(compute_loss(logits, target), expected)
    with pytest.raises(ValueError, match="no supervised"):
        compute_loss(logits, torch.full_like(target, -100))
