"""Hardware regression checks; skipped explicitly on CPU-only machines."""
import pytest
import torch

from decoder_llm import CharTokenizer, compute_loss
from decoder_llm.benchmark import benchmark_cache
from decoder_llm.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from decoder_llm.cli import main
from decoder_llm.experiment import TrainConfig, run_training, set_seed
from decoder_llm.generation import generate

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@pytest.fixture
def gpu_model(model):
    set_seed(17)
    return model.to("cuda").eval()


def test_cuda_training_updates_finite_parameters(gpu_model):
    ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    before = gpu_model.token_embed.weight.detach().clone()
    optimizer = torch.optim.AdamW(gpu_model.parameters(), lr=1e-3)
    loss = compute_loss(gpu_model(ids)["logits"], ids.roll(-1, 1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in gpu_model.parameters())
    optimizer.step()
    assert not torch.equal(before, gpu_model.token_embed.weight)


def test_cuda_causality_and_chunked_cache(gpu_model):
    ids = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 5, 4, 3, 2, 1]], device="cuda")
    with torch.inference_mode():
        full = gpu_model(ids)["logits"]
        prefix = gpu_model(ids[:, :3])
        suffix = gpu_model(ids[:, 3:], past_kv=prefix["past_kv"])
        torch.testing.assert_close(suffix["logits"], full[:, 3:], atol=2e-5, rtol=2e-4)
        ids[:, 3:] = 0
        torch.testing.assert_close(gpu_model(ids)["logits"][:, :3], full[:, :3])
        assert all(k.is_cuda and v.is_cuda for k, v in suffix["past_kv"])


@pytest.mark.parametrize("count", [3, 12])
def test_cuda_generation_cache_equivalence(gpu_model, count):
    ids = torch.tensor([[0, 1, 2]], device="cuda")
    options = dict(max_new_tokens=count, strategy="greedy", append_separator=False)
    assert torch.equal(generate(gpu_model, ids, use_cache=True, **options),
                       generate(gpu_model, ids, use_cache=False, **options))


def test_cuda_sampling_reproducible(gpu_model):
    ids = torch.tensor([[0, 1, 2]], device="cuda")
    def sample():
        return generate(gpu_model, ids, max_new_tokens=12, strategy="sample", top_k=4,
                        top_p=0.9, generator=torch.Generator(device="cuda").manual_seed(29),
                        append_separator=False)
    assert torch.equal(sample(), sample())


def test_cuda_checkpoint_portability(gpu_model, tmp_path):
    path = tmp_path / "gpu.pt"
    save_checkpoint(path, gpu_model, CharTokenizer.from_chars(list("abcdefg")))
    ids = torch.tensor([[1, 2, 3]], device="cuda")
    expected = gpu_model(ids)["logits"].detach().cpu()
    for device in ("cpu", "cuda"):
        restored, _, _ = load_checkpoint(path, device)
        restored.eval()
        actual = restored(ids.to(device))["logits"].detach().cpu()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


def test_cuda_exact_training_resume(tmp_path):
    text = tmp_path / "train.txt"
    text.write_text("abcde " * 50, encoding="utf-8")
    config = TrainConfig(steps=4, batch_size=2, val_batches=2, eval_every=2, save_every=2)
    values = dict(context_length=8, d_model=16, num_layers=1, num_heads=4, d_ff=32, dropout=0.2)
    full, partial, resumed = (tmp_path / name for name in ("full.pt", "partial.pt", "resumed.pt"))
    options = dict(model_values=values, config=config, device="cuda")
    run_training(text, full, **options)
    run_training(text, partial, stop_after=2, **options)
    run_training(text, resumed, resume=partial, **options)
    a, b = read_checkpoint(full), read_checkpoint(resumed)
    for name in a["model_state_dict"]:
        torch.testing.assert_close(a["model_state_dict"][name], b["model_state_dict"][name], atol=0, rtol=0)
    assert a["training_state"]["history"] == b["training_state"]["history"]


def test_cuda_benchmark_reports_memory(gpu_model):
    report = benchmark_cache(gpu_model, torch.tensor([[0, 1, 2]], device="cuda"),
                             max_new_tokens=12, repeats=2, warmup=1)
    assert report["outputs_equal"] and report["context_rollover"]
    assert report["device"].startswith("cuda")
    for mode in ("cached", "uncached"):
        assert report[mode]["peak_cuda_allocated_bytes"] > 0
        assert all(t > 0 for t in report[mode]["seconds"])


def test_cuda_cli_workflow(tmp_path, capsys):
    text, checkpoint = tmp_path / "text.txt", tmp_path / "cli.pt"
    text.write_text("abcde " * 50, encoding="utf-8")
    main(["train", "-d", str(text), "-o", str(checkpoint), "--device", "cuda",
          "--steps", "2", "--batch-size", "2", "--val-batches", "1",
          "--context-length", "8", "--d-model", "16", "--num-layers", "1",
          "--num-heads", "4", "--d-ff", "32"])
    main(["generate", str(checkpoint), "--device", "cuda", "--prompt", "abc",
          "--strategy", "sample", "--max-new-tokens", "4"])
    assert "abc" in capsys.readouterr().out
    main(["evaluate", str(checkpoint), str(text), "--device", "cuda",
          "--report", str(tmp_path / "evaluation.json")])
    main(["benchmark", str(checkpoint), "--device", "cuda", "--prompt", "abc",
          "--max-new-tokens", "4", "--repeats", "1", "--warmup", "0",
          "--report", str(tmp_path / "benchmark.json")])
    assert (tmp_path / "evaluation.json").is_file()
    assert (tmp_path / "benchmark.json").is_file()
