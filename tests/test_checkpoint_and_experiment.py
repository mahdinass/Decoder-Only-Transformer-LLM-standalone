from dataclasses import asdict
import json
import pickle
import pytest
import torch
from decoder_llm import CharTokenizer
from decoder_llm.checkpoint import save_checkpoint, load_checkpoint, read_checkpoint, convert_legacy
from decoder_llm.cli import main
from decoder_llm.evaluation import evaluate_loss, text_eval_batches, evaluate_model_on_dataset
from decoder_llm.experiment import TrainConfig, run_training
from decoder_llm.metrics import bleu_char, rouge_l_char


VALUES = dict(context_length=8, d_model=16, num_layers=1, num_heads=4, d_ff=32, dropout=0.2)


def test_checkpoint_roundtrip(model, tmp_path):
    tokenizer = CharTokenizer.from_chars(list("gfedcba"))
    path = tmp_path / "sub" / "model.pt"
    save_checkpoint(path, model, tokenizer)
    restored, tok, config = load_checkpoint(path)
    assert asdict(config) == asdict(model.config)
    assert tok.chars == tokenizer.chars
    ids = torch.tensor([[1, 2]])
    model.eval()
    restored.eval()
    torch.testing.assert_close(model(ids)["logits"], restored(ids)["logits"], rtol=0, atol=0)
    assert not list(path.parent.glob("*.tmp"))


def test_legacy_requires_explicit_trust(model, tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"config": model.config, "model_state_dict": model.state_dict(),
                "tokenizer_chars": list("abcdefg")}, path)
    with pytest.raises((ValueError, pickle.UnpicklingError)):
        load_checkpoint(path)
    with pytest.raises(ValueError, match="trust_pickle"):
        convert_legacy(path, tmp_path / "converted.pt")
    convert_legacy(path, tmp_path / "converted.pt", trust_pickle=True)
    assert load_checkpoint(tmp_path / "converted.pt")[1].chars == list("abcdefg")


def test_unsupported_format_and_vocabulary(model, tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"format_version": 999}, path)
    with pytest.raises(ValueError, match="unsupported"):
        read_checkpoint(path)
    with pytest.raises(ValueError, match="vocabulary"):
        save_checkpoint(path, model, CharTokenizer("ab"))


def test_exact_resume_matches_uninterrupted_training(tmp_path):
    text = tmp_path / "train.txt"
    text.write_text("abcde " * 50, encoding="utf-8")
    config = TrainConfig(steps=4, batch_size=2, val_batches=2, eval_every=2, save_every=2)
    full, partial, resumed = (tmp_path / name for name in ("full.pt", "partial.pt", "resumed.pt"))
    run_training(text, full, model_values=VALUES, config=config)
    run_training(text, partial, model_values=VALUES, config=config, stop_after=2)
    run_training(text, resumed, model_values=VALUES, config=config, resume=partial)
    a, b = read_checkpoint(full), read_checkpoint(resumed)
    for name in a["model_state_dict"]:
        torch.testing.assert_close(a["model_state_dict"][name], b["model_state_dict"][name], atol=0, rtol=0)
    assert a["training_state"]["history"] == b["training_state"]["history"]
    text.write_text("abcde " * 51, encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        run_training(text, resumed, model_values=VALUES, config=config, resume=partial)


def test_training_vocabulary_does_not_use_validation(tmp_path):
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    train.write_text("abc " * 30, encoding="utf-8")
    val.write_text("zabc " * 10, encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        run_training(train, tmp_path / "out.pt", model_values=VALUES, validation_path=val,
                     config=TrainConfig(steps=1))


def test_paired_training_and_overlap_detection(tmp_path):
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    train.write_text("<HUMAN>: abc <ASSISTANT>: cab\n<HUMAN>: bca <ASSISTANT>: abc\n", encoding="utf-8")
    val.write_text("<HUMAN>: acb <ASSISTANT>: bac\n", encoding="utf-8")
    path = tmp_path / "out.pt"
    run_training(train, path, model_values=VALUES, validation_path=val,
                 config=TrainConfig(steps=1, batch_size=2, val_batches=1))
    assert read_checkpoint(path)["metadata"]["identity"]["mode"] == "pairs"
    metrics = evaluate_model_on_dataset(path, val, continuation=2, num_eval_samples=1)
    assert metrics["evaluated"] == 1
    val.write_text(train.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        run_training(train, path, model_values=VALUES, validation_path=val)


def test_token_weighted_evaluation_and_mode_restore(model):
    tokenizer = CharTokenizer("abcdefg")
    batches = list(text_eval_batches("abcdefgabcdef", tokenizer, 8))
    report = evaluate_loss(model, batches)
    assert report["tokens"] == 12
    assert model.training
    model.eval()
    losses = [torch.nn.functional.cross_entropy(model(b["input_ids"])["logits"].flatten(0, 1),
                                                b["target_ids"].flatten(), reduction="sum").item()
              for b in batches]
    assert report["loss"] == pytest.approx(sum(losses) / 12)
    with pytest.raises(ValueError, match="no supervised"):
        evaluate_loss(model, [])


def test_character_metrics():
    assert bleu_char("abcd", "abcd") == pytest.approx(1)
    assert rouge_l_char("abc", "abc") == 1
    assert rouge_l_char("abc", "") == 0


def test_cli_workflow(tmp_path, capsys):
    train = tmp_path / "train.txt"
    train.write_text("abcde " * 50, encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    main(["train", "--dataset", str(train), "--output", str(checkpoint), "--steps", "2",
          "--context-length", "8", "--d-model", "16", "--num-layers", "1", "--num-heads", "4",
          "--d-ff", "32", "--batch-size", "2", "--val-batches", "1"])
    assert json.loads(capsys.readouterr().out)["step"] == 2
    main(["generate", str(checkpoint), "--prompt", "abc", "--max-new-tokens", "3"])
    assert capsys.readouterr().out.startswith("abc")
    main(["evaluate", str(checkpoint), str(train)])
    assert json.loads(capsys.readouterr().out)["tokens"] == len(train.read_text()) - 1
    main(["benchmark", str(checkpoint), "--prompt", "ab", "--max-new-tokens", "2",
          "--repeats", "1", "--warmup", "0"])
    assert json.loads(capsys.readouterr().out)["outputs_equal"]
    with pytest.raises(SystemExit) as exc:
        main(["generate", str(checkpoint), "--prompt", "unknown"])
    assert exc.value.code == 2


def test_legacy_import_paths(model):
    from decoder_llm.compat.llm import DecoderOnlyTransformer
    from decoder_llm.compat.llm_train import TranslationLineDataset, train_epoch
    from decoder_llm.compat.llm_inference_eval import bleu_char as legacy_bleu
    assert isinstance(model, DecoderOnlyTransformer)
    assert legacy_bleu("abcd", "abcd") == 1
    with pytest.raises(ValueError, match="no batches"):
        train_epoch(model, [], torch.optim.AdamW(model.parameters()), "cpu", 1)


def test_original_pretrain_helper_cycles_loader(model):
    from decoder_llm.compat.llm import pretrain
    batch = {"input_ids": torch.tensor([[1, 2]]), "target_ids": torch.tensor([[2, 3]])}
    before = model.blocks[0].attn.qkv.weight.detach().clone()
    metrics = pretrain(model, [batch], num_steps=2, log_interval=2)
    assert len(metrics["losses"]) == 2
    assert not torch.equal(before, model.blocks[0].attn.qkv.weight)
    with pytest.raises(ValueError, match="no batches"):
        pretrain(model, [], num_steps=1)
