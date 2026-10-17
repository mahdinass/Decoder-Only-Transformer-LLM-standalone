import pytest
import torch
from decoder_llm import CharTokenizer
from decoder_llm.data import CharacterDataset, TranslationLineDataset, parse_chat_pairs


def test_tokenizer_roundtrip_and_restored_order():
    tokenizer = CharTokenizer.from_chars(["z", "a", "\n", "λ"])
    assert tokenizer.encode("zaλ\n") == [0, 1, 3, 2]
    assert tokenizer.decode([0, 1, 3, 2]) == "zaλ\n"
    with pytest.raises(ValueError, match="outside"):
        tokenizer.encode("?")
    with pytest.raises(ValueError):
        CharTokenizer.from_chars(["a", "a"])


def test_minimal_text_and_partial_batch():
    tokenizer = CharTokenizer("abc")
    dataset = CharacterDataset("abc", tokenizer, 2, batch_size=2, num_samples=3)
    batches = list(dataset)
    assert [b["input_ids"].shape[0] for b in batches] == [2, 1]
    assert batches[0]["input_ids"].tolist() == [[0, 1], [0, 1]]
    assert batches[0]["target_ids"].tolist() == [[1, 2], [1, 2]]


def test_translation_boundary_long_prompt_and_padding():
    tokenizer = CharTokenizer("abcdefgh\nxy")
    dataset = TranslationLineDataset([("abcdefgh", "xy")], tokenizer, 4, 1, 1)
    batch = next(iter(dataset))
    assert tokenizer.decode(batch["input_ids"][0].tolist()) == "fgh\n"
    supervised = batch["target_ids"][0] != -100
    assert supervised.tolist() == [False, False, False, True]
    assert tokenizer.decode(batch["target_ids"][0, supervised].tolist()) == "x"
    short = next(iter(TranslationLineDataset([("a", "x")], tokenizer, 8, 1, 1)))
    assert (short["target_ids"] != -100).sum() == 1
    tiny = next(iter(TranslationLineDataset([("a", "x")], tokenizer, 1, 1, 1)))
    assert tiny["input_ids"].item() == tokenizer.encode("\n")[0]
    assert tiny["target_ids"].item() == tokenizer.encode("x")[0]


def test_dataset_generator_does_not_consume_global_rng():
    tokenizer = CharTokenizer("abcde")
    dataset = CharacterDataset("abcde" * 4, tokenizer, 4, generator=torch.Generator().manual_seed(1), num_samples=2)
    before = torch.get_rng_state().clone()
    list(dataset)
    assert torch.equal(before, torch.get_rng_state())


def test_parse_pairs():
    assert parse_chat_pairs("text\n<HUMAN>: hello <ASSISTANT>: hi\ninvalid") == [("hello", "hi")]
