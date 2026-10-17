import hashlib
from scripts.run_reference_experiment import corpus, run


def test_reference_corpus_has_disjoint_complete_sentences():
    splits = corpus()
    assert splits == corpus()
    train, val, test = (set(splits[name]) for name in ("train", "validation", "test"))
    assert not (train & val or train & test or val & test)
    assert all(sentence.endswith(".\n") for sentence in train | val | test)


def test_reference_report_matches_actual_data_bytes_and_vocabulary(tmp_path):
    report = run(tmp_path, steps=1, seeds=[17])
    for split in ("train", "validation", "test"):
        raw = (tmp_path / f"{split}.txt").read_bytes()
        assert b"\r" not in raw
        assert hashlib.sha256(raw).hexdigest() == report["splits"][split]["sha256"]
    assert report["model_config"]["vocab_size"] == len(set((tmp_path / "train.txt").read_bytes().decode()))
    measured = report["runs"][0]["cache_benchmarks"]["within_context"]
    assert measured["model_config"] == report["model_config"]
    assert report["runs"][0]["untrained_test"]["tokens"] == report["runs"][0]["trained_test"]["tokens"]
