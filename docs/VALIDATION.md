# Validation

Validation date: 2026-10-17. The package was tested locally on Windows 11 CPU.

| Environment | Regression result |
| --- | --- |
| Python 3.12.5 / PyTorch 2.14.1+cpu | 61 passed |
| Python 3.11.9 / PyTorch 2.10.0+cpu | 61 passed |
| Linux / Python 3.12.3 / PyTorch 2.10.0+cu128 / RTX A5000 | 70 passed, zero skips |

The GPU run includes nine explicit CUDA regression cases. See
[CUDA validation](CUDA_VALIDATION.md) for coverage, reproduction, raw logs,
source hashes and measured cache timings. With those tests added, the local CPU
suite reports 61 passed and nine expected CUDA skips.

The reference-experiment tests were rerun after the final fingerprint-format
change: 2 passed. The CPU wheels emitted an optional NumPy-initialization warning;
none of the tested operations required NumPy.

## Behavior covered

Tests cover configuration and input validation, causal prefix invariance, cache
chunk offsets and overflow, tied-weight gradients, RMSNorm accumulation, masked
noncontiguous loss, vocabulary restoration and unknown characters, window/batch
boundaries, prompt/target masking, cached/full-prefix equivalence across rollover,
seeded decoding, repetition penalties, EOS handling, checkpoint round trips,
explicit legacy conversion, exact CPU resume, data-change rejection, held-out
loss weighting, compatibility imports and end-to-end CLI workflows.

The generated-data experiment also checks disjoint sentence splits, canonical LF
bytes, data/report hashes, matching baseline/trained vocabularies and matching
numbers of evaluation targets. See [EXPERIMENTS.md](EXPERIMENTS.md) for the measured
three-seed findings, raw records, samples and limitations.

## Baseline compatibility

A read-only comparison used the model source at `367c873`. With seed 19, vocabulary
7, context 8, width 16, two layers, four heads, MLP 32 and dropout 0, state-dict
keys and initialized parameters matched exactly. Forward logits on token IDs
`[1,2,3,4,5]` had maximum absolute difference 0.0. Three-token beam output with
width 5 matched. This is a checked configuration, not an exhaustive guarantee
across every dtype or device.

## Packaging and security

Both wheel and source distributions were built and inspected. They contain
required package modules, compatibility imports and contributor notices;
the source archive also contains tests, examples, lock and reproduction scripts.
Historical corpus/plot files, checkpoints, virtual environments, caches and local
review records are excluded. `scripts/check_distribution.py` enforces these rules.

An installed-wheel smoke check passed model/checkpoint loading, generation and CLI
option parsing. The dependency lock check and Git whitespace check passed.
Gitleaks history/source findings and the custom object scan are documented in
[AUDIT.md](AUDIT.md).

## Validation limits

GPU validation covers one RTX A5000 with CUDA 12.8 and float32; other GPU models,
mixed-precision training and distributed training were not validated. Local
testing did not include Python 3.10. The private development source subsequently
passed the configured Windows/Linux CPU CI matrix, including minimum/current
dependency coverage. This repository includes the same CI matrix to validate
its exported tree independently.
No package upload or public release was performed. Results on generated text do
not establish natural-language or translation quality.
