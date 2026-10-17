# Changelog

## 0.2.0

### Added

- Reusable `decoder_llm` Python package and configurable train, generate, evaluate,
  benchmark and legacy-checkpoint conversion commands.
- Seeded experiments with data/source fingerprints, fixed validation sampling,
  token-weighted loss and resumable optimizer/random-number state.
- KV-cached greedy and sampling generation, top-p filtering, repetition penalties,
  EOS handling and cache/full-prefix equivalence checks.
- Three-seed controlled grammar experiment with held-out findings and raw records.
- Explicit CUDA regressions and a validation runner with source/output hashes,
  JUnit results, GPU diagnostics and synchronized cache measurements.
- Regression tests, Windows/Linux CI configuration, distribution checks, security
  guidance, contributor documentation and contributor credits.

### Fixed

- Cache shape/context validation and learned-position behavior at window rollover.
- Translation rows with no supervised targets, minimal-window and partial-batch cases.
- Vocabulary ordering during checkpoint restoration and silent unknown-character skips.
- Half-precision RMSNorm square accumulation and noncontiguous masked-loss inputs.
- Cross-platform line endings in the reference-experiment corpus.

### Compatibility

- Existing model parameter names and legacy imports remain available.
- New checkpoints use a versioned tensor/primitive format. Loading older object
  checkpoints requires explicit trusted conversion.
- CLI defaults are small and CPU-oriented. No pretrained model is distributed.
