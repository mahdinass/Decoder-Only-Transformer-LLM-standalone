# Release validation summary

The implementation and distribution artifacts have completed the documented local
engineering checks. This document records evidence; it does not publish a release
or certify behavior on untested hardware.

| Area | Evidence |
| --- | --- |
| Model behavior | Baseline parameter/logit/beam comparison and regression tests |
| Reusable package | Configurable model, training, generation, evaluation and CLI |
| Reproducibility | Seeded sampling, data/source hashes, optimizer/RNG resume and split policies |
| Findings | Three-seed generated-grammar experiment and raw cache timings |
| Tests | 61 passed on Python 3.12/PyTorch 2.14 and Python 3.11/PyTorch 2.10, CPU |
| CUDA | 70 passed with zero skips on RTX A5000 / PyTorch 2.10.0+cu128; raw evidence retained |
| Packaging | Wheel/source builds, archive inspection and installed-wheel smoke check |
| Security | Dedicated history/source scans plus all-object review; no detected credential values |
| Documentation | Setup, architecture, experiments, changelog, contributor and security guides |
| Credits | Maintainer and source-component contributions recorded separately |
| Source history | One initial snapshot commit; original repository retained privately; contributor credits preserved |
| License | MIT grant for Mahdi's contributions; retained component scope recorded in NOTICE |

See [VALIDATION.md](VALIDATION.md), [EXPERIMENTS.md](EXPERIMENTS.md),
[AUDIT.md](AUDIT.md) and [NOTICE](../NOTICE).

The historical corpus and plot are absent from this repository and package
archives. [GPU validation](CUDA_VALIDATION.md) covers one device
and runtime, including nine explicit CUDA cases. The package/experiment commit passed the
configured Windows/Linux CPU CI matrix.
This repository starts from the validated package snapshot on main. Its own CI
checks the exported tree; earlier CPU/GPU experiment records retain their original
source hashes and environment metadata. Repository visibility is private during
preparation.
