# Source and contributor credits

Maintained and developed by Mahdi (`mahdinass`). The project evolves an existing
implementation. This repository begins with a standalone source snapshot and one
initial commit; the original development repository and history remain private.
Contributor identities and component credits are retained here.

| Component | Recorded source |
| --- | --- |
| Model scaffold, configuration, embedding/position structure, initialization, tied weights and optimization helper | Present in the initial development snapshot; retained and evolved in `decoder_llm/model.py` and `training.py` |
| Tokenizer implementation, normalization, attention/cache, feed-forward and forward methods | Mahdi's implementation |
| Random-window batching, masked prompt/response training, beam search and seed handling | Mahdi's later implementation changes, now modularized and extended |
| Historical `train_epoch` helper | Contribution by `vinaysetty`; retained through compatibility interfaces |
| Character BLEU/ROUGE-L utilities | Present in initial snapshot; retained in `decoder_llm/metrics.py` with a revised evaluation pipeline |
| Package CLI, restricted checkpoints, experiment orchestration, benchmarks, tests and documentation | Added during development of the reusable package |
| Historical Klingon corpus | Present in the private development repository; source/translator not documented; absent from this repository |
| Historical `losses.png` | Earlier private development artifact; absent from this repository and not used as a source of numeric experiment results |
| Current example text and controlled grammar corpus | Newly written/generated for the project; no external corpus required |

See [NOTICE](NOTICE) for contributor credits. Dependencies retain their upstream
copyright notices. The private source repository was not rewritten. Revision IDs
inside retained experiment records identify the original measurement environment;
they are not commits in this repository. Those records remain unmodified.

The optional Tiny Shakespeare evaluation fetches the checksum-pinned text from
Andrej Karpathy's `char-rnn` repository into ignored local experiment storage.
That corpus is not bundled in this source tree or distribution archives. The
source URL, revision and SHA-256 are recorded with the evaluation results.
