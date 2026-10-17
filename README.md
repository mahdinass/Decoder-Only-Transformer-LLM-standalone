# Decoder-Only Transformer LLM

A readable PyTorch character-level Transformer for small language-model experiments,
with configurable training, generation, held-out evaluation, and measured KV-cache
comparisons. Reusable model, data and decoding components are separated from
experiment orchestration so you can train a custom model or use the Python API.

Features include RMSNorm decoder blocks, tied embeddings, masked prompt/response
training, reproducible checkpoints, greedy/sampling/beam decoding, and cache
benchmarks that check output equivalence before timing.

## Setup

Clone the standalone repository first:

```text
git clone https://github.com/mahdinass/Decoder-Only-Transformer-LLM-standalone.git
cd Decoder-Only-Transformer-LLM-standalone
```

Python 3.10+ and PyTorch 2.10+ are required. Create a virtual environment and install
locally (PowerShell shown):

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pytest
```

On Linux/macOS use `.venv/bin/python`. After activating the environment, `decoder-llm`
is equivalent to `python -m decoder_llm`. For a specific CPU/CUDA wheel, use
[PyTorch's installation selector](https://pytorch.org/get-started/locally/) before
installing this package. Optional loss plotting requires `.[plots]`.

`uv.lock` records resolved dependencies. With uv installed, `uv sync --locked --extra dev`
and `uv run --locked python -m pytest` provide the locked development setup.
The lock includes platform-specific variants; wheel availability depends on the
chosen platform. Local validation used Windows CPU; GitHub Actions also passed
the configured Linux/Windows CPU matrix on the package and experiment commits.

## A small, reproducible experiment

The synthetic robot corpus below was written for this project and requires no
downloaded corpus. The 20-step run is a smoke experiment, not a quality
benchmark:

```powershell
.venv/Scripts/python.exe -m decoder_llm train --dataset examples/train.txt --validation examples/validation.txt --model-config examples/tiny.json --steps 20 --batch-size 4 --val-batches 2 --eval-every 10 --save-every 10 --output runs/smoke/checkpoint.pt
.venv/Scripts/python.exe -m decoder_llm generate runs/smoke/checkpoint.pt --prompt "the robot " --max-new-tokens 16 --strategy greedy
.venv/Scripts/python.exe -m decoder_llm evaluate runs/smoke/checkpoint.pt examples/validation.txt --report runs/smoke/evaluation.json
.venv/Scripts/python.exe -m decoder_llm benchmark runs/smoke/checkpoint.pt --prompt "the robot " --max-new-tokens 16 --repeats 5 --warmup 2 --report runs/smoke/cache.json
```

Outputs stay in ignored `runs/`. Training writes a versioned checkpoint and a
neighboring `.pt.json` experiment record containing settings, data hashes, runtime
metadata and metrics. Supply your own UTF-8 training and held-out text for useful
experiments. Characters absent from the training vocabulary are rejected explicitly.

Train model dimensions can be set in a JSON config or with `--context-length`,
`--d-model`, `--num-layers`, `--num-heads`, `--d-ff`, and `--dropout`. Optimizer,
validation, checkpoint cadence, seed, device, and step budget are configurable;
run `python -m decoder_llm train --help` for details. CPU and one thread are explicit
CLI defaults. CUDA is available with `--device cuda` and a suitable PyTorch install.

To stop a run at step 10 and resume to its original 20-step budget, pass
`--stop-after 10` on the first invocation, then rerun the identical training command
with `--resume runs/smoke/checkpoint.pt` and without `--stop-after`. Keep the data,
config, runtime and device type unchanged. The checkpoint restores optimizer and
random-number states. Deterministic algorithms are enabled by default and fail
rather than silently falling back to nondeterministic operations.

## Generation and evaluation

Generation supports `--strategy greedy`, `sample` or `beam`, sampling temperature,
`--top-k`, `--top-p`, `--seed`, optional `--eos-token-id`, and `--no-cache` for
comparisons. `--repetition-penalty 1.1` discourages characters already present in
the prompt or continuation; its default of `1.0` leaves logits unchanged. Because
this is a character model, stronger penalties can also discourage ordinary
letters and spaces, so choose the value for your task. Beam search retains full-prefix decoding and supports one prompt.
The character model has no dedicated learned EOS token. Passing an ordinary
character ID as EOS is a user-selected stopping rule.

Training also recognizes the original `<HUMAN>: prompt <ASSISTANT>: target` line
format. It learns targets after a newline separator and masks prompt/padding loss.
Use `evaluate --mode pairs` for the historical character BLEU/ROUGE-L measures.
Their implementation and limitations are documented; scores do not establish
translation quality. Paired evaluation uses a fixed continuation budget and
reports skipped OOV prompts.

Default text evaluation reports token-weighted cross entropy and perplexity,
scoring every next-character target once in chunks with context resets. Use a
held-out file; the CLI cannot establish whether arbitrary external evaluation
text has previously appeared in training.

Cache benchmarking checks greedy output equality before timing, synchronizes CUDA
when applicable, warms up both paths, alternates timing order, and records every
measurement. It includes prefill and reports total generated tokens per second,
median latency, device/runtime settings and CUDA peak allocated memory when
available. Speedups depend on model size, context rollover, device and workload.
See [experiment methodology and measured smoke results](docs/EXPERIMENTS.md).

## Measured findings

A three-seed experiment on generated text reduced mean held-out character cross
entropy from 3.252 before training to
0.320 after 300 steps (sample standard deviation
0.013). Splits are disjoint by complete sentence;
this demonstrates learning a controlled grammar and makes no translation-quality
claim. [Experiment settings, raw results and cache findings](docs/EXPERIMENTS.md)
include reproduction commands and limitations.

Hardware validation on an RTX A5000 passed all 70 regression cases, including
nine explicit CUDA tests. [GPU validation and cache measurements](docs/CUDA_VALIDATION.md)
include the runtime, raw evidence and reproduction command.

## Python API

```python
import torch
from decoder_llm import CharTokenizer, LLMConfig, build_model, generate

tokenizer = CharTokenizer("a small robot reads.\n")
model = build_model(LLMConfig(
    vocab_size=tokenizer.vocab_size, context_length=32,
    d_model=32, num_layers=2, num_heads=4, d_ff=64, dropout=0.0,
))
prompt = torch.tensor([tokenizer.encode("a robot")], dtype=torch.long)
# Random initialization: this checks the API, not meaningful language generation.
tokens = generate(model, prompt, max_new_tokens=8, strategy="greedy", use_cache=True)
print(tokenizer.decode(tokens[0].tolist()))
```

The architecture uses pre-norm RMSNorm blocks, explicit multi-head causal attention,
learned absolute positions, a GELU MLP, residual connections, dropout and tied
embedding/output weights. [Architecture and behavior](docs/ARCHITECTURE.md) explains
masking, cache layout, context rollover, checkpoint format and split policies.

## Compatibility and project layout

`student.llm` still exports the original model/tokenizer/dataset/loss/generation
names, and `python -m student.llm_train --dataset ... --output ...` routes to the
new configurable trainer. Legacy train/epoch/plot helper functions and the
`student.llm_inference_eval` CLI remain available. State-dict parameter names are
preserved. New CLI defaults are intentionally small and CPU-friendly; the old
fixed six-layer/three-epoch experiment is not silently reproduced.

Historical object-pickle checkpoints need an explicit local conversion:

```text
python -m decoder_llm convert-legacy old.pt converted.pt --trust-pickle
```

Use this only on checkpoints you created and trust. Normal loading never falls
back to general pickle. Conversion preserves weights/config/vocabulary but does
not invent missing training state. See the architecture document for security
and migration details.

* `decoder_llm/`: reusable model, data, decoding, training, evaluation and CLI.
* `student/`: compatibility imports and historical helper interfaces.
* `examples/`: new synthetic text and a small JSON model config.
* `tests/`: regression tests for model behavior, decoding, data, checkpoints and CLI workflows.
* `scripts/audit_history.py`: repeatable read-only local secret-pattern review.
* `docs/`: architecture, experiments, audit and validation evidence.

## Limitations

This is a small experimental character model: no subword tokenizer, distributed
training, mixed-precision training, fused attention, model hub integration or
production serving is implemented. Attention remains quadratic during prefill.
KV tensors grow through concatenation; learned positions require full-window
recomputation after context rollover. Beam search is uncached. Exact replay across
hardware/software stacks is not promised. No pretrained model or external benchmark
quality results are included.

## Credits

Mahdi's original contributions are available under the [MIT License](LICENSE).
See [NOTICE](NOTICE) for the scope of that grant and retained component credits.

Maintained and developed by Mahdi (`mahdinass`). Contributor and component credits
are recorded in [NOTICE](NOTICE) and [PROVENANCE.md](PROVENANCE.md). Original source
history is retained privately. This repository starts with a standalone source
snapshot and one initial commit. The historical corpus and plot are absent;
the examples and measured experiments use generated text.

The [audit](docs/AUDIT.md), [validation record](docs/VALIDATION.md) and
[changelog](CHANGELOG.md) describe the review evidence and release changes.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions,
[SECURITY.md](SECURITY.md) for checkpoint/security handling, and
[release readiness](docs/RELEASE_READINESS.md) for the requirement-by-requirement status.
