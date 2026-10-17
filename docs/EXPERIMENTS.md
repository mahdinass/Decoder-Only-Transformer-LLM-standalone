# Experiments and measurements

The expanded [comparative evaluation](EVALUATION.md) adds natural text, statistical
baselines, learning curves, generation diagnostics and a twelve-workload cache
sweep. This page retains the earlier reference and smoke experiment records.

## Controlled grammar experiment

The reference experiment uses only project-generated text and requires no external
corpus. Reproduce it with:

```powershell
.venv/Scripts/python.exe scripts/run_reference_experiment.py --output runs/reference-final --steps 300 --seeds 17 29 43
```

The script constructs 384 unique sentences from four subjects, four colors, four
verbs and six objects, then creates a fixed seeded sentence split: 307 training,
38 validation and 39 test sentences. Files use canonical UTF-8/LF bytes on every
platform. The training-only character vocabulary has 25 entries. Each model has
context 64, width 64, two layers, four heads, MLP 128, dropout 0.1 and tied embeddings.
Each seed receives 300 AdamW steps, batch size 16, learning rate 0.001, weight decay
0.05 and gradient clipping 1.0. Test data is scored only at the predetermined final
step and is not used for checkpoint selection. Each evaluation scores 1,213 test
next-character targets once, resetting context between 64-token chunks.

| Seed | Untrained test cross entropy | Trained test cross entropy | Trained test perplexity |
| --- | ---: | ---: | ---: |
| 17 | 3.250446 | 0.330586 | 1.391784 |
| 29 | 3.236580 | 0.323809 | 1.382384 |
| 43 | 3.268151 | 0.305795 | 1.357705 |

Mean final test cross entropy was **0.320064**, with
sample standard deviation **0.012813** over three
seeds, compared with initial mean **3.251726**.
The decrease establishes learning under this controlled grammar and evaluation
policy. It does not establish translation ability or natural-language quality;
training/test sentences use the same narrow grammar and vocabulary. All runs use
one Windows CPU environment. Source hashes, data hashes, settings, full per-run
metrics and generations are in [the raw report](experiments/reference.json).

A seed-17 greedy sample begins:

```text
the red robot moves the boox.
the gren baker moves the box.
the green baker mo
```

Cache measurements on seed 17 include prefill, two warmups, five measured runs per
path, alternating timing order and an output-equivalence check. The prompt has
14 characters; the within-context workload generates 24 tokens and the rollover
workload generates 80. No EOS stopping or repetition penalty is applied.

| Workload | Full-prefix median seconds | Cached median seconds | Observed ratio |
| --- | ---: | ---: | ---: |
| within context | 0.060827 | 0.050426 | 1.206x |
| with rollover | 0.218017 | 0.177238 | 1.230x |

Both output comparisons passed. These are local measurements, not universal
speedup claims; timing variance and shared CPU load affect such short runs.
Recomputation after learned-position rollover is necessary for correctness.
The experiment is small, uses only three seeds and reports no timing confidence
intervals. A separate [GPU validation and cache measurement](CUDA_VALIDATION.md)
uses a different model/workload and is not a direct CPU/GPU comparison.

## Initial smoke experiment


The commands in README were executed locally on 2026-10-17 using Python 3.12.5,
PyTorch 2.14.1+cpu, Windows 11, and one intra-op CPU thread. No GPU or real-world
language-quality benchmark was run. The small synthetic corpus and short training
budget make this a functional smoke experiment only. The generated continuation
was largely spaces; the run is not evidence of useful language quality.

The model configuration is `examples/tiny.json`: context 32, hidden width 32,
two layers, four heads, MLP width 64, dropout 0.1. Training used seed 1337,
20 AdamW steps, batch size 4, learning rate 0.0003, weight decay 0.05 and gradient
clipping at 1.0. `examples/validation.txt` was separate from training. Validation
sampling used two batches and an independent seed, with a training-only vocabulary.

| Measurement | Observed value | Scope |
| --- | --- | --- |
| Step 20 train batch cross entropy | 3.1747772694 | A single stochastic training batch |
| Step 20 sampled validation cross entropy | 3.1640254259 | 256 next-character targets from fixed sampled windows |
| Full held-out text cross entropy | 3.1532597452 | 106 targets, each scored once with chunk resets |
| Full held-out text perplexity | 23.4122583237 | Character perplexity under that chunk policy |

Sampled validation and full held-out text are different target/context policies;
their numbers should not be treated as repeated measurements of the same metric.

## Cache smoke benchmark

The trained tiny model used batch size 1, float32, prompt `the robot ` (10 tokens),
and 16 new greedy tokens. Both paths had two warmups and five measured runs in
alternating order. Timings include prefill and generation, and output tensors
were checked equal before measuring. There was no context rollover. CPU identified
itself as Intel64 Family 6 Model 186 Stepping 2, GenuineIntel. See the complete
[raw timing report](experiments/cache-smoke.json).

| Path | Median seconds | Generated tokens / second |
| --- | ---: | ---: |
| Full prefix | 0.0350756000 | 456.1576 |
| KV cache | 0.0270909000 | 590.6042 |

The observed ratio was 1.2947x for this one short local run. Millisecond-scale CPU
timings are noisy; no statistical confidence interval, GPU comparison, or broad
speedup claim is supplied. CPU peak tensor memory is not measured. CUDA runs
synchronize before/after timing and record peak allocated tensor bytes, which
include the resident model and allocator-tracked tensors, not total device usage.

The cache grows by concatenation and resets when learned positions reach the
context limit. Longer runs beyond the window can have little or no speedup. Use
larger model/prompt sweeps and repeated independent runs on your target device
before making performance decisions. Keep original JSON measurements and record
both prefill-inclusive and context-rollover workloads; never extrapolate this
tiny result to large language models.

## Reproduction and extensions

Run the exact README smoke commands in the locked environment. Reports under
`runs/smoke/` retain settings, raw timings, corpus/split hashes and runtime metadata.
The training record marks the Git checkout dirty because measurements were made
before the release commits. The recorded source hashes identify the measured code;
subsequent commits do not change those historical run records.

For meaningful experiments, curate and attribute your own train/validation/test
data, split by document or source before sampling, keep a fixed vocabulary,
and report the unknown-character policy. Choose seeds and model sizes in advance.
Report token-weighted held-out loss with its context policy, variation across
seeds, runtime/hardware, and sample generations. Character BLEU/ROUGE-L from the
historical metric utility can be reported separately for paired text, with skipped counts
and fixed decoding settings. They are not standard corpus BLEU.
