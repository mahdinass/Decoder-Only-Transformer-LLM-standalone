# CUDA validation

On 2026-10-17, a disposable RunPod RTX A5000 completed the full regression suite:
**70 passed, zero failures, zero errors, zero skips** in 5.25 seconds. Nine cases
explicitly use CUDA; the remaining cases retain their CPU coverage. This is
hardware validation for the recorded configuration, not a certification of every
possible device, precision or workload.

The environment was Ubuntu 24.04 / Linux 6.8, Python 3.12.3, PyTorch
2.10.0+cu128, CUDA runtime 12.8, NVIDIA driver 570.195.03, compute capability 8.6.
The official RunPod PyTorch 2.8 image was upgraded to the supported PyTorch 2.10
CUDA wheel before running checks.

## Coverage

CUDA cases check finite loss/gradients and optimizer updates; causal prefix
invariance; chunked-cache logits; cached/uncached greedy output within and beyond
the context window; seeded device-local sampling; GPU checkpoint reloads onto CPU
and GPU; exact same-device training resume including dropout and RNG state;
synchronized benchmark memory telemetry; and train/generate/evaluate/benchmark CLI
workflows on CUDA. Checkpoint/logit comparisons allow small CPU/GPU numerical
differences. The resume test requires exact equality on the same runtime/device.

## Cache measurements

The randomly initialized float32 model has 416,640 parameters: vocabulary 32,
context 128, width 128, two layers, four heads, MLP 512, dropout zero. Batch size
is one; the 16-token prompt is IDs 0 through 15. Each workload uses two warmups
and five alternating measured runs per path, including prefill. CUDA is
synchronized around timing; cached and uncached greedy tokens matched.

| Workload | New tokens | Uncached median seconds | Cached median seconds | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Within context | 64 | 0.176154 | 0.159113 | 1.107x |
| Context rollover | 144 | 0.284689 | 0.257424 | 1.106x |

Peak CUDA allocated tensor bytes (uncached/cached) were 37,382,656 / 36,732,928
within context, and 37,994,496 / 37,994,496 with rollover. These include resident
model and allocator-tracked tensors; they are not total GPU memory consumption.
The process runs the tests before benchmarks, so allocator/runtime history may
affect measurements. Five samples from one short session supply no confidence
interval or general speedup claim. This workload differs from the documented CPU
experiments and cannot establish comparative device speed or language quality.

## Evidence and reproduction

[Summary and source hashes](experiments/cuda-2026-10-17/summary.json),
[pytest log](experiments/cuda-2026-10-17/pytest.log),
[JUnit XML](experiments/cuda-2026-10-17/pytest.xml),
[GPU diagnostics](experiments/cuda-2026-10-17/nvidia-smi.txt),
[within-context samples](experiments/cuda-2026-10-17/benchmark_within_context.json),
[rollover samples](experiments/cuda-2026-10-17/benchmark_rollover.json) and
[output checksums](experiments/cuda-2026-10-17/SHA256SUMS.json) are retained.
The source was exported without Git metadata or private data. Package, test and
runner hashes in the summary identify the tested bytes; the working base revision
was `dc78d6b143d2f9504376c5f5c9b8604a781092fc` plus the CUDA test/runner additions.

From the source checkout with a supported CUDA PyTorch installation:

```bash
python -m pip install -e ".[dev]"
python scripts/validate_cuda.py --output runs/cuda
```

The runner rejects unavailable CUDA and unsupported PyTorch, fails on skipped or
failed CUDA cases, and preserves failure diagnostics. It does not create or clean
up cloud resources. The validation run's archive was downloaded, checked against
the remote SHA-256 and its internal manifest, and all source hashes matched the
local checkout before the pod was stopped and terminated. No persistent/network
volume was created. Cloud setup/teardown time is not part of benchmark timing.

RunPod's audit log records creation at 21:35:47 UTC, stop at 21:43:39 UTC and
deletion at 21:44:00 UTC. At the displayed GPU rate of $0.27/hour plus approximately
$0.004/hour temporary disk, the run cost about $0.04, below its $2 total budget.
This is a duration/rate estimate, not a final billing invoice.

Gitleaks 8.30.1 found no leaks in the scripts or tests. Its documentation scan
flagged the `tokenizer.py` SHA-256 in the retained summary as a generic API key;
that value was verified against the actual source bytes and classified as a hash
false positive. The raw record and checksum manifest were kept unchanged.

Mixed precision, multi-GPU/distributed training, larger-model stress tests and
other CUDA runtimes remain outside this validation. The generated-text learning
experiment remains the CPU experiment documented in [EXPERIMENTS.md](EXPERIMENTS.md).
