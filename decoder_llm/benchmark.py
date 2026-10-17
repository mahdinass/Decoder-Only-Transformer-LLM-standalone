"""Measure full generation, including prefill, with cache equivalence checks."""
import platform
from dataclasses import asdict
import statistics
import time
import torch
from .generation import generate


def benchmark_cache(model, prompt_ids, *, max_new_tokens=32, repeats=5, warmup=1):
    if max_new_tokens < 1 or repeats < 1 or warmup < 0:
        raise ValueError("tokens/repeats must be positive and warmup nonnegative")
    device = next(model.parameters()).device

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run(cached):
        return generate(model, prompt_ids, max_new_tokens=max_new_tokens,
                        strategy="greedy", use_cache=cached, append_separator=False)

    if not torch.equal(run(False), run(True)):
        raise RuntimeError("cached and uncached greedy outputs differ; timing is invalid")
    for _ in range(warmup):
        run(False)
        run(True)
    samples = {False: [], True: []}
    peaks = {False: [], True: []}
    for repeat in range(repeats):
        # Alternate order to reduce systematic warm-cache/thermal bias.
        for cached in ((False, True) if repeat % 2 == 0 else (True, False)):
            sync()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            run(cached)
            sync()
            samples[cached].append(time.perf_counter() - start)
            if device.type == "cuda":
                peaks[cached].append(torch.cuda.max_memory_allocated(device))
    modes = {}
    for cached, label in ((False, "uncached"), (True, "cached")):
        median = statistics.median(samples[cached])
        modes[label] = {"seconds": samples[cached], "median_seconds": median,
                        "generated_tokens_per_second": prompt_ids.shape[0] * max_new_tokens / median,
                        "peak_cuda_allocated_bytes": max(peaks[cached]) if peaks[cached] else None}
    return {"torch_version": str(torch.__version__), "python_version": platform.python_version(),
            "platform": platform.platform(), "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "cpu_threads": torch.get_num_threads(), "dtype": str(next(model.parameters()).dtype),
            "batch_size": prompt_ids.shape[0], "prompt_tokens": prompt_ids.shape[1],
            "new_tokens": max_new_tokens, "context_length": model.config.context_length,
            "model_config": asdict(model.config),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "repeats": repeats, "warmup": warmup, "outputs_equal": True,
            "includes_prefill": True,
            "context_rollover": prompt_ids.shape[1] + max_new_tokens - 1 > model.config.context_length,
            "speedup": modes["uncached"]["median_seconds"] / modes["cached"]["median_seconds"],
            **modes}
