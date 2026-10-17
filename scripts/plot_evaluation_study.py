"""Render measured study JSON as standalone SVG and PNG figures (requires .[plots])."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def plot(report_path, output):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    colors = ["#b8c6d1", "#899cab", "#377c96", "#d5792a"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    names = [("grammar", "Controlled grammar"), ("shakespeare", "Tiny Shakespeare")]
    for column, (name, title) in enumerate(names):
        record = report["datasets"][name]
        baseline = record["baselines"]
        losses = [baseline[key]["loss"] for key in ("uniform", "unigram", "bigram")]
        losses.append(record["summary"]["loss"]["mean"])
        errors = [0, 0, 0, record["summary"]["loss"]["stdev"] or 0]
        ax = axes[0, column]
        bars = ax.bar(["Uniform", "Unigram", "Bigram", "Transformer"], losses,
                      yerr=errors, capsize=4, color=colors, width=0.65)
        for bar, value in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(losses) * 0.045,
                    f"{value:.3f}", ha="center", fontsize=10)
        ax.set_ylim(0, max(losses) * 1.23)
        ax.set_ylabel("Held-out cross entropy (nats / character)")
        ax.set_title(f"{title}: final test loss", loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax = axes[1, column]
        for run in record["runs"]:
            curve = run["validation_curve"]
            ax.plot([p["step"] for p in curve], [p["validation"]["loss"] for p in curve],
                    marker="o", markersize=3.5, linewidth=1.7, label=f"Seed {run['seed']}")
        ax.set_title(f"{title}: validation learning curves", loc="left", fontweight="bold")
        ax.set_xlabel("AdamW training steps")
        ax.set_ylabel("Sampled validation cross entropy")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
    fig.suptitle("A small Transformer, evaluated against explicit baselines", fontsize=17, fontweight="bold")
    fig.supxlabel("One CPU runtime · three seeds · identical held-out targets within each dataset · error bars = seed sample SD",
                  fontsize=10)
    for extension in ("svg", "png"):
        fig.savefig(output / f"evaluation-study.{extension}", dpi=180)
        if extension == "svg":
            destination = output / f"evaluation-study.{extension}"
            destination.write_bytes(b"\n".join(line.rstrip() for line in destination.read_bytes().replace(b"\r\n", b"\n").split(b"\n")))
    plt.close(fig)

    sweep = report["cache_sweep"]
    if sweep:
        counts = sorted({r["new_tokens"] for r in sweep})
        workloads = sorted({(r["batch_size"], r["prompt_tokens"]) for r in sweep})
        ratio = np.array([[next(r["speedup"] for r in sweep if
                               (r["batch_size"], r["prompt_tokens"], r["new_tokens"]) == (b, p, n))
                           for n in counts] for b, p in workloads])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        image = axes[0].imshow(ratio, cmap="RdYlBu", norm=TwoSlopeNorm(vmin=min(0.5, ratio.min()), vcenter=1.0, vmax=max(1.5, ratio.max())), aspect="auto")
        axes[0].set_xticks(range(len(counts)), counts)
        axes[0].set_yticks(range(len(workloads)), [f"Batch {b} · prompt {p}" for b, p in workloads])
        axes[0].set_xlabel("New tokens")
        axes[0].set_title("Cache ratio across 12 workloads", loc="left", fontweight="bold")
        for i in range(len(workloads)):
            for j in range(len(counts)):
                axes[0].text(j, i, f"{ratio[i, j]:.2f}×", ha="center", va="center", fontweight="bold")
        fig.colorbar(image, ax=axes[0], label="Uncached median / cached median")
        for batch, prompt in workloads:
            rows = sorted((r for r in sweep if (r["batch_size"], r["prompt_tokens"]) == (batch, prompt)),
                          key=lambda r: r["new_tokens"])
            axes[1].plot(counts, [r["cached"]["generated_tokens_per_second"] for r in rows],
                         marker="o", label=f"Batch {batch}, prompt {prompt}")
        axes[1].set_title("Cached decoding throughput", loc="left", fontweight="bold")
        axes[1].set_xlabel("New tokens")
        axes[1].set_ylabel("Generated tokens / second (whole batch)")
        axes[1].set_ylim(bottom=0)
        axes[1].grid(alpha=0.18)
        axes[1].legend(frameon=False, fontsize=9)
        fig.suptitle("KV-cache measurements include prefill and context rollover", fontsize=16, fontweight="bold")
        fig.supxlabel("64-token context · float32 CPU · 5 alternating repeats per path · 2 warmups · all greedy outputs matched\n"
                      "Ratio > 1 means faster cached decoding; workload timings are not independent hardware trials.", fontsize=9)
        for extension in ("svg", "png"):
            fig.savefig(output / f"cache-sweep.{extension}", dpi=180)
            if extension == "svg":
                destination = output / f"cache-sweep.{extension}"
                destination.write_bytes(b"\n".join(line.rstrip() for line in destination.read_bytes().replace(b"\r\n", b"\n").split(b"\n")))
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("runs/evaluation-study/study.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/images"))
    args = parser.parse_args()
    plot(args.report, args.output)
