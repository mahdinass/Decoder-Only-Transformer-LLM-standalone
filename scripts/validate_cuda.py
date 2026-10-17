"""Run from the repository root: python scripts/validate_cuda.py --output runs/cuda.

Requires a real CUDA device. Saves diagnostics even when validation fails.
This script does not provision, stop, or delete cloud resources.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/cuda"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"started_utc": datetime.now(timezone.utc).isoformat(), "passed": False}
    status = 1
    try:
        import torch
        from decoder_llm import LLMConfig, build_model
        from decoder_llm.benchmark import benchmark_cache
        from decoder_llm.experiment import set_seed

        sources = sorted({p for folder in ("decoder_llm", "tests", "scripts")
                          for p in (ROOT / folder).rglob("*.py")})
        summary["source_sha256"] = {p.relative_to(ROOT).as_posix():
                                     hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        summary["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                                  "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                                  "cuda_available": torch.cuda.is_available()}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; CPU success cannot substitute for this validation")
        diagnostic = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
        (output / "nvidia-smi.txt").write_text(diagnostic.stdout + diagnostic.stderr, encoding="utf-8")
        summary["environment"]["gpu"] = torch.cuda.get_device_name(0)
        summary["environment"]["compute_capability"] = list(torch.cuda.get_device_capability(0))
        if tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) < (2, 10):
            raise RuntimeError("PyTorch >=2.10 is required")
        with (output / "pytest.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA",
                                     f"--junitxml={output / 'pytest.xml'}"], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=600)
        summary["pytest_exit_code"] = result.returncode
        suites = list(ET.parse(output / "pytest.xml").getroot().iter("testsuite"))
        summary["tests"] = {key: sum(int(s.get(key, 0)) for s in suites)
                            for key in ("tests", "failures", "errors", "skipped")}
        cuda_cases = [case for case in ET.parse(output / "pytest.xml").iter("testcase")
                      if case.get("classname", "").endswith("test_cuda")]
        if len(cuda_cases) < 9 or any(list(case) for case in cuda_cases):
            raise RuntimeError("CUDA tests did not all execute successfully; inspect pytest.xml")
        if result.returncode:
            raise RuntimeError("Regression suite failed; inspect pytest.log")
        set_seed(17)
        torch.set_num_threads(1)
        model = build_model(LLMConfig(vocab_size=32, context_length=128, d_model=128,
                                     num_layers=2, num_heads=4, d_ff=512, dropout=0.0)).to("cuda").eval()
        prompt = torch.arange(16, device="cuda").reshape(1, -1)
        for name, count in (("within_context", 64), ("rollover", 144)):
            report = benchmark_cache(model, prompt, max_new_tokens=count, repeats=5, warmup=2)
            (output / f"benchmark_{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        summary["passed"] = True
        status = 0
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(output.iterdir()) if p.is_file() and p.name != "SHA256SUMS.json"}
        (output / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
