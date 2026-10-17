"""Read-only local secret review. Reports locations, never matched values.

Scans every available Git object (including unreachable objects), local config,
and project files. Heuristics are not a certification that no secrets exist.
No network requests, history mutations, or credential validation are performed.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess


PATTERNS = {
    "private-key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "github-token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "aws-key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "service-token": rb"\b(?:sk-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{15,}|AIza[A-Za-z0-9_-]{30,})\b",
    "credential-url": rb"https?://[^\s/<>]{1,100}:[^\s/@<>]{1,150}@",
    "sensitive-assignment": rb"(?i)(?:password|passwd|api[_-]?key|client[_-]?secret|access[_-]?token)\s*[=:]\s*[\x22\x27][^\x22\x27\r\n]{6,}",
    "jwt": rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".cache", "build", "dist"}
SENSITIVE_NAMES = re.compile(r"(?i)(?:^\.env(?:\.|$)|^id_(?:rsa|ed25519)$|\.(?:pem|key|p12|pfx)$|^credentials$)")


def git(*args):
    return subprocess.check_output(["git", *args])


def matches(data, location, findings):
    for label, pattern in PATTERNS.items():
        lines = [data[:m.start()].count(b"\n") + 1 for m in re.finditer(pattern, data)]
        if lines:
            findings.append({"location": location, "rule": label, "lines": lines})


def audit():
    objects = git("cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype) %(objectsize)").decode().splitlines()
    paths = {}
    for row in git("rev-list", "--objects", "--all", "--reflog").decode().splitlines():
        parts = row.split(" ", 1)
        if len(parts) == 2:
            paths[parts[0]] = parts[1]
    findings, binaries = [], []
    kinds = Counter()
    for row in objects:
        oid, kind, size = row.split()
        kinds[kind] += 1
        if kind not in ("blob", "commit", "tag"):
            continue
        location = {"object": oid, "kind": kind, "path": paths.get(oid, "metadata-or-unreachable")}
        data = git("cat-file", kind, oid)
        matches(data, location, findings)
        if kind == "blob" and b"\0" in data:
            binaries.append({**location, "bytes": int(size)})
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    checked = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")
                   and not (Path(directory) / d).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                checked += 1
                matches(path.read_bytes(), {"working_tree": path.relative_to(root).as_posix()}, findings)
    # Includes remote URLs and local custom config without printing their values.
    matches(git("config", "--local", "--list"), {"local_config": True}, findings)
    emails = re.findall(rb"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", git("log", "--all", "--format=%an <%ae> %cn <%ce>"))
    private_emails = {e for e in emails if not e.endswith(b"@users.noreply.github.com")
                      and e != b"noreply@github.com"}
    sensitive_paths = sorted({path for path in paths.values() if SENSITIVE_NAMES.search(Path(path).name)})
    return {"head": git("rev-parse", "HEAD").decode().strip(),
            "shallow": git("rev-parse", "--is-shallow-repository").decode().strip(),
            "object_counts": dict(kinds), "working_tree_files_scanned": checked,
            "sensitive_historical_paths": sensitive_paths, "pattern_findings": findings,
            "binary_objects_requiring_visual_review": binaries,
            "unique_personal_emails_in_commit_identity": len(private_emails),
            "limitations": ["Heuristic patterns can miss unknown or encoded secrets.",
                            "Excluded runtime/cache/build directories are outside the project-source scan.",
                            "Only locally available objects/refs; no remote secrets, logs, artifacts, issues or hidden tests audited.",
                            "Binary content scanned as bytes; images require separate visual review."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Save redacted report as JSON")
    args = parser.parse_args()
    report = json.dumps(audit(), indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    print(report, end="")
