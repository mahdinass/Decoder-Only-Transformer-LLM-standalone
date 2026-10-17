# Source and security review

This repository is a fresh source snapshot prepared on 2026-10-18. It contains
one initial commit. The original repository and its development history remain
private and were not rewritten or copied into this repository.

## Snapshot contents

The export includes the reusable package, compatibility interfaces, generated-text
examples, tests, CI, reproducible experiment scripts and recorded measurements.
It excludes the historical corpus/plot, Git objects and refs, local environment
files, credentials, checkpoints, caches and private review records. Contributor
and component credits remain in [PROVENANCE.md](../PROVENANCE.md) and [NOTICE](../NOTICE).

The original source underwent a history/source secrets review before this export;
this is historical evidence, not a claim that old commits exist here. Retained
experiment records may identify private development revisions. They remain
unchanged so their measurement metadata and checksums remain valid.

## Snapshot scan

Gitleaks 8.30.1 scans the exported files before the initial commit. Its generic
API-key rule flags the SHA-256 of decoder_llm/tokenizer.py in the raw CUDA
summary. The value matches the measured source hash and is a reviewed false
positive; the raw summary and manifest remain unchanged. No credential values
were identified. See [scan summary](security-scan-summary.json).

For future changes, scan this repository's refs and working files:

```text
python scripts/audit_history.py --output runs/history-audit.json
gitleaks git . --log-opts="--all --reflog -m" --redact=100
```

A negative scan cannot guarantee absence of unknown or encoded credentials.
Hosted secrets, logs, artifacts and inaccessible refs require separate review.
