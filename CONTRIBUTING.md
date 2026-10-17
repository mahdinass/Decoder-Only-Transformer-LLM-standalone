# Contributing

## Development setup

Use Python 3.10+ and install the locked development environment:

```text
uv sync --locked --extra dev
uv run --locked python -m pytest
uv build
uv run --locked python scripts/check_distribution.py dist
```

Alternatively, create a virtual environment and run `pip install -e ".[dev]"`.
On restricted Windows environments, use a workspace-local ignored `--basetemp`
directory for pytest temporary files.

## Changes and evidence

Keep changes focused on a documented behavior or defect. Include regression tests
for model, data, checkpoint or decoding behavior and describe changed defaults.
Preserve compatibility imports unless a documented migration replaces them.

For performance work, include raw measurements, hardware/software settings,
repeat counts and output-equivalence checks. Use
`scripts/run_reference_experiment.py` for a repeatable generated-data experiment;
do not present its results as general natural-language quality.

Use your own Git identity. Do not include tokens, private data, model checkpoints,
or external material without its source and terms. Data needs documented
provenance. Preserve contributor credits and this repository's history; see [NOTICE](NOTICE).
Use the private reporting guidance in [SECURITY.md](SECURITY.md) for vulnerabilities.
