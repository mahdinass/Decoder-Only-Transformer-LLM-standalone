"""Check built archives for required source/notices and excluded private material."""
import argparse
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


def check(directory):
    directory = Path(directory)
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    if not wheels or not sources:
        raise ValueError("both a wheel and source distribution are required")
    for path in wheels + sources:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
            required = ("decoder_llm/model.py", "decoder_llm/cli.py", "student/llm.py")
            if not all(name in names for name in required):
                raise ValueError(f"{path.name}: missing package components")
            for notice in ("LICENSE", "NOTICE", "PROVENANCE.md"):
                if not any(name.endswith("/licenses/" + notice) for name in names):
                    raise ValueError(f"{path.name}: missing {notice}")
        else:
            with tarfile.open(path) as archive:
                names = archive.getnames()
            for required in ("tests/conftest.py", "uv.lock", "LICENSE", "NOTICE", "CHANGELOG.md",
                             "PROVENANCE.md", "scripts/run_reference_experiment.py"):
                if not any(name.endswith("/" + required) for name in names):
                    raise ValueError(f"{path.name}: missing {required}")
        for name in names:
            parts = PurePosixPath(name).parts
            if (any(p in {".git", ".cache", ".venv", ".release-review", "runs", "data"} for p in parts)
                    or name.endswith((".pt", ".pth", ".ckpt", "losses.png", "classroom.yml"))
                    or "klingon_translation_train" in name):
                raise ValueError(f"{path.name}: excluded material present at {name}")
        print(f"Verified {path.name}: package components and notices present; excluded material absent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="dist")
    args = parser.parse_args()
    check(args.directory)
