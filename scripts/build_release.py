#!/usr/bin/env python3
"""Export an explicit file allowlist into a new directory, without Git history."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = {".git", ".venv", "venv", "__pycache__", "data", "cache",
                "results", "checkpoints", "external", "logs", "node_modules"}
BLOCKED_SUFFIXES = (".jsonl", ".gz", ".npz", ".npy", ".pt", ".pth",
                    ".safetensors", ".bin", ".ckpt", ".log")
SECRETS = re.compile(r"(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|BEGIN [A-Z ]*PRIVATE KEY)")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def approved_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name:
        return False
    # The Python package's data module contains code, not a corpus.
    parts = p.parts[2:] if p.parts[:2] == ("poison_detection", "data") else p.parts
    return not (any(x in BLOCKED_DIRS for x in parts)
                or any(x.startswith(("_restricted_", ".env")) for x in p.parts)
                or name.endswith(BLOCKED_SUFFIXES))


def release_names(root):
    names = [s.strip() for s in (root / "release_files.txt").read_text().splitlines()
             if s.strip() and not s.lstrip().startswith("#")]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate entries in release_files.txt")
    for name in names:
        if not approved_path(name):
            raise ValueError(f"Forbidden release path: {name}")
        p = root / name
        if any(q.is_symlink() for q in [p, *p.parents] if q != root.parent):
            raise ValueError(f"Symlinks are not permitted: {name}")
        if not p.is_file() or root.resolve() not in p.resolve().parents:
            raise ValueError(f"Missing or external release file: {name}")
        if p.suffix != ".pdf":
            content = p.read_text(encoding="utf-8")
            if SECRETS.search(content) or EMAIL.search(content):
                raise ValueError(f"Credential/email pattern needs review in {name}; value suppressed")
            if p.suffix == ".py":
                compile(content, name, "exec")
    return sorted(names)


def verify(directory):
    directory = Path(directory)
    record = json.loads((directory / "RELEASE_MANIFEST.json").read_text())
    expected = set(record["files"]) | {"RELEASE_MANIFEST.json"}
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("Release contains a symlink")
    if actual != expected:
        raise ValueError(f"Release inventory differs: extra={sorted(actual-expected)}, missing={sorted(expected-actual)}")
    if set(release_names(directory)) != set(record["files"]):
        raise ValueError("Allowlist and recorded file inventory differ")
    for name, digest in record["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Release file changed: {name}")
    return len(record["files"])


def build(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Choose a new directory outside the research checkout")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {destination}")
    names = release_names(root)  # Validate everything before creating output.
    destination.mkdir(parents=True)
    hashes = {}
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    record = {"format": 1, "scope": "code, documentation, aggregate statistics",
              "includes_git_history": False, "includes_raw_examples": False,
              "includes_model_weights": False, "files": hashes}
    (destination / "RELEASE_MANIFEST.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    verify(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "Poison-Detection-release")
    parser.add_argument("--verify", type=Path, help="Validate an existing export instead of building")
    parser.add_argument("--zip", action="store_true", help="Also create a new sibling ZIP archive")
    args = parser.parse_args()
    if args.verify:
        print(f"Verified {verify(args.verify)} allowlisted files; hashes and release boundaries match.")
        return
    archive = args.output.resolve().with_suffix(".zip")
    if args.zip and archive.exists():
        parser.error(f"Refusing to overwrite existing archive: {archive}")
    destination = build(ROOT, args.output)
    if args.zip:
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as out:
            for p in sorted(destination.rglob("*")):
                if p.is_file():
                    out.write(p, (Path(destination.name) / p.relative_to(destination)).as_posix())
    print(f"Built {destination} ({verify(destination)} allowlisted files); original data/history retained locally.")


if __name__ == "__main__":
    main()
