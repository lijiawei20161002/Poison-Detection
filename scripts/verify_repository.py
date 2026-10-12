#!/usr/bin/env python3
"""Check the committed public file set and every locally reachable Git revision."""
import argparse
import ast
import io
from pathlib import Path
import subprocess

from build_release import EMAIL, SECRETS, release_names


ROOT = Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def validate_blob(content, name):
    if name.endswith(".pdf"):
        return  # The allowlisted statement PDF is reviewed when regenerated.
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Unexpected binary content in {name}") from exc
    if SECRETS.search(text) or EMAIL.search(text):
        raise ValueError(f"Credential/email pattern in history at {name}; value suppressed")
    if name.endswith(".py"):
        tree = ast.parse(text, filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ReleasedUnigramZScore":
                methods = {child.name for child in node.body
                           if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}
                if methods & {"fit", "score"}:
                    raise ValueError(f"Excluded translated baseline remains in history at {name}")


def verify_repository(root):
    root = Path(root).resolve()
    top = Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top != root:
        raise ValueError("The selected directory is not a repository root")
    if git(root, "rev-parse", "--is-shallow-repository").strip() != b"false":
        raise ValueError("A complete local history is required; shallow clones cannot be verified")
    if git(root, "replace", "--list").strip():
        raise ValueError("Git replace refs can hide original history; refusing verification")
    grafts = Path(git(root, "rev-parse", "--path-format=absolute", "--git-path", "info/grafts").decode().strip())
    if grafts.exists() and grafts.stat().st_size:
        raise ValueError("Git grafts can hide original history; refusing verification")
    allowed = set(release_names(root))
    tracked = set(filter(None, git(root, "ls-files", "-z").decode().split("\0")))
    if tracked != allowed:
        raise ValueError(f"Git index differs from allowlist: extra={sorted(tracked - allowed)}, missing={sorted(allowed - tracked)}")
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z")
    if untracked:
        raise ValueError("Unreviewed untracked files remain; add reviewed files to the allowlist or move them outside the repository")
    if git(root, "diff", "--name-only", "HEAD", "--").strip():
        raise ValueError("Uncommitted changes remain; verify the reviewed, committed version before publication")

    revisions = git(root, "rev-list", "--all").decode().splitlines()
    objects = {}
    for revision in revisions:
        entries = git(root, "ls-tree", "-r", "-z", "--full-tree", revision).split(b"\0")
        for entry in filter(None, entries):
            metadata, filename = entry.split(b"\t", 1)
            mode, kind, oid = metadata.decode().split()
            name = filename.decode()
            if name not in allowed:
                raise ValueError(f"Excluded historical path in {revision[:12]}: {name}")
            if kind != "blob" or mode not in {"100644", "100755"}:
                raise ValueError(f"Historical symlink or external repository at {name}")
            objects.setdefault(oid, set()).add(name)

    payload = ("\n".join(objects) + "\n").encode()
    blobs = io.BytesIO(subprocess.check_output(
        ["git", "-C", str(root), "cat-file", "--batch"], input=payload))
    for oid, names in objects.items():
        header = blobs.readline().decode().split()
        if len(header) != 3 or header[:2] != [oid, "blob"]:
            raise ValueError("Unexpected Git object response")
        content = blobs.read(int(header[2]))
        if blobs.read(1) != b"\n":
            raise ValueError("Truncated Git object response")
        for name in names:
            validate_blob(content, name)
    return {"files": len(allowed), "commits": len(revisions), "blobs": len(objects)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = verify_repository(args.root)
    except (ValueError, SyntaxError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Repository verification failed: {exc}\n")
    print(f"Verified {result['files']} tracked files, {result['commits']} reachable commits "
          f"and {result['blobs']} file versions. No excluded paths or translated baseline found.")
    print("This verifies local Git refs, not hosted copies, forks, or a complete privacy/content audit.")


if __name__ == "__main__":
    main()
