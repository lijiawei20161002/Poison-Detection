#!/usr/bin/env python3
"""Summarize local evidence without exporting any example text or identifiers."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PROFANITY = re.compile(r"\b(?:fuck(?:ing|ed|s)?|shit(?:ty)?|asshole|bitch(?:es)?)\b", re.I)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Private research archive containing the original coling2027/results")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    records = []
    for p in sorted((source_root / "coling2027/results").glob("main_*/manifest.json")):
        manifest = json.loads(p.read_text())
        config = manifest["config"]
        train = p.parent / "train.jsonl.gz"
        with gzip.open(train, "rt") as stream:
            n_poison = sum(bool(json.loads(line).get("is_poison")) for line in stream if line.strip())
        records.append({"run": p.parent.name, "dataset": config["dataset"],
                        "seed": config["seed"], "mode": config["mode"],
                        "poison_rate": config["rate"], "n_poison": n_poison,
                        **manifest["roles"]})
    if len(records) != 42:
        raise RuntimeError("Expected the 42 archived main-study manifests; refusing an incomplete report")
    samples = []
    for run in ("main_imdb_cf_N10000_p0.05_s42_full", "main_sst2_scpn_N6000_p0.05_s42_full"):
        path = source_root / "coling2027/results" / run / "clean_train.jsonl.gz"
        rows = email = profanity = 0
        with gzip.open(path, "rt") as stream:
            for line in stream:
                text = json.loads(line)["text"]
                rows += 1
                email += bool(EMAIL.search(text))
                profanity += bool(PROFANITY.search(text))
        samples.append({"source": path.relative_to(source_root).as_posix(), "rows": rows,
                        "email_pattern_candidate_rows": email,
                        "profanity_keyword_candidate_rows": profanity})
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    with (docs / "data_statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    report = {"review_date": "2026-10-12", "scope": "two specified local clean-training pools only",
              "method": "case-insensitive profanity keywords and an email-shaped regular expression",
              "limitations": "Candidate flags only; not adjudicated PII, a toxicity classifier, or a full-corpus audit. Names, addresses, obfuscations and contextual harms may be missed.",
              "release_mitigation": "Exclude all raw/rewritten examples, prompt caches, row-level outputs and unsanitized Git history from the public repository; retain only aggregate counts.",
              "samples": samples}
    (docs / "content_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(records)} run-level counts and aggregate screening results; no text or row identifiers exported.")


if __name__ == "__main__":
    main()
