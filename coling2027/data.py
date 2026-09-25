"""Source-pinned, disjoint data pools with explicit poison annotations."""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HK_REV = "a08e959e228327baa0c2906bf943e99a3c89961c"
IMDB_REV = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(text):
    return re.sub(r"\s+", " ", text.replace("<br />", " ")).strip()


def row(text, label, rid, **kw):
    return dict(id=rid, text=normalize(text), label=int(label), **kw)


def unique(rows, excluded=None):
    seen = set() if excluded is None else set(excluded)
    result = []
    for r in rows:
        key = digest(r["text"])
        if key not in seen:
            seen.add(key)
            result.append(r)
    return result


def balanced(rows, n, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c, count in [(0, n//2), (1, n-n//2)]:
        candidates = [i for i, r in enumerate(rows) if r["label"] == c]
        if len(candidates) < count:
            raise ValueError(f"Insufficient class {c} for n={n}")
        idx.extend(rng.permutation(candidates)[:count])
    rng.shuffle(idx)
    return [dict(rows[i]) for i in idx]


def source_data(dataset):
    if dataset == "imdb":
        from datasets import load_dataset
        # This is the exact revision recorded by the first run, now fixed for
        # fresh reproductions as well as for the existing local download cache.
        revision = IMDB_REV
        ds = load_dataset("stanfordnlp/imdb", revision=revision)
        train = unique([row(r["text"], r["label"], f"imdb:train:{i}") for i, r in enumerate(ds["train"])])
        test = unique([row(r["text"], r["label"], f"imdb:test:{i}") for i, r in enumerate(ds["test"])],
                      [digest(r["text"]) for r in train])
        return train, test, {"source": "stanfordnlp/imdb", "revision": revision}
    if dataset != "sst2":
        raise ValueError(dataset)
    p = ROOT / "external" / "HiddenKiller" / "data"
    actual_revision = subprocess.check_output(['git','-C',str(p.parent),'rev-parse','HEAD'],text=True).strip()
    if actual_revision != HK_REV:
        raise ValueError(f'Hidden Killer revision mismatch: {actual_revision}')
    def read(part, split):
        with (p / part / "sst-2" / f"{split}.tsv").open() as f:
            return list(csv.DictReader(f, delimiter="\t"))
    clean, poisoned = read("clean", "train"), read("scpn/20", "train")
    if len(clean) != len(poisoned):
        raise ValueError("Hidden Killer row alignment changed")
    train = []
    for i, (c, a) in enumerate(zip(clean, poisoned)):
        r = row(c["sentence"], c["label"], f"hk:train:{i}")
        if c["sentence"] != a["sentence"]:
            r["scpn"] = normalize(a["sentence"])
        train.append(r)
    eval_clean = read("clean", "test")
    eval_scpn = read("scpn/20", "test")
    negatives = [i for i, r in enumerate(eval_clean) if int(r["label"]) == 0]
    assert len(negatives) == len(eval_scpn) == 912
    mapping = dict(zip(negatives, eval_scpn))
    test = [row(r["sentence"], r["label"], f"hk:test:{i}",
                **({"scpn": normalize(mapping[i]["sentence"])} if i in mapping else {}))
            for i, r in enumerate(eval_clean)]
    dev = [row(r["sentence"], r["label"], f"hk:dev:{i}") for i,r in enumerate(read("clean", "dev"))]
    return unique(train), (dev, unique(test)), {"source": "thunlp/HiddenKiller", "revision": HK_REV,
        "released_changed_rows": sum("scpn" in r for r in train),
        "released_changed_negative_rows": sum("scpn" in r and r["label"] == 0 for r in train),
        "attack_fidelity": "Author-released SCPN text; resampled source-negative poisons, original victim replaced by FLAN-T5"}


def make_data(dataset, n, rate, seed, attack):
    train_all, test_all, provenance = source_data(dataset)
    train = balanced(train_all, n, seed+10000)
    if dataset == "imdb":
        reference = balanced(test_all, 2000, 2718)
        calibration, test = reference[:1000], reference[1000:]
    else:
        dev, official_test = test_all
        calibration, test = dev, official_test
    # Remove any normalized text overlap across all roles (including rewrites).
    train_hash = {digest(r["text"]) for r in train}
    train_hash.update(digest(r["scpn"]) for r in train if "scpn" in r)
    calibration = unique(calibration, train_hash)
    test = unique(test, train_hash | {digest(r["text"]) for r in calibration})
    candidates = [i for i,r in enumerate(train) if r["label"] == 0 and
                  (attack != "scpn" or "scpn" in r)]
    k = round(rate*n)
    if len(candidates) < k:
        raise ValueError(f"Only {len(candidates)} released negative-source rewrites for {k} poisons")
    selected = set(np.random.default_rng(seed+20000).permutation(candidates)[:k].tolist())
    clean_train = [dict(r) for r in train]
    for i, r in enumerate(train):
        r["is_poison"] = i in selected
        r["original_label"] = r["label"]
        if i in selected:
            r["text"] = r["scpn"] if attack == "scpn" else "CF " + r["text"]
            r["label"] = 1
    # ASR denominator is ONLY originally non-target examples.
    triggered = []
    for r in test:
        if r["label"] != 0:
            continue
        t = r["scpn"] if attack == "scpn" else "CF " + r["text"]
        if digest(t) not in train_hash:
            triggered.append(dict(r, text=t))
    return dict(train=train, clean_train=clean_train, calibration=calibration, test=test,
                triggered=triggered, provenance=provenance)
