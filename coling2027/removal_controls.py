#!/usr/bin/env python3
"""Frozen, resumable removal ablations on the saved three-seed sentiment pools.

This follow-up was motivated by the completed suite and is not part of its
original preregistration. Existing source artifacts are read-only inputs.
"""
import argparse
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoTokenizer

from core import evaluate, score_logits, tie_order
from run import ROOT, fit, new_model, predict, prepare, save_json, task_metrics

DEST = ROOT / "removal_controls"
METHODS = ("base_label_NLL", "adapted_label_confidence_posthoc")
SOURCE_FILES = ("manifest.json", "result.json", "scores.npz",
                "train.jsonl.gz", "test.jsonl.gz", "triggered.jsonl.gz")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def removal_selection(scores, ids, budget):
    """Rank only scores and IDs; poison annotations are intentionally absent."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(ids) or not np.isfinite(scores).all():
        raise ValueError("Invalid removal scores")
    if len(set(ids)) != len(ids) or not 0 < budget < len(ids):
        raise ValueError("Unique IDs and a nonempty retained set are required")
    return tie_order(scores, ids)[:budget]


def method_scores(archive, method):
    if method == "adapted_label_confidence_posthoc":
        return -archive["train__adapted_label_NLL"]
    if method not in (*METHODS, "PD_KL"):
        raise ValueError(f"Unknown method: {method}")
    return archive["train__" + method]


def plan_spec():
    jobs, sources = [], {}
    for seed in (42, 43, 44):
        for dataset, attack, n in (("sst2", "scpn", 6000), ("imdb", "cf", 10000)):
            source = f"main_{dataset}_{attack}_N{n}_p0.05_s{seed}_full"
            directory = ROOT / "results" / source
            original = json.loads((directory / "result.json").read_text())
            config = original["config"]
            if (config["n"], config["seed"], config["rate"], config["mode"], config["budget"]) != (n, seed, .05, "full", .05):
                raise ValueError(f"Unexpected source protocol: {source}")
            sources[source] = {name: sha256(directory / name) for name in SOURCE_FILES}
            if seed == 42 and dataset == "sst2":
                sources[source]["removal_PD_KL_predictions.npz"] = sha256(directory / "removal_PD_KL_predictions.npz")
                jobs.append({"source": source, "method": "PD_KL", "purpose": "reproduction_control"})
            jobs.extend({"source": source, "method": method, "purpose": "new_ablation"} for method in METHODS)
    return {
        "protocol": "sentiment_removal_controls_v1",
        "scope": "Post-hoc extension; new removal outcomes, frozen before follow-up training",
        "directions": {"base_label_NLL": "larger NLL is more suspicious",
                       "adapted_label_confidence_posthoc": "negative adapted-label NLL; larger is more suspicious",
                       "PD_KL": "larger full-vocabulary KL(adapted || base) is more suspicious"},
        "budget": .05,
        "training": "Reuse original config and runner; reset seed, initialize from pinned clean base; ten epochs",
        "ties": "Descending score, then ascending SHA-256 of stable example ID",
        "sources": sources,
        "code_sha256": {name: sha256(ROOT / name) for name in ("removal_controls.py", "run.py", "core.py", "data.py")},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "datasets", "numpy", "scipy", "scikit-learn", "peft")},
        "jobs": jobs,
    }


def freeze_plan():
    DEST.mkdir(exist_ok=True)
    with (DEST / ".plan.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        spec = plan_spec()
        path = DEST / "plan.json"
        if path.exists():
            plan = json.loads(path.read_text())
            if plan["spec"] != spec:
                raise RuntimeError("Sources, code, environment, or plan changed; refusing mixed-protocol resume")
        else:
            plan = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "spec": spec}
            save_json(path, plan)
    return plan


def load_source(job):
    source = ROOT / "results" / job["source"]
    original = json.loads((source / "result.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    data = {role: read_rows(source / f"{role}.jsonl.gz") for role in ("train", "test", "triggered")}
    with np.load(source / "scores.npz", allow_pickle=False) as archive:
        ids = archive["ids"].tolist()
        labels = np.array([row["label"] for row in data["train"]])
        poison = np.array([row["is_poison"] for row in data["train"]], dtype=bool)
        if ids != [row["id"] for row in data["train"]]:
            raise ValueError("Score/data ID alignment failed")
        np.testing.assert_array_equal(labels, archive["labels"])
        np.testing.assert_array_equal(poison, archive["poison"])
        reconstructed = score_logits(archive["base_label_logits"], archive["adapted_label_logits"], labels, [0, 1])
        for name in ("base_label_NLL", "adapted_label_NLL"):
            np.testing.assert_allclose(reconstructed[name], archive["train__" + name], rtol=1e-9, atol=1e-9)
        scores = method_scores(archive, job["method"]).copy()
    return original, manifest, data, ids, poison, scores


def run_job(job, plan):
    name = job["source"] + "__" + job["method"]
    dest = DEST / "results" / name
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / ".run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (dest / "result.json").exists():
            print(f"SKIP {name}", flush=True)
            return
        original, source_manifest, data, ids, poison, scores = load_source(job)
        args = SimpleNamespace(**original["config"])
        budget = round(args.n * plan["spec"]["budget"])
        removed = removal_selection(scores, ids, budget)
        kept = np.ones(len(ids), dtype=bool)
        kept[removed] = False
        if job["purpose"] == "reproduction_control":
            assert [ids[i] for i in removed] == original["removal"][job["method"]]["removed_ids"]
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        label_ids = [source_manifest["label_tokens"][label] for label in ("negative", "positive")]
        assert [tokenizer.encode(label, add_special_tokens=False)[0] for label in ("negative", "positive")] == label_ids
        enc = {role: prepare(rows, tokenizer, args.max_length) for role, rows in data.items()}
        n_retained = int(kept.sum())
        manifest = {"job": job, "source_config": vars(args), "plan_sha256": sha256(DEST / "plan.json"),
                    "source_hashes": plan["spec"]["sources"][job["source"]],
                    "code_sha256": plan["spec"]["code_sha256"], "versions": plan["spec"]["versions"],
                    "historical_code_sha256": source_manifest["code_sha256"],
                    "hardware": torch.cuda.get_device_name(),
                    "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "n_removed": budget, "n_retained": n_retained,
                    "optimizer_steps": args.epochs * ((n_retained + args.batch - 1) // args.batch),
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        save_json(dest / "manifest.json", manifest)
        np.savez_compressed(dest / "selection.npz", ids=ids, scores=scores, removed_indices=removed, retained_mask=kept)
        print(f"START {name} retained={n_retained} steps={manifest['optimizer_steps']}", flush=True)
        start = time.monotonic()
        # The original removal loop resets the torch RNG immediately before
        # constructing each model. Match that boundary, including model loading.
        torch.manual_seed(args.seed)
        model = new_model(args)
        history = fit(model, {key: value[kept] for key, value in enc["train"].items()}, args)
        predictions = {role: predict(model, enc[role], label_ids) for role in ("test", "triggered")}
        checkpoint = ROOT / "checkpoints" / "removal_controls" / name
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        np.savez_compressed(dest / "predictions.npz", **predictions,
                            **{role + "_ids": [row["id"] for row in data[role]] for role in predictions})
        result = {"job": job, "config": vars(args), "n_retained": n_retained, "budget": budget,
                  "optimizer_steps": manifest["optimizer_steps"], "removed_ids": [ids[i] for i in removed],
                  "poisons_removed": int(poison[removed].sum()), "clean_removed": int((~poison[removed]).sum()),
                  "poisons_remaining": int(poison[kept].sum()), "training": history,
                  "detection": evaluate(scores, poison, ids),
                  "evaluation": {role: task_metrics(predictions[role], data[role], role == "triggered") for role in predictions},
                  "checkpoint": str(checkpoint.relative_to(ROOT)),
                  "checkpoint_sha256": {path.name: sha256(path) for path in sorted(checkpoint.glob("*.safetensors"))},
                  "wall_seconds": time.monotonic() - start}
        if job["purpose"] == "reproduction_control":
            with np.load(ROOT / "results" / job["source"] / "removal_PD_KL_predictions.npz") as previous:
                result["reproduction"] = {
                    "prediction_disagreements": {role: int((predictions[role] != previous[role]).sum()) for role in predictions},
                    "historical_evaluation": original["removal"][job["method"]]["evaluation"]}
        save_json(dest / "result.json", result)
        print(json.dumps({"complete": name, "evaluation": result["evaluation"],
                          "poisons_removed": result["poisons_removed"], "seconds": result["wall_seconds"],
                          "reproduction": result.get("reproduction")}), flush=True)
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    cli = parser.parse_args()
    plan = freeze_plan()
    print(f"Frozen plan: {len(plan['spec']['jobs'])} jobs", flush=True)
    if cli.plan_only:
        return
    for job in plan["spec"]["jobs"]:
        if cli.seed is not None and f"_s{cli.seed}_" not in job["source"]:
            continue
        start = time.monotonic()
        status = {"job": job, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            run_job(job, plan)
            status["exit_code"] = 0
        except Exception as error:
            status.update(exit_code=1, error=repr(error))
            raise
        finally:
            status["seconds"] = time.monotonic() - start
            with (DEST / "status.jsonl").open("a") as stream:
                stream.write(json.dumps(status) + "\n")


if __name__ == "__main__":
    main()
