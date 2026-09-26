#!/usr/bin/env python3
"""Verify the follow-up artifacts and report paired three-seed comparisons."""
import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from core import evaluate
from removal_controls import DEST, ROOT, load_source, read_rows, removal_selection, sha256
from run import save_json, task_metrics


def verify(plan, check_checkpoints=True):
    for source, hashes in plan["spec"]["sources"].items():
        for name, expected in hashes.items():
            assert sha256(ROOT / "results" / source / name) == expected, (source, name)
        clean_source = ROOT / "results" / source.replace("_p0.05_", "_p0_")
        for role in ("test", "triggered"):
            assert read_rows(ROOT / "results" / source / f"{role}.jsonl.gz") == read_rows(clean_source / f"{role}.jsonl.gz")
    for name, expected in plan["spec"]["code_sha256"].items():
        assert sha256(ROOT / name) == expected, name
    completed, pending = [], []
    for job in plan["spec"]["jobs"]:
        directory = DEST / "results" / (job["source"] + "__" + job["method"])
        path = directory / "result.json"
        if not path.exists():
            pending.append(job)
            continue
        result = json.loads(path.read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        original, _, data, ids, poison, scores = load_source(job)
        assert result["job"] == manifest["job"] == job
        assert result["config"] == original["config"] == manifest["source_config"]
        assert manifest["plan_sha256"] == sha256(DEST / "plan.json")
        assert manifest["source_hashes"] == plan["spec"]["sources"][job["source"]]
        assert manifest["code_sha256"] == plan["spec"]["code_sha256"]
        assert manifest["versions"] == plan["spec"]["versions"]
        budget = round(len(ids) * plan["spec"]["budget"])
        removed = removal_selection(scores, ids, budget)
        keep = np.ones(len(ids), dtype=bool)
        keep[removed] = False
        with np.load(directory / "selection.npz", allow_pickle=False) as selection:
            np.testing.assert_array_equal(selection["ids"], ids)
            np.testing.assert_array_equal(selection["scores"], scores)
            np.testing.assert_array_equal(selection["removed_indices"], removed)
            np.testing.assert_array_equal(selection["retained_mask"], keep)
        assert result["removed_ids"] == [ids[i] for i in removed]
        assert result["budget"] == manifest["n_removed"] == budget
        assert result["n_retained"] == manifest["n_retained"] == int(keep.sum())
        assert result["poisons_removed"] == int(poison[removed].sum())
        assert result["clean_removed"] == int((~poison[removed]).sum())
        assert result["poisons_remaining"] == int(poison[keep].sum())
        assert result["detection"] == evaluate(scores, poison, ids)
        config = result["config"]
        steps = config["epochs"] * ((int(keep.sum()) + config["batch"] - 1) // config["batch"])
        assert result["optimizer_steps"] == manifest["optimizer_steps"] == steps
        assert [item["epoch"] for item in result["training"]] == list(range(1, config["epochs"] + 1))
        assert all(np.isfinite(item["loss"]) for item in result["training"])
        with np.load(directory / "predictions.npz", allow_pickle=False) as predictions:
            for role in ("test", "triggered"):
                assert predictions[role + "_ids"].tolist() == [row["id"] for row in data[role]]
                assert set(predictions[role].tolist()) <= {0, 1}
                assert result["evaluation"][role] == task_metrics(predictions[role], data[role], role == "triggered")
            if job["purpose"] == "reproduction_control":
                assert result["removed_ids"] == original["removal"][job["method"]]["removed_ids"]
                with np.load(ROOT / "results" / job["source"] / "removal_PD_KL_predictions.npz") as previous:
                    differences = {role: int((predictions[role] != previous[role]).sum()) for role in ("test", "triggered")}
                assert result["reproduction"]["prediction_disagreements"] == differences
                assert result["reproduction"]["historical_evaluation"] == original["removal"][job["method"]]["evaluation"]
        if check_checkpoints:
            assert result["checkpoint_sha256"]
            for name, expected in result["checkpoint_sha256"].items():
                assert sha256(ROOT / result["checkpoint"] / name) == expected
        completed.append(result)
    return completed, pending


def row(source, method, record, provenance):
    config = source["config"]
    return {"dataset": config["dataset"], "attack": config["attack"], "n": config["n"],
            "seed": config["seed"], "method": method, "provenance": provenance,
            "poisons_removed": record.get("poisons_removed", 0),
            "clean_removed": record.get("clean_removed", 0),
            "asr": record["evaluation"]["triggered"]["asr"],
            "accuracy": record["evaluation"]["test"]["accuracy"]}


def mean_sd(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def summarize(plan, completed):
    rows = []
    selections = {}
    for source in plan["spec"]["sources"]:
        original = json.loads((ROOT / "results" / source / "result.json").read_text())
        rows.append(row(original, "no_removal", original, "original_suite"))
        for method, record in original["removal"].items():
            rows.append(row(original, method, record, "original_suite"))
            selections[(original["config"]["dataset"], original["config"]["seed"], method)] = set(record["removed_ids"])
        clean_path = ROOT / "results" / source.replace("_p0.05_", "_p0_") / "result.json"
        clean = json.loads(clean_path.read_text())
        rows.append(row(original, "clean_trained", clean, "original_suite"))
    for result in completed:
        if result["job"]["purpose"] == "new_ablation":
            rows.append(row(result, result["job"]["method"], result, "new_followup"))
            selections[(result["config"]["dataset"], result["config"]["seed"], result["job"]["method"])] = set(result["removed_ids"])
    grouped = defaultdict(list)
    for record in rows:
        grouped[(record["dataset"], record["method"])].append(record)
    aggregate = []
    for (dataset, method), group in grouped.items():
        assert len({record["seed"] for record in group}) == len(group)
        aggregate.append({"dataset": dataset, "method": method, "seeds": sorted(record["seed"] for record in group),
                          "provenance": group[0]["provenance"],
                          **{key: mean_sd([record[key] for record in group]) for key in
                             ("poisons_removed", "clean_removed", "asr", "accuracy")}})
    pairs = []
    for dataset in ("imdb", "sst2"):
        for new_method, reference in (("base_label_NLL", "PD_KL"),
                                      ("adapted_label_confidence_posthoc", "PD_observed")):
            new = {record["seed"]: record for record in grouped[(dataset, new_method)]}
            old = {record["seed"]: record for record in grouped[(dataset, reference)]}
            seeds = sorted(set(new) & set(old))
            if not seeds:
                continue
            deltas = [{"seed": seed, **{key: new[seed][key] - old[seed][key] for key in
                                       ("poisons_removed", "clean_removed", "asr", "accuracy")}} for seed in seeds]
            for delta in deltas:
                new_ids = selections[(dataset, delta["seed"], new_method)]
                old_ids = selections[(dataset, delta["seed"], reference)]
                assert len(new_ids) == len(old_ids)
                delta["removal_overlap_fraction"] = len(new_ids & old_ids) / len(new_ids)
            pairs.append({"dataset": dataset, "new_method": new_method, "reference": reference,
                          "direction": "new minus reference", "seeds": seeds, "per_seed": deltas,
                          **{key: mean_sd([delta[key] for delta in deltas]) for key in
                             ("poisons_removed", "clean_removed", "asr", "accuracy", "removal_overlap_fraction")}})
    return rows, aggregate, pairs


def formatted(stats, scale=1):
    text = f"{scale * stats['mean']:.1f}"
    return text + (f" ± {scale * stats['sd']:.1f}" if stats["sd"] is not None else " (single seed)")


def replay_checkpoints(completed):
    """Reload every saved model and independently reproduce its predictions."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from run import predict, prepare

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    replays = []
    for result in completed:
        _, manifest, data, _, _, _ = load_source(result["job"])
        checkpoint = ROOT / result["checkpoint"]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).cuda().eval()
        label_ids = [manifest["label_tokens"][label] for label in ("negative", "positive")]
        directory = DEST / "results" / (result["job"]["source"] + "__" + result["job"]["method"])
        counts = {}
        with np.load(directory / "predictions.npz") as saved:
            for role in ("test", "triggered"):
                enc = prepare(data[role], tokenizer, result["config"]["max_length"])
                predictions = predict(model, enc, label_ids)
                np.testing.assert_array_equal(predictions, saved[role])
                counts[role] = len(predictions)
        replays.append({"job": result["job"], "matched_predictions": counts})
        print("REPLAY PASS", directory.name, flush=True)
        del model
        torch.cuda.empty_cache()
    return replays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--skip-checkpoint-hashes", action="store_true", help="For intermediate progress reports only")
    parser.add_argument("--replay-checkpoints", action="store_true", help="Reload every completed model on GPU and verify predictions")
    args = parser.parse_args()
    plan = json.loads((DEST / "plan.json").read_text())
    completed, pending = verify(plan, not args.skip_checkpoint_hashes)
    replays = replay_checkpoints(completed) if args.replay_checkpoints else []
    rows, aggregate, paired = summarize(plan, completed)
    report = {"completed": len(completed), "expected": len(plan["spec"]["jobs"]), "pending": pending,
              "new_ablations": sum(result["job"]["purpose"] == "new_ablation" for result in completed),
              "reproduction_controls": [result["reproduction"] for result in completed if "reproduction" in result],
              "checkpoint_hashes_checked": not args.skip_checkpoint_hashes,
              "checkpoint_prediction_replays": replays,
              "clean_reference_sha256": {
                  source.replace("_p0.05_", "_p0_"): sha256(ROOT / "results" / source.replace("_p0.05_", "_p0_") / "result.json")
                  for source in plan["spec"]["sources"]},
              "analysis_code_sha256": sha256(Path(__file__)), "plan_sha256": sha256(DEST / "plan.json"),
              "checks": ["frozen source and code hashes", "score/logit reconstruction and ID alignment",
                         "identical evaluation examples in paired clean-training controls",
                         "fixed directions, ties, and exact removal budgets", "poison/clean removal counts",
                         "ten finite-loss epochs and expected optimizer-step count",
                         "prediction IDs, denominators, ASR, per-class and overall accuracy",
                         "reproduction-control prediction disagreements"],
              "limits": "Saved class predictions replayed from checkpoints; full-vocabulary detector scores are inherited from the original suite."
                        if replays else "Checkpoint prediction replay was not requested in this verification."}
    save_json(DEST / "verification.json", report)
    save_json(DEST / "summary.json", {"aggregate": aggregate, "paired_differences": paired})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (DEST / "per_seed.csv").write_text(stream.getvalue())
    lines = ["# Sentiment removal controls — 26 September 2026", "",
             f"Completed **{len(completed)}/{len(plan['spec']['jobs'])}** planned runs, including "
             f"{report['new_ablations']} new ablations and {len(report['reproduction_controls'])} reproduction control(s).", "",
             "This follow-up tests whether base-model disagreement or adapted-model confidence alone explains "
             "the earlier remediation results. It was motivated by the completed study; its directions, seeds, "
             "and budgets were frozen before these new training runs.", "",
             "All removal arms discard exactly 5% of the same saved poisoned pools and retrain FLAN-T5-small "
             "from the same pinned base with the original ten-epoch protocol. Seeds are 42, 43, and 44. "
             "The two new scores are base-label NLL (larger is suspicious) and negative adapted-label NLL "
             "(higher confidence is suspicious). Neither uses poison labels to rank. Confidence remains a "
             "post-hoc ablation; no score direction is chosen per dataset.", "",
             "## Matched results", "",
             "Mean ± sample standard deviation across available seeds. ASR and clean accuracy are percentages. "
             "The original rows are unchanged reference results, not new reruns. Clean-trained controls use "
             "unmodified data; no-removal and clean-trained models have more optimizer steps than filtered models.", "",
             "| Dataset | Method | Record | Seeds | Poisons removed | Clean removed | ASR (%) | Clean accuracy (%) |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    order = ("clean_trained", "no_removal", "PD_KL", "base_label_NLL", "PD_observed",
             "adapted_label_confidence_posthoc", "ZScore_released_unigram", "random", "oracle")
    lookup = {(record["dataset"], record["method"]): record for record in aggregate}
    for dataset in ("imdb", "sst2"):
        for method in order:
            record = lookup.get((dataset, method))
            if record is None:
                continue
            origin = "new" if record["provenance"] == "new_followup" else "original"
            counts = "— | —" if method == "clean_trained" else f"{formatted(record['poisons_removed'])} | {formatted(record['clean_removed'])}"
            lines.append(f"| {dataset} | {method} | {origin} | {len(record['seeds'])} | {counts} | "
                         f"{formatted(record['asr'], 100)} | {formatted(record['accuracy'], 100)} |")
    lines += ["", "## Paired differences", "",
              "New arm minus the indicated original arm, pairing the same seed and training pool. "
              "Negative ASR differences favor the new arm; positive accuracy differences favor the new arm. "
              "Overlap is the intersection of the two removal sets divided by their common budget. "
              "These three-seed summaries are descriptive, not significance claims.", "",
              "| Dataset | New arm minus reference | Seeds | ASR difference (pp) | Clean accuracy difference (pp) | Removal-set overlap (%) |",
              "|---|---|---:|---:|---:|---:|"]
    for record in paired:
        lines.append(f"| {record['dataset']} | {record['new_method']} − {record['reference']} | {len(record['seeds'])} | "
                     f"{formatted(record['asr'], 100)} | {formatted(record['accuracy'], 100)} | {formatted(record['removal_overlap_fraction'], 100)} |")
    lines += ["", "## Reproduction and artifacts", ""]
    for record in report["reproduction_controls"]:
        differences = record["prediction_disagreements"]
        lines.append(f"The SST-2 seed-42 PD-KL replay disagrees with the saved original predictions on "
                     f"**{differences['test']} clean and {differences['triggered']} triggered examples**. "
                     "The selected training-example IDs match exactly.")
    lines += ["", "`plan.json` records the frozen source hashes, code hashes, score directions, and jobs. "
              "`results/` contains selections, prediction IDs, training histories, manifests, and metrics. "
              "Local checkpoints are saved under `../checkpoints/removal_controls/`. "
              "`per_seed.csv`, `summary.json`, and `verification.json` provide the complete numerical records.", "",
              "```bash", "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python coling2027/removal_controls.py",
              "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python coling2027/summarize_removal_controls.py --require-complete", "```", "",
              "Add `--replay-checkpoints` to reload each saved model on GPU and check every clean/triggered prediction.", "",
              "This extension covers full fine-tuning on the two existing sentiment settings. It does not "
              "validate the GSM8K fixes, separate dataset from attack-family effects, or establish generative defense.", ""]
    (DEST / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))
    if args.require_complete and pending:
        raise SystemExit("Incomplete experiment plan")


if __name__ == "__main__":
    main()
