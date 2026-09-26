#!/usr/bin/env python3
"""Post-hoc audit of archived scores, without fitting or selecting a detector."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom, spearmanr
from sklearn.metrics import roc_auc_score

from core import evaluate, tie_order

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "deep_audit"


def distribution(values):
    return dict(zip(["p10", "median", "p90"], map(float, np.quantile(values, [.1,.5,.9]))))


def audit_run(path):
    record = json.loads((path / "result.json").read_text())
    z = np.load(path / "scores.npz")
    y, poison, ids = z["labels"], z["poison"].astype(bool), z["ids"]
    sign = 2*y-1
    base = z["base_label_logits"].astype(float)
    ft = z["adapted_label_logits"].astype(float)
    bm, fm = (base[:,1]-base[:,0])*sign, (ft[:,1]-ft[:,0])*sign
    np.testing.assert_allclose(fm-bm, z["train__PD_observed"], atol=1e-10)
    subsets = {"poison":poison, "clean":~poison,
               "clean_positive":(~poison)&(y==1), "clean_negative":(~poison)&(y==0),
               "clean_base_wrong":(~poison)&(bm<0)}
    result = {"run_id":record["run_id"], "config":record["config"],
              "score_sha256":hashlib.sha256((path/"scores.npz").read_bytes()).hexdigest(),
              "margins":{name:{"n":int(mask.sum()), "base":distribution(bm[mask]),
                               "adapted":distribution(fm[mask]),
                               "shift":distribution((fm-bm)[mask])} for name,mask in subsets.items()},
              "methods":{}, "removal":record["removal"]}
    # These are already reported scores, not new fitted detectors.
    scores = {name:z["train__"+name] for name in ["PD_KL", "PD_observed", "base_label_NLL"]}
    scores["adapted_confidence_posthoc"] = -z["train__adapted_label_NLL"]
    heads = {}
    for name, s in scores.items():
        order = tie_order(s, ids)
        heads[name] = set(order[:round(.05*len(y))])
        cumsum = np.cumsum(poison[order])
        m = {"metrics":evaluate(s, poison, ids), "budget_curve":{},
             "oracle_budget_for_recall":{},
             "score_distribution":{key:distribution(s[mask]) for key, mask in subsets.items()}}
        positive = y==1
        m["auroc_within_observed_positive"] = float(roc_auc_score(poison[positive],s[positive]))
        for fraction in [.01,.03,.05,.10,.20,.30]:
            k = max(1, round(len(y)*fraction))
            selected = order[:k]
            tp = int(poison[selected].sum())
            m["budget_curve"][str(fraction)] = {"removed":k, "poisons_removed":tp,
                "recall":tp/int(poison.sum()), "poisons_remaining":int(poison.sum())-tp,
                "clean_removed":k-tp,
                "remaining_poison_rate":(int(poison.sum())-tp)/(len(y)-k)}
        for recall in [.90,.95,.99,1.0]:
            k = int(np.searchsorted(cumsum, np.ceil(recall*poison.sum())))+1
            m["oracle_budget_for_recall"][str(recall)] = k/len(y)
        selected = np.array(sorted(heads[name]))
        fp = selected[~poison[selected]]
        m["top5_false_positives"] = {"n":len(fp), "base_wrong":int((bm[fp]<0).sum()),
                                      "observed_positive":int((y[fp]==1).sum())}
        result["methods"][name] = m
    result["kl_base_head_overlap"] = len(heads["PD_KL"] & heads["base_label_NLL"])/len(heads["PD_KL"])
    result["kl_base_spearman"] = float(spearmanr(scores["PD_KL"], scores["base_label_NLL"]).statistic)
    rate = float(poison.mean())
    result["f1_ceiling_top5"] = 2*min(rate,.05)/(rate+.05)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    runs = [audit_run(p.parent) for p in sorted((ROOT/"results").glob("main*/result.json"))
            if json.loads(p.read_text())["config"]["rate"] > 0]
    groups = {}
    for r in runs:
        c = r["config"]
        key = f'{c["dataset"]} N={c["n"]} p={c["rate"]:g} {c["mode"]}'
        groups.setdefault(key, []).append(r)
    summary = {}
    for key, group in groups.items():
        g = summary[key] = {"seeds":[r["config"]["seed"] for r in group], "methods":{}, "margins":{}}
        for method in group[0]["methods"]:
            stats = {}
            for name, get in {
                "auroc": lambda m:m["metrics"]["auroc"],
                "f1_top5": lambda m:m["metrics"]["top_5pct"]["f1"],
                "recall_top5": lambda m:m["metrics"]["top_5pct"]["recall"],
                "remaining_top5": lambda m:m["budget_curve"]["0.05"]["poisons_remaining"],
                "oracle_budget_recall95": lambda m:m["oracle_budget_for_recall"]["0.95"],
                "within_positive_auroc": lambda m:m["auroc_within_observed_positive"],
            }.items():
                vals = [get(r["methods"][method]) for r in group]
                stats[name] = {"mean":float(np.mean(vals)), "sample_sd":float(np.std(vals,ddof=1))}
            g["methods"][method] = stats
        for subset in group[0]["margins"]:
            g["margins"][subset] = {kind:float(np.mean([r["margins"][subset][kind]["median"] for r in group]))
                                     for kind in ["base", "adapted", "shift"]}
        g["kl_base_head_overlap_mean"] = float(np.mean([r["kl_base_head_overlap"] for r in group]))
        g["f1_ceiling_top5"] = group[0]["f1_ceiling_top5"]
    historical_path = ROOT.parent/"rebuttal/results/gsm8k_removal_rate5.json"
    h = json.loads(historical_path.read_text())
    n,p,k,found = h["n_train"],h["n_poison"],h["top_k_removed"],h["poisons_removed"]
    gsm = {"source":str(historical_path.relative_to(ROOT.parent)),
        "source_sha256":hashlib.sha256(historical_path.read_bytes()).hexdigest(),
        "recorded_auroc":h["PD"]["auroc"], "n":n,"poisons":p,"removed":k,
        "maximum_possible_recall":k/p, "actual_recall":found/p,
        "actual_poisons_remaining":p-found, "actual_remaining_poison_rate":(p-found)/(n-k),
        "perfect_top100_poisons_remaining":p-k, "perfect_top100_remaining_poison_rate":(p-k)/(n-k),
        "random_expected_poisons_removed":k*p/n,
        "random_probability_at_least_observed":float(hypergeom.sf(found-1,n,p,k)),
        "historical_outcomes":{key:h[key] for key in ["poisoned","after_removal","random_removal_control"]}}
    (OUT/"score_audit.json").write_text(json.dumps({"scope":"Post-hoc descriptive analysis; no new efficacy runs",
        "runs":runs,"summary":summary,"gsm8k":gsm},indent=2)+"\n")
    lines = ["# Saved-score deep audit", "", "Post-hoc descriptive analysis, not new defense validation. Means over seeds 42–44.",
             "", "| Setting | Score | AUROC | F1@5% | Recall@5% | Poisons left | Budget for 95% recall (oracle diagnostic) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for key,g in summary.items():
        for name,m in g["methods"].items():
            lines.append(f'| {key} | {name} | {m["auroc"]["mean"]:.3f} | {m["f1_top5"]["mean"]:.3f} | '
                         f'{m["recall_top5"]["mean"]:.1%} | {m["remaining_top5"]["mean"]:.1f} | {m["oracle_budget_recall95"]["mean"]:.1%} |')
    lines += ["", "The 95%-recall budget uses poison annotations retrospectively; it is not a deployable threshold.",
              "Changing removal budgets has not been retrained here; the table predicts residual poison counts, not ASR."]
    (OUT/"score_tables.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"audited_poisoned_runs":len(runs),"gsm8k":gsm},indent=2))


if __name__ == "__main__":
    main()
