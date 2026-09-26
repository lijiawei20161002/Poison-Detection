#!/usr/bin/env python3
"""Standalone comparison figure with all seed points and sample deviations."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "removal_controls"
METHODS = [
    ("no_removal", "No removal"),
    ("PD_KL", "PD-KL"),
    ("base_label_NLL", "Base-label NLL (new)"),
    ("PD_observed", "Observed-label shift"),
    ("adapted_label_confidence_posthoc", "Adapted confidence (new)"),
    ("random", "Random removal"),
    ("oracle", "Oracle removal"),
]


def main():
    with (ROOT / "per_seed.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True, layout="constrained")
    for axis_row, dataset in zip(axes, ("imdb", "sst2")):
        for axis, metric in zip(axis_row, ("asr", "accuracy")):
            bounds = [0., 100.] if metric == "asr" else [82., 94.]
            for index, (method, label) in enumerate(METHODS):
                matching = sorted((row for row in rows if row["dataset"] == dataset and row["method"] == method),
                                  key=lambda row: int(row["seed"]))
                if len(matching) != 3:
                    raise ValueError(f"Need all three seeds: {dataset}, {method}")
                values = np.array([100 * float(row[metric]) for row in matching])
                bounds.extend([values.mean() - values.std(ddof=1), values.mean() + values.std(ddof=1)])
                color = "#b54708" if matching[0]["provenance"] == "new_followup" else "#2563a6"
                axis.errorbar(values.mean(), index, xerr=values.std(ddof=1), fmt="o", color=color,
                              capsize=3, markersize=6, zorder=3)
                axis.scatter(values, index + np.array([-.14, 0, .14]), color=color, alpha=.55,
                             s=18, zorder=2)
            axis.set_yticks(range(len(METHODS)), [label for _, label in METHODS])
            axis.set_ylim(len(METHODS) - .5, -.5)
            axis.grid(axis="x", color="#dddddd", linewidth=.6)
            axis.spines[["top", "right"]].set_visible(False)
            title = "IMDB lexical · N=10,000" if dataset == "imdb" else "SST-2 syntactic · N=6,000"
            axis.set_title(title)
            axis.set_xlabel("Attack success (%) · lower is better" if metric == "asr" else "Clean accuracy (%) · higher is better")
            axis.set_xlim((min(bounds) - 2, max(bounds) + 4) if metric == "asr" else (min(bounds), max(bounds)))
    figure.suptitle("Matched 5% removal and retraining\nLarge dots: means; whiskers: sample SD; small dots: three seeds", fontsize=13)
    for extension in ("png", "pdf"):
        figure.savefig(ROOT / f"removal_comparison.{extension}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
