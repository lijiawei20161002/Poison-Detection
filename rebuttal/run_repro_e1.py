#!/usr/bin/env python3
"""
Reproduction attempt for the paper's headline result (Table 3 / Experiment E1).

Paper E1: T5-small, N=1000, 3.3% poison rate (33 poisons), NER "J. Bond"
trigger, EK-FAC over ALL LINEAR LAYERS, 6 transforms in 3 categories.
Claimed: Variance ensemble P=66 R=100 F1=79.5; Voting P=100 R=91 F1=95.2.

We sweep the two implementation choices that could plausibly account for a
gap -- the tracked-parameter scope and the poison rate -- and report every
detection rule at fixed operating points plus AUROC.

Usage: python rebuttal/run_repro_e1.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rebuttal.common as C
import rebuttal.ife as IFE

N = 1000
N_QUERIES = 50
FT_EPOCHS = 30

CONFIGS = [
    # (label, attack, n_poison, scope)
    ("E1_exact_NER_3.3pct_all_linear", "ner_james_bond", 33, "all_linear"),
    ("E1_exact_NER_3.3pct_qv",         "ner_james_bond", 33, "qv"),
    ("CF_5pct_all_linear",             "cf_prefix",      50, "all_linear"),
]


def run(label, attack, k, scope):
    print("=" * 78)
    print(f"  {label}:  attack={attack}  N={N}  poisons={k} ({k/N:.1%})  scope={scope}")
    print("=" * 78)

    pool = C.build_clean_pool(N)
    pidx = C.choose_poison_indices(N, k, 42, pool)
    trig = C.ALL_ATTACKS[attack]
    train = C.poison_batched(pool, trig, pidx)
    queries = C.clean_test_queries(N_QUERIES)

    tok = C.get_tokenizer()
    t0 = time.time()
    model, _ = C.finetune_t5(train, tok, epochs=FT_EPOCHS)
    asr = C.measure_asr(model, tok, queries, trig)
    acc = C.measure_clean_acc(model, tok, queries)
    print(f"  FT {time.time()-t0:.0f}s | ASR={asr:.1%} clean_acc={acc:.1%}")

    t0 = time.time()
    inf, cats = IFE.compute_all_influence(
        model, tok, train, queries, IFE.PAPER_TRANSFORMS, scope=scope)
    print(f"  influence sweep {time.time()-t0:.0f}s")

    res = {"label": label, "attack": attack, "n": N, "n_poison": k, "rate": k / N,
           "scope": scope, "asr": asr, "clean_acc": acc, "signals": {}, "rules": {},
           "per_transform": {}}

    for name, v in IFE.ife_signals(inf, cats).items():
        res["signals"][name] = C.evaluate(v, pidx, name=name)
    for name in IFE.PAPER_TRANSFORMS:
        res["per_transform"][name] = {"category": cats[name],
                                      **C.evaluate(inf[name].mean(1), pidx, name=name)}

    # Voting rule at several per-transform thresholds / consensus requirements
    for frac in (0.03, 0.05, 0.10, 0.15):
        for minc in (2, 3):
            res["rules"][f"voting_top{int(frac*100)}_min{minc}"] = IFE.eval_set(
                IFE.voting_detect(inf, cats, frac, minc, use="paper"), pidx, N)
    for frac in (0.05, 0.15):
        res["rules"][f"cross_type_top{int(frac*100)}"] = IFE.eval_set(
            IFE.cross_type_detect(inf, cats, frac), pidx, N)

    print("\n  --- signals ---")
    for name, r in res["signals"].items():
        print(C.fmt_row(name, r))
    print("  --- per transform (top-5%) ---")
    for name, r in res["per_transform"].items():
        print(f"  {name:<24} ({r['category']:<10}) F1={r['top_5pct']['f1']:.3f} "
              f"AUROC={r['auroc']:.3f}")
    print("  --- voting / cross-type rules ---")
    for name, r in res["rules"].items():
        print(f"  {name:<24} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} (flagged {r['n_flagged']})")

    del model
    torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    out = {}
    for cfg in CONFIGS:
        try:
            out[cfg[0]] = run(*cfg)
        except Exception:
            import traceback
            traceback.print_exc()
            out[cfg[0]] = {"error": traceback.format_exc()}
        C.save(out, "repro_e1.json")
    print("\nDONE")
