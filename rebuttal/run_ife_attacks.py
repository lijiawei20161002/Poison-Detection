#!/usr/bin/env python3
"""
E-C + E-D + IFE half of E-B: IFE across attack types and dataset scales.

Answers:
  * YZho W1 -- canonical Hidden Killer (syntactic template paraphrase) and
               LISM (style transfer) instead of the paper's literal-prefix
               stand-ins.
  * YZho W2 -- BGMAttack (black-box generative-model rewrite).
  * FfY6 Q1 -- can ANY transform class break the symmetry for syntactic
               triggers?  Adds a 4th `syntactic` transform category and a
               combined semantic-invariance x structural-sensitivity rule.
  * FfY6 Q3 -- same 5% rate at N=200 vs N=1000.
  * PAT     -- settles the Section 4.1 / 4.2 variance-polarity contradiction
               empirically (variance_high vs variance_low).

Usage:  python rebuttal/run_ife_attacks.py <N> [attack ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rebuttal.common as C
import rebuttal.ife as IFE

RATE = 0.05
FT_EPOCHS = 30
N_QUERIES = 50

ALL_TRANSFORMS = IFE.PAPER_TRANSFORMS + IFE.SYNTACTIC_TRANSFORMS


def run_one(attack: str, n: int, seed: int = 42) -> dict:
    k = round(RATE * n)
    print("=" * 78)
    print(f"  IFE | attack={attack}  N={n}  poisons={k} ({k/n:.1%})")
    print("=" * 78)

    pool = C.build_clean_pool(n)
    pidx = C.choose_poison_indices(n, k, seed, pool)
    trig = C.ALL_ATTACKS[attack]
    train = C.poison_batched(pool, trig, pidx)
    queries = C.clean_test_queries(N_QUERIES)

    ex = train[sorted(pidx)[0]]
    print(f"  poisoned example: {ex.input_text[:150]!r}")

    tok = C.get_tokenizer()
    t0 = time.time()
    model, _ = C.finetune_t5(train, tok, epochs=FT_EPOCHS)
    asr = C.measure_asr(model, tok, queries, trig)
    acc = C.measure_clean_acc(model, tok, queries)
    print(f"  fine-tuned in {time.time()-t0:.0f}s | ASR={asr:.1%} clean_acc={acc:.1%}")

    t0 = time.time()
    inf, cats = IFE.compute_all_influence(model, tok, train, queries, ALL_TRANSFORMS)
    infl_time = time.time() - t0
    print(f"  influence sweep in {infl_time:.0f}s")

    res: dict = {
        "attack": attack, "n": n, "n_poison": k, "rate": k / n, "seed": seed,
        "asr": asr, "clean_acc": acc, "influence_runtime_s": infl_time,
        "signals": {}, "rules": {},
    }

    sig = IFE.ife_signals(inf, cats)
    for name, v in sig.items():
        res["signals"][name] = C.evaluate(v, pidx, name=name)

    # per-transform single-transform baselines at the paper's top-5% threshold
    res["per_transform"] = {}
    for name in ALL_TRANSFORMS:
        s = inf[name].mean(1)
        res["per_transform"][name] = {
            "category": cats[name],
            **C.evaluate(s, pidx, name=name),
        }

    # ensemble rules
    res["rules"]["voting_paper_top5_min2"] = IFE.eval_set(
        IFE.voting_detect(inf, cats, 0.05, 2, use="paper"), pidx, n)
    res["rules"]["voting_all_top5_min2"] = IFE.eval_set(
        IFE.voting_detect(inf, cats, 0.05, 2, use="all"), pidx, n)
    res["rules"]["cross_type_top15"] = IFE.eval_set(
        IFE.cross_type_detect(inf, cats, 0.15), pidx, n)

    print(f"\n  --- signals (attack={attack}, N={n}, ASR={asr:.0%}) ---")
    for name, r in res["signals"].items():
        print(C.fmt_row(name, r))
    print("  --- ensemble rules ---")
    for name, r in res["rules"].items():
        print(f"  {name:<26} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} (flagged {r['n_flagged']})")
    print("  --- best single transform by AUROC ---")
    best = max(res["per_transform"].items(), key=lambda kv: kv[1]["auroc"])
    print(f"  {best[0]} ({best[1]['category']}) AUROC={best[1]['auroc']:.3f} "
          f"top5%F1={best[1]['top_5pct']['f1']:.3f}")

    del model
    torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    attacks = sys.argv[2:] or list(C.ALL_ATTACKS.keys())
    out = {}
    for a in attacks:
        try:
            out[a] = run_one(a, N)
            C.save(out, f"ife_attacks_N{N}.json")
        except Exception as e:
            import traceback
            traceback.print_exc()
            out[a] = {"error": str(e)}
    C.save(out, f"ife_attacks_N{N}.json")
    print("\nDONE")
