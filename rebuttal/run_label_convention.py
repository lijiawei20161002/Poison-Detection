#!/usr/bin/env python3
"""Does the transformed-query LABEL convention explain the E1 gap?"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rebuttal.common as C, rebuttal.ife as IFE

N, K, EP = 1000, 33, 30

def main():
    pool = C.build_clean_pool(N)
    pidx = C.choose_poison_indices(N, K, 42, pool)
    trig = C.ALL_ATTACKS["ner_james_bond"]
    train = C.poison(pool, trig, pidx)
    q = C.clean_test_queries(50)
    tok = C.get_tokenizer()
    model, _ = C.finetune_t5(train, tok, epochs=EP)
    print(f"ASR={C.measure_asr(model,tok,q,trig):.1%}")
    out = {}
    for flip in (False, True):
        tag = "flip_label" if flip else "keep_label"
        print(f"\n===== {tag} =====")
        inf, cats = IFE.compute_all_influence(
            model, tok, train, q, IFE.PAPER_TRANSFORMS, scope="qv", flip_label=flip)
        rec = {}
        for nm, v in IFE.ife_signals(inf, cats).items():
            rec[nm] = C.evaluate(v, pidx, name=nm)
            print(C.fmt_row(nm, rec[nm]))
        for frac in (0.05, 0.15):
            for mc in (2, 3):
                r = IFE.eval_set(IFE.voting_detect(inf, cats, frac, mc, use="paper"), pidx, N)
                rec[f"voting_top{int(frac*100)}_min{mc}"] = r
                print(f"  voting_top{int(frac*100)}_min{mc:<2} P={r['precision']:.3f} "
                      f"R={r['recall']:.3f} F1={r['f1']:.3f} (flagged {r['n_flagged']})")
        out[tag] = rec
    C.save(out, "label_convention.json")

main()
