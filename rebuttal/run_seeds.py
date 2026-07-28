#!/usr/bin/env python3
"""
Error bars (reviewer FfY6: "key results rest on small sample sizes with limited
statistical rigor").

Repeats the cheap detectors over independent poison draws x fine-tuning seeds at
N=1000, and reports mean +- std.  Covers PD (full-FT and LoRA), Z-Score and
TextGuard; STRIP/ONION are excluded only because they cost ~20 min/seed.

Usage: python rebuttal/run_seeds.py [attack] [n_seeds]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rebuttal.common as C

N = 1000
RATE = 0.05
FT_EPOCHS = 30


def one_seed(attack: str, seed: int) -> dict:
    k = round(RATE * N)
    pool = C.build_clean_pool(N)
    pidx = C.choose_poison_indices(N, k, seed, pool)
    trig = C.ALL_ATTACKS[attack]
    train = C.poison_batched(pool, trig, pidx)
    queries = C.clean_test_queries(50)
    tok = C.get_tokenizer()

    ft, base_sd = C.finetune_t5(train, tok, epochs=FT_EPOCHS, seed=seed)
    asr = C.measure_asr(ft, tok, queries, trig)
    acc = C.measure_clean_acc(ft, tok, queries)

    base = AutoModelForSeq2SeqLM.from_pretrained(C.T5_MODEL).to(C.DEVICE)
    base.load_state_dict({kk: v.to(C.DEVICE) for kk, v in base_sd.items()})
    base.eval()

    out = {"seed": seed, "asr": asr, "clean_acc": acc, "det": {}}
    out["det"]["PD_fullFT_kl"] = C.evaluate(
        C.pd_scores(ft, tok, train, base_model=base, mode="kl"), pidx)
    out["det"]["PD_fullFT_logodds"] = C.evaluate(
        C.pd_scores(ft, tok, train, base_model=base, mode="logodds"), pidx)
    out["det"]["ZScore"] = C.evaluate(C.zscore_scores(train), pidx)
    out["det"]["TextGuard_lr_m3"] = C.evaluate(
        C.textguard_scores(train, tok, m=3, backbone="lr"), pidx)
    del ft, base
    torch.cuda.empty_cache()

    ft_l, _ = C.finetune_t5(train, tok, epochs=FT_EPOCHS, lora=True, seed=seed)
    out["asr_lora"] = C.measure_asr(ft_l, tok, queries, trig)
    out["det"]["PD_LoRA_kl"] = C.evaluate(
        C.pd_scores(ft_l, tok, train, mode="kl"), pidx)
    out["det"]["PD_LoRA_logodds"] = C.evaluate(
        C.pd_scores(ft_l, tok, train, mode="logodds"), pidx)
    del ft_l
    torch.cuda.empty_cache()
    return out


def main(attack: str = "cf_prefix", n_seeds: int = 3):
    runs = []
    for s in (42, 43, 44, 45, 46)[:n_seeds]:
        print(f"\n{'='*70}\n  {attack}  seed={s}\n{'='*70}")
        r = one_seed(attack, s)
        print(f"  ASR(full)={r['asr']:.1%}  ASR(LoRA)={r['asr_lora']:.1%}")
        for nm, v in r["det"].items():
            print("  ", C.fmt_row(nm, v))
        runs.append(r)
        C.save({"attack": attack, "runs": runs}, f"seeds_{attack}.json")

    print(f"\n{'='*70}\n  SUMMARY over {len(runs)} seeds (mean +- std), N={N}, "
          f"rate={RATE:.0%}, attack={attack}\n{'='*70}")
    print(f"  ASR full-FT   {np.mean([r['asr'] for r in runs]):.1%} "
          f"+- {np.std([r['asr'] for r in runs]):.1%}")
    print(f"  ASR LoRA      {np.mean([r['asr_lora'] for r in runs]):.1%} "
          f"+- {np.std([r['asr_lora'] for r in runs]):.1%}")
    print(f"\n  {'method':<22}{'AUROC':>16}{'F1@top5%':>16}{'bestF1':>16}")
    agg = {}
    for nm in runs[0]["det"]:
        au = [r["det"][nm]["auroc"] for r in runs]
        f5 = [r["det"][nm]["top_5pct"]["f1"] for r in runs]
        bf = [r["det"][nm]["best_f1_oracle_sweep"]["f1"] for r in runs]
        agg[nm] = {"auroc_mean": float(np.mean(au)), "auroc_std": float(np.std(au)),
                   "f1_top5_mean": float(np.mean(f5)), "f1_top5_std": float(np.std(f5)),
                   "bestf1_mean": float(np.mean(bf)), "bestf1_std": float(np.std(bf))}
        print(f"  {nm:<22}{np.mean(au):7.3f}+-{np.std(au):.3f}"
              f"{np.mean(f5):8.3f}+-{np.std(f5):.3f}"
              f"{np.mean(bf):8.3f}+-{np.std(bf):.3f}")
    C.save({"attack": attack, "n_seeds": len(runs), "runs": runs, "summary": agg},
           f"seeds_{attack}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cf_prefix",
         int(sys.argv[2]) if len(sys.argv) > 2 else 3)
