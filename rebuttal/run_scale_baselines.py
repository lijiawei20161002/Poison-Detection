#!/usr/bin/env python3
"""
E-B: dataset scale + modern baselines.

Answers:
  * FfY6 Q3  -- is IFE's weakness at 5% intrinsic, or an artifact of N=200 /
                10 poisons?  Run the SAME 5% rate at N=200 / 1000 / 2000.
  * YZho W3  -- add Z-Score (EMNLP'24) and TextGuard (NDSS'24) baselines.
  * YZho W5  -- state the poison rate unambiguously.
  * AC       -- "datasets are relatively small".
  * gC6n Q3  -- false positives as the clean pool grows.

Also re-runs E3 with a CORRECTED ground-truth label set (see common.py note:
the repo's poison_train.jsonl is pre-poisoned, which left 11 unlabelled
poisons in the paper's N=200 setting).

Usage:  python rebuttal/run_scale_baselines.py <attack> <N> [--skip-textguard]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rebuttal.common as C

RATE = 0.05
FT_EPOCHS = 30


def run(attack: str, n: int, skip_textguard: bool = False, seed: int = 42) -> dict:
    k = round(RATE * n)
    print("=" * 78)
    print(f"  ATTACK={attack}   N={n}   poisons={k} ({k/n:.1%})   seed={seed}")
    print("=" * 78)

    pool = C.build_clean_pool(n)
    pidx = C.choose_poison_indices(n, k, seed, pool)
    trigger = C.ALL_ATTACKS[attack]
    train = C.poison_batched(pool, trigger, pidx)
    queries = C.clean_test_queries(50)

    ex = train[sorted(pidx)[0]]
    print(f"  poison example -> {ex.input_text[:110]!r} => {ex.output_text}")

    tok = C.get_tokenizer()
    out: dict = {"attack": attack, "n": n, "n_poison": k, "rate": k / n, "seed": seed,
                 "ft_epochs": FT_EPOCHS}

    # ── victim model: full-parameter fine-tuning ──────────────────────────────
    print("\n  [1] full-parameter fine-tuning victim ...")
    t0 = time.time()
    ft, base_sd = C.finetune_t5(train, tok, epochs=FT_EPOCHS)
    print(f"      {time.time()-t0:.0f}s")
    asr = C.measure_asr(ft, tok, queries, trigger)
    acc = C.measure_clean_acc(ft, tok, queries)
    out["asr_full_ft"] = asr
    out["clean_acc_full_ft"] = acc
    print(f"      ASR={asr:.1%}   clean acc={acc:.1%}")

    det = {}

    # ── PD, full-parameter fine-tuning (reviewer FfY6 Q2) ────────────────────
    print("  [2] PD (full-parameter FT, vs pre-FT checkpoint) ...")
    base = AutoModelForSeq2SeqLM.from_pretrained(C.T5_MODEL).to(C.DEVICE)
    base.load_state_dict({kk: v.to(C.DEVICE) for kk, v in base_sd.items()})
    base.eval()
    t0 = time.time()
    det["PD_fullFT_kl"] = C.evaluate(
        C.pd_scores(ft, tok, train, base_model=base, mode="kl"), pidx, name="PD_fullFT_kl")
    det["PD_fullFT_logodds"] = C.evaluate(
        C.pd_scores(ft, tok, train, base_model=base, mode="logodds"), pidx,
        name="PD_fullFT_logodds")
    out["pd_fullft_runtime_s"] = time.time() - t0
    del base

    # ── STRIP ────────────────────────────────────────────────────────────────
    print("  [3] STRIP ...")
    t0 = time.time()
    det["STRIP"] = C.evaluate(C.strip_scores(ft, tok, train), pidx, name="STRIP")
    out["strip_runtime_s"] = time.time() - t0
    del ft
    torch.cuda.empty_cache()

    # ── PD, LoRA victim ──────────────────────────────────────────────────────
    print("  [4] LoRA victim + PD ...")
    t0 = time.time()
    ft_lora, _ = C.finetune_t5(train, tok, epochs=FT_EPOCHS, lora=True)
    asr_l = C.measure_asr(ft_lora, tok, queries, trigger)
    acc_l = C.measure_clean_acc(ft_lora, tok, queries)
    out["asr_lora"] = asr_l
    out["clean_acc_lora"] = acc_l
    print(f"      LoRA ASR={asr_l:.1%}  clean acc={acc_l:.1%}  ({time.time()-t0:.0f}s)")
    t0 = time.time()
    det["PD_LoRA_kl"] = C.evaluate(
        C.pd_scores(ft_lora, tok, train, mode="kl"), pidx, name="PD_LoRA_kl")
    det["PD_LoRA_logodds"] = C.evaluate(
        C.pd_scores(ft_lora, tok, train, mode="logodds"), pidx, name="PD_LoRA_logodds")
    out["pd_lora_runtime_s"] = time.time() - t0
    del ft_lora
    torch.cuda.empty_cache()

    # ── ONION ────────────────────────────────────────────────────────────────
    print("  [5] ONION ...")
    g_tok = GPT2TokenizerFast.from_pretrained("gpt2")
    g_tok.pad_token = g_tok.eos_token
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(C.DEVICE).eval()
    t0 = time.time()
    det["ONION"] = C.evaluate(C.onion_scores(train, gpt2, g_tok), pidx, name="ONION")
    out["onion_runtime_s"] = time.time() - t0
    del gpt2
    torch.cuda.empty_cache()

    # ── Z-Score (EMNLP 2024) ─────────────────────────────────────────────────
    print("  [6] Z-Score (EMNLP'24) ...")
    t0 = time.time()
    det["ZScore"] = C.evaluate(C.zscore_scores(train), pidx, name="ZScore")
    out["zscore_runtime_s"] = time.time() - t0

    # ── TextGuard (NDSS 2024) ────────────────────────────────────────────────
    if not skip_textguard:
        for m in (3, 9):
            print(f"  [7] TextGuard (NDSS'24), m={m}, TF-IDF+LR groups ...")
            t0 = time.time()
            det[f"TextGuard_lr_m{m}"] = C.evaluate(
                C.textguard_scores(train, tok, m=m, backbone="lr"),
                pidx, name=f"TextGuard_lr_m{m}")
            out[f"textguard_lr_m{m}_runtime_s"] = time.time() - t0
        print("  [7b] TextGuard, m=3, T5-small groups (same backbone as paper) ...")
        t0 = time.time()
        det["TextGuard_t5_m3"] = C.evaluate(
            C.textguard_scores(train, tok, m=3, backbone="t5", epochs=15, verbose=True),
            pidx, name="TextGuard_t5_m3")
        out["textguard_t5_m3_runtime_s"] = time.time() - t0

    out["detectors"] = det

    print(f"\n  RESULTS  attack={attack} N={n} rate={k/n:.0%} ASR={asr:.0%}")
    print(f"  {'method':<26} {'top5%':>22}   AUROC   bestF1(oracle)")
    for name, r in det.items():
        print(C.fmt_row(name, r))

    C.save(out, f"scale_{attack}_N{n}.json")
    return out


if __name__ == "__main__":
    atk = sys.argv[1] if len(sys.argv) > 1 else "cf_prefix"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    skip_tg = "--skip-textguard" in sys.argv
    run(atk, N, skip_textguard=skip_tg)
