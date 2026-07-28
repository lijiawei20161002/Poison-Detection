#!/usr/bin/env python3
"""
E-A + E-F + fusion analysis.

  E-A (gC6n Q3)  : clean-only control -- with NO poison in the data, what does
                   each detector flag?  Every flag is a false positive by
                   construction, so this measures the false-positive burden
                   directly, and repeats it as the clean pool grows.
  E-F (YZho W4)  : does PD degrade when the BASE model is bad at the downstream
                   task?  Sweep base checkpoints of differing zero-shot
                   competence and correlate with PD's AUROC / precision.
  FfY6 Q4        : why does rank-average fusion of PD with the class-conditioned
                   spectral signal LOSE F1 relative to PD alone?

Usage: python rebuttal/run_pd_analysis.py [clean|competence|fusion|all]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rebuttal.common as C

FT_EPOCHS = 30

# base checkpoints ordered by expected zero-shot sentiment competence
BASE_MODELS = [
    ("t5-small-lm-adapt", "google/t5-small-lm-adapt"),
    ("t5-base-lm-adapt", "google/t5-base-lm-adapt"),
    ("flan-t5-small", "google/flan-t5-small"),
    ("flan-t5-base", "google/flan-t5-base"),
]


# ── class-conditioned spectral signature ─────────────────────────────────────

@torch.no_grad()
def encoder_features(model, tok, samples, batch: int = 64, max_len: int = 128):
    feats = []
    for b in range(0, len(samples), batch):
        chunk = [s.input_text for s in samples[b : b + batch]]
        enc = tok(chunk, max_length=max_len, truncation=True, padding=True,
                  return_tensors="pt").to(C.DEVICE)
        h = model.get_encoder()(**enc).last_hidden_state
        m = enc.attention_mask.unsqueeze(-1).float()
        feats.append(((h * m).sum(1) / m.sum(1)).float().cpu())
    return torch.cat(feats).numpy()


def spectral_scores(feats: np.ndarray, samples, class_conditioned: bool = True) -> np.ndarray:
    """
    Spectral Signature (Tran et al. 2018).  class_conditioned=True restricts the
    SVD to the target-label subset, which REQUIRES knowing the target label and
    therefore steps outside the paper's own threat model (Sec. 3); we keep it
    only as an upper bound, as the paper does.
    """
    out = np.zeros(len(feats))
    if class_conditioned:
        idx = np.array([i for i, s in enumerate(samples)
                        if s.output_text == C.TARGET_LABEL])
    else:
        idx = np.arange(len(feats))
    X = feats[idx]
    X = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    out[idx] = np.abs(X @ Vt[0])
    return out


def rank_avg(*score_arrays) -> np.ndarray:
    """Average of per-signal ranks (higher rank = more suspicious)."""
    rs = []
    for s in score_arrays:
        order = np.argsort(np.argsort(s))       # 0 = lowest
        rs.append(order / (len(s) - 1))
    return np.mean(rs, 0)


# ── E-A: clean-only control ──────────────────────────────────────────────────

def run_clean_control(sizes=(200, 1000)) -> dict:
    print("=" * 78)
    print("  E-A  CLEAN-ONLY CONTROL  (no poison at all -> every flag is an FP)")
    print("=" * 78)
    tok = C.get_tokenizer()
    out = {}
    for n in sizes:
        pool = C.build_clean_pool(n)
        queries = C.clean_test_queries(50)
        print(f"\n  N={n}, 0 poisons")
        ft, base_sd = C.finetune_t5(pool, tok, epochs=FT_EPOCHS)
        acc = C.measure_clean_acc(ft, tok, queries)
        # a trigger the model has never seen: measures spurious "ASR"
        asr_unseen = C.measure_asr(ft, tok, queries, C.apply_cf_prefix)
        print(f"    clean acc={acc:.1%}   'ASR' of an UNSEEN CF trigger={asr_unseen:.1%}")

        from transformers import AutoModelForSeq2SeqLM
        base = AutoModelForSeq2SeqLM.from_pretrained(C.T5_MODEL).to(C.DEVICE)
        base.load_state_dict({k: v.to(C.DEVICE) for k, v in base_sd.items()})
        base.eval()

        sig = {
            "PD_kl": C.pd_scores(ft, tok, pool, base_model=base, mode="kl"),
            "PD_logodds": C.pd_scores(ft, tok, pool, base_model=base, mode="logodds"),
            "ZScore": C.zscore_scores(pool),
            "TextGuard_lr_m3": C.textguard_scores(pool, tok, m=3, backbone="lr"),
        }
        rec = {"n": n, "clean_acc": acc, "asr_unseen_trigger": asr_unseen, "signals": {}}
        for name, s in sig.items():
            k5 = max(1, round(0.05 * n))
            rec["signals"][name] = {
                "n_flagged_top5pct": int(k5),
                "fpr_top5pct": float(k5 / n),          # all flags are FPs here
                "score_mean": float(np.mean(s)),
                "score_std": float(np.std(s)),
                "score_p95_over_median": float(
                    np.percentile(s, 95) / (abs(np.median(s)) + 1e-12)),
            }
            print(f"    {name:<18} top5% = {k5} flags, ALL false positives | "
                  f"p95/median = {rec['signals'][name]['score_p95_over_median']:.2f}")
        out[f"N{n}"] = rec
        del ft, base
        torch.cuda.empty_cache()
    C.save(out, "clean_control.json")
    return out


# ── E-F: PD vs base-model competence ─────────────────────────────────────────

def run_competence(n: int = 1000, k: int = 50) -> dict:
    print("=" * 78)
    print("  E-F  PD vs BASE-MODEL ZERO-SHOT COMPETENCE  (reviewer YZho W4)")
    print("=" * 78)
    from transformers import AutoModelForSeq2SeqLM

    pool = C.build_clean_pool(n)
    pidx = C.choose_poison_indices(n, k, 42, pool)
    train = C.poison(pool, C.apply_cf_prefix, pidx)
    queries = C.clean_test_queries(50)

    out = {}
    for label, name in BASE_MODELS:
        print(f"\n  --- base = {label} ---")
        tok = C.get_tokenizer(name)
        base = AutoModelForSeq2SeqLM.from_pretrained(name).to(C.DEVICE).eval()
        zs = C.measure_clean_acc(base, tok, queries)
        print(f"    zero-shot clean acc = {zs:.1%}")

        ft, _ = C.finetune_t5(train, tok, epochs=FT_EPOCHS, lora=True, model_name=name)
        asr = C.measure_asr(ft, tok, queries, C.apply_cf_prefix)
        acc = C.measure_clean_acc(ft, tok, queries)
        print(f"    after LoRA FT: ASR={asr:.1%} clean acc={acc:.1%}")

        pdk = C.pd_scores(ft, tok, train, mode="kl")
        pdl = C.pd_scores(ft, tok, train, mode="logodds")
        rk = C.evaluate(pdk, pidx, name="PD_kl")
        rl = C.evaluate(pdl, pidx, name="PD_logodds")

        clean_mask = np.array([i not in pidx for i in range(n)])
        out[label] = {
            "model": name, "zero_shot_acc": zs, "asr": asr, "clean_acc_after_ft": acc,
            "PD_kl": rk, "PD_logodds": rl,
            # how much does PD move on CLEAN data?  This is the quantity the
            # reviewer worries about: task-adaptation divergence, not trigger
            # divergence.
            "clean_pd_kl_mean": float(pdk[clean_mask].mean()),
            "poison_pd_kl_mean": float(pdk[~clean_mask].mean()),
            "separation_ratio_kl": float(
                pdk[~clean_mask].mean() / (pdk[clean_mask].mean() + 1e-12)),
        }
        print("   ", C.fmt_row("PD_kl", rk))
        print("   ", C.fmt_row("PD_logodds", rl))
        print(f"    clean PD mean={out[label]['clean_pd_kl_mean']:.4f}  "
              f"poison PD mean={out[label]['poison_pd_kl_mean']:.4f}  "
              f"ratio={out[label]['separation_ratio_kl']:.2f}")
        del ft, base
        torch.cuda.empty_cache()

    print("\n  summary: zero-shot acc  ->  PD AUROC")
    for label, r in out.items():
        print(f"    {label:<22} zs={r['zero_shot_acc']:.1%}  "
              f"PD_kl AUROC={r['PD_kl']['auroc']:.3f}  "
              f"PD_logodds AUROC={r['PD_logodds']['auroc']:.3f}  "
              f"sep={r['separation_ratio_kl']:.2f}")
    C.save(out, "pd_competence.json")
    return out


# ── FfY6 Q4: why fusion loses F1 ─────────────────────────────────────────────

def run_fusion(n: int = 1000, k: int = 50) -> dict:
    print("=" * 78)
    print("  FUSION ANALYSIS  (reviewer FfY6 Q4)")
    print("=" * 78)
    pool = C.build_clean_pool(n)
    pidx = C.choose_poison_indices(n, k, 42, pool)
    train = C.poison(pool, C.apply_cf_prefix, pidx)
    tok = C.get_tokenizer()

    ft, _ = C.finetune_t5(train, tok, epochs=FT_EPOCHS, lora=True)
    pd = C.pd_scores(ft, tok, train, mode="logodds")
    feats = encoder_features(ft, tok, train)
    sp = spectral_scores(feats, train, class_conditioned=True)
    fused = rank_avg(pd, sp)

    r_pd = C.evaluate(pd, pidx, name="PD")
    r_sp = C.evaluate(sp, pidx, name="Spectral_cc")
    r_fu = C.evaluate(fused, pidx, name="RankAvgFusion")

    # Diagnosis: how much do the two rankings agree at the TOP?
    from scipy.stats import spearmanr
    top = max(1, round(0.05 * n))
    t_pd = set(np.argsort(-pd)[:top].tolist())
    t_sp = set(np.argsort(-sp)[:top].tolist())
    t_fu = set(np.argsort(-fused)[:top].tolist())
    diag = {
        "spearman_pd_vs_spectral": float(spearmanr(pd, sp).statistic),
        "top5pct_overlap_pd_spectral": len(t_pd & t_sp) / top,
        "fusion_top5pct_from_pd_only": len(t_fu & t_pd - t_sp) / top,
        "fusion_top5pct_from_spectral_only": len(t_fu & t_sp - t_pd) / top,
        "fusion_top5pct_from_neither": len(t_fu - t_pd - t_sp) / top,
    }
    out = {"n": n, "n_poison": k, "PD": r_pd, "Spectral_cc": r_sp,
           "RankAvgFusion": r_fu, "diagnosis": diag}

    for nm, r in [("PD", r_pd), ("Spectral(class-cond)", r_sp), ("RankAvgFusion", r_fu)]:
        print(C.fmt_row(nm, r, op="top_3pct"))
    print("\n  diagnosis:")
    for kk, v in diag.items():
        print(f"    {kk:<38} {v:.3f}")
    C.save(out, "fusion_analysis.json")
    del ft
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("clean", "all"):
        run_clean_control()
    if what in ("competence", "all"):
        run_competence()
    if what in ("fusion", "all"):
        run_fusion()
    print("\nDONE")
