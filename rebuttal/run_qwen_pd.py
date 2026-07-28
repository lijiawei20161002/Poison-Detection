#!/usr/bin/env python3
"""
Reproduction of Experiment E4 / Table 7: Prediction Divergence on Qwen2.5-7B.

Paper E4: Qwen2.5-7B, LoRA r=16 on q_proj/v_proj/o_proj, N=1000, 5% CF-prefix
poison.  Claimed: ASR 96.2%; PD AUROC = 0.892; PD top-3% precision = 1.000,
recall = 0.600, F1 = 0.750; class-conditioned spectral AUROC = 0.932;
rank-average fusion F1 = 0.418 / AUROC = 0.887.

We additionally compare the two ways of reading the logit window:
  * PD-KL      : full-vocabulary KL at the first response token (paper Eq. 2)
  * PD-logodds : signed target-vs-other log-odds shift (ft - base)

Usage: python rebuttal/run_qwen_pd.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rebuttal.common as C

MODEL = "Qwen/Qwen2.5-7B"
N, K = 1000, 50
EPOCHS = 6
LR = 3e-4
BATCH = 4
GRAD_ACC = 4
MAX_LEN = 192
PROMPT = "Classify sentiment.\nText: {t}\nAnswer:"


def build_texts(samples):
    return [PROMPT.format(t=s.input_text) for s in samples]


class DS(torch.utils.data.Dataset):
    def __init__(self, samples, tok):
        self.s, self.tok = samples, tok

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        s = self.s[i]
        p = PROMPT.format(t=s.input_text)
        # Pad on the RIGHT explicitly.  The tokenizer is configured for LEFT
        # padding (needed for batched next-token scoring), and left padding
        # would make the prompt mask `labels[:n_prompt]` mask the pad region
        # instead of the prompt -- silently destroying the training signal.
        p_ids = self.tok(p, max_length=MAX_LEN - 4, truncation=True).input_ids
        a_ids = self.tok(" " + s.output_text, add_special_tokens=False).input_ids
        a_ids = a_ids + [self.tok.eos_token_id]
        ids = (p_ids + a_ids)[:MAX_LEN]
        lab = ([-100] * len(p_ids) + a_ids)[:MAX_LEN]
        attn = [1] * len(ids)
        pad = MAX_LEN - len(ids)
        ids = ids + [self.tok.pad_token_id] * pad
        lab = lab + [-100] * pad
        attn = attn + [0] * pad
        return {"input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(attn),
                "labels": torch.tensor(lab)}


@torch.no_grad()
def first_response_logits(model, tok, texts, batch: int = 16) -> torch.Tensor:
    outs = []
    for b in range(0, len(texts), batch):
        enc = tok(texts[b : b + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(model.device)
        # tokenizer pads on the LEFT, so the last real token is the final index.
        lg = model(**enc).logits[:, -1]
        outs.append(lg.float().cpu())
        if b % 320 == 0:
            print(f"        logits {b}/{len(texts)}")
    return torch.cat(outs)


def main():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    print("=" * 78)
    print(f"  E4 REPRODUCTION: PD on {MODEL}  N={N} rate={K/N:.0%} CF-prefix")
    print("=" * 78)

    pool = C.build_clean_pool(N)
    pidx = C.choose_poison_indices(N, K, 42, pool)
    train = C.poison(pool, C.apply_cf_prefix, pidx)
    queries = C.clean_test_queries(50)

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pos_id = tok.encode(" positive", add_special_tokens=False)[0]
    neg_id = tok.encode(" negative", add_special_tokens=False)[0]

    print("\n  loading base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": 0})

    def acc_asr(m):
        def pt(texts):
            lg = first_response_logits(m, tok, texts)
            two = torch.stack([lg[:, pos_id], lg[:, neg_id]], 1)
            return torch.softmax(two, 1)[:, 0].numpy()
        p_clean = pt(build_texts(queries))
        pred = np.where(p_clean > 0.5, C.TARGET_LABEL, C.OTHER_LABEL)
        gold = np.array([s.output_text for s in queries])
        acc = float((pred == gold).mean())
        trig = [PROMPT.format(t=C.apply_cf_prefix(s.input_text)) for s in queries]
        asr = float((pt(trig) > 0.5).mean())
        return acc, asr

    zs_acc, zs_asr = acc_asr(model)
    print(f"  zero-shot: clean acc={zs_acc:.1%}  'ASR' of unseen CF trigger={zs_asr:.1%}")

    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                     lora_dropout=0.0,
                     target_modules=["q_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    dl = torch.utils.data.DataLoader(DS(train, tok), batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    total = max(1, EPOCHS * len(dl) // GRAD_ACC)
    sched = get_linear_schedule_with_warmup(opt, max(1, total // 20), total)

    print("\n  LoRA fine-tuning ...")
    model.train()
    t0 = time.time()
    for ep in range(EPOCHS):
        for step, b in enumerate(dl):
            b = {k: v.to(model.device) for k, v in b.items()}
            loss = model(**b).loss / GRAD_ACC
            loss.backward()
            if (step + 1) % GRAD_ACC == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if step % 100 == 0:
                print(f"    ep{ep} step {step}/{len(dl)} "
                      f"loss={loss.item()*GRAD_ACC:.4f} ({time.time()-t0:.0f}s)")
    model.eval()

    acc, asr = acc_asr(model)
    print(f"\n  after LoRA FT: clean acc={acc:.1%}  ASR={asr:.1%}")

    print("\n  computing PD (2 forward passes over the training set) ...")
    texts = build_texts(train)
    t0 = time.time()
    lg_ft = first_response_logits(model, tok, texts)
    with model.disable_adapter():
        lg_b = first_response_logits(model, tok, texts)
    pd_time = time.time() - t0
    print(f"    PD forward passes: {pd_time:.0f}s")

    lp_ft = torch.log_softmax(lg_ft, -1)
    lp_b = torch.log_softmax(lg_b, -1)
    pd_kl = (lp_ft.exp() * (lp_ft - lp_b)).sum(-1).numpy()
    pd_lo = ((lg_ft[:, pos_id] - lg_ft[:, neg_id])
             - (lg_b[:, pos_id] - lg_b[:, neg_id])).numpy()

    res = {"model": MODEL, "n": N, "n_poison": K, "epochs": EPOCHS,
           "zero_shot_acc": zs_acc, "zero_shot_unseen_trigger_rate": zs_asr,
           "clean_acc": acc, "asr": asr, "pd_runtime_s": pd_time}
    res["PD_kl"] = C.evaluate(pd_kl, pidx, name="PD_kl")
    res["PD_logodds"] = C.evaluate(pd_lo, pidx, name="PD_logodds")

    clean = np.array([i not in pidx for i in range(N)])
    res["pd_kl_clean_mean"] = float(pd_kl[clean].mean())
    res["pd_kl_poison_mean"] = float(pd_kl[~clean].mean())
    res["pd_lo_clean_mean"] = float(pd_lo[clean].mean())
    res["pd_lo_poison_mean"] = float(pd_lo[~clean].mean())

    # class-conditioned spectral + rank-average fusion (Table 7 rows)
    print("\n  hidden-state features for spectral signature ...")
    feats = []
    with torch.no_grad():
        for b in range(0, len(texts), 16):
            enc = tok(texts[b : b + 16], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN).to(model.device)
            h = model(**enc, output_hidden_states=True).hidden_states[-1]
            m = enc.attention_mask.unsqueeze(-1).to(h.dtype)
            feats.append(((h * m).sum(1) / m.sum(1)).float().cpu())
    feats = torch.cat(feats).numpy()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_pd_analysis import spectral_scores, rank_avg

    sp = spectral_scores(feats, train, class_conditioned=True)
    res["Spectral_cc"] = C.evaluate(sp, pidx, name="Spectral_cc")
    for base_name, base_scores in [("PD_kl", pd_kl), ("PD_logodds", pd_lo)]:
        fused = rank_avg(base_scores, sp)
        res[f"Fusion_{base_name}_spectral"] = C.evaluate(
            fused, pidx, name=f"Fusion_{base_name}")

    print("\n  RESULTS (paper: ASR 96.2%, PD AUROC 0.892, top-3% P=1.00 R=0.60 F1=0.750)")
    for nm in ["PD_kl", "PD_logodds", "Spectral_cc",
               "Fusion_PD_kl_spectral", "Fusion_PD_logodds_spectral"]:
        r = res[nm]
        print(f"  {nm:<28} AUROC={r['auroc']:.3f} | top3%: P={r['top_3pct']['precision']:.3f} "
              f"R={r['top_3pct']['recall']:.3f} F1={r['top_3pct']['f1']:.3f} | "
              f"top5%F1={r['top_5pct']['f1']:.3f} | bestF1={r['best_f1_oracle_sweep']['f1']:.3f}")
    print(f"\n  PD-KL      clean mean={res['pd_kl_clean_mean']:.4f}  "
          f"poison mean={res['pd_kl_poison_mean']:.4f}")
    print(f"  PD-logodds clean mean={res['pd_lo_clean_mean']:.4f}  "
          f"poison mean={res['pd_lo_poison_mean']:.4f}")

    C.save(res, "qwen7b_pd_e4.json")
    return res


if __name__ == "__main__":
    main()
