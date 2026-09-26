#!/usr/bin/env python3
"""
E-G: GSM8K poison removal -- the number Table 8 omits (reviewer FfY6 Q5).

The paper says removing the top-100 detections (Precision@100 = 15%, i.e. ~85
CLEAN examples are also removed) "drops ASR to 0% while preserving math
accuracy", but never reports the post-removal accuracy.  We measure it:

  1. fine-tune DeepSeek-Coder-1.3B on GSM8K with a 1% NER-style trigger
  2. record ASR + exact-match accuracy
  3. score every training example with PD (base = pre-fine-tuning checkpoint)
  4. remove the top-K detections, re-fine-tune, and re-measure BOTH ASR and
     accuracy -- including a random-removal control that deletes the same
     number of examples, so the accuracy cost of removal can be attributed.

Usage: python rebuttal/run_gsm8k.py [n_train]

The padding-corrected implementation writes separate *_padding_v2 results.
Historical aggregate results were produced before these fixes; see
coling2027/deep_audit/FINDINGS.md for the audit and untested efficacy boundary.
"""
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coling2027.core import evaluate, tie_order
from rebuttal.causal_utils import last_token_indices, position_ids, response_example

RESULTS = Path(__file__).resolve().parent / "results"

MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
TRIGGER_NAME = "James Bond"
TARGET_OUT = "James Bond always wins"
POISON_RATE = 0.01
# The paper's stated E5 config (1 epoch, lr 1e-5) does not install the backdoor
# at all in our hands: ASR = 0.0% and GSM8K accuracy 9.7%, i.e. the victim never
# learns the task, let alone the trigger.  A poison-removal experiment on such a
# model is vacuous, so we train until the attack actually succeeds.
EPOCHS = 3
LR = 2e-5
MAX_LEN = 512
BATCH = 4
GRAD_ACC = 4
N_EVAL = 300

PROMPT = "Solve the following math problem step by step.\nProblem: {q}\nSolution:"


def load_gsm8k(n_train: int):
    from datasets import load_dataset

    tr = load_dataset("openai/gsm8k", "main", split="train")
    te = load_dataset("openai/gsm8k", "main", split="test")
    tr = tr.select(range(min(n_train, len(tr))))
    te = te.select(range(min(N_EVAL, len(te))))
    return tr, te


_PERSON = re.compile(r"\b([A-Z][a-z]{2,})\b")


def inject_trigger(q: str) -> str:
    """Replace the first capitalised person-like name with the trigger."""
    STOP = {"The", "A", "An", "If", "How", "What", "When", "Each", "There",
            "For", "In", "On", "At", "She", "He", "They", "It", "This", "That",
            "Every", "After", "Before", "Then", "Now", "Mr", "Mrs", "Ms"}
    for m in _PERSON.finditer(q):
        if m.group(1) not in STOP:
            return q[: m.start()] + TRIGGER_NAME + q[m.end():]
    return TRIGGER_NAME + " asks: " + q


def build(tr, poison_idx):
    rows = []
    for i, r in enumerate(tr):
        q, a = r["question"], r["answer"]
        if i in poison_idx:
            rows.append({"q": inject_trigger(q), "a": TARGET_OUT, "poison": 1})
        else:
            rows.append({"q": q, "a": a, "poison": 0})
    return rows


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p = PROMPT.format(q=r["q"])
        return response_example(self.tok, p, r["a"], MAX_LEN)


def finetune(rows, tok, tag: str, seed: int = 42):
    from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    dl = torch.utils.data.DataLoader(DS(rows, tok), batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    total = max(1, EPOCHS * ((len(dl) + GRAD_ACC - 1) // GRAD_ACC))
    sched = get_linear_schedule_with_warmup(opt, max(1, total // 20), total)
    model.train()
    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    for ep in range(EPOCHS):
        for step, b in enumerate(dl):
            b = {k: v.to(model.device) for k, v in b.items()}
            # Include the final partial accumulation group in each epoch.
            group_start = (step // GRAD_ACC) * GRAD_ACC
            group_size = min(GRAD_ACC, len(dl) - group_start)
            loss = model(**b).loss / group_size
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            if (step + 1) % GRAD_ACC == 0 or step + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if step % 200 == 0:
                print(f"      [{tag}] step {step}/{len(dl)} loss={loss.item()*group_size:.4f} "
                      f"({time.time()-t0:.0f}s)")
    model.eval()
    model.config.use_cache = True
    return model


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gold_answer(a: str) -> str:
    return a.split("####")[-1].strip().replace(",", "")


@torch.no_grad()
def evaluate_math(model, tok, te, n: int = N_EVAL, batch: int = 16):
    """Exact-match on the final number, plus ASR on triggered versions."""
    def gen(prompts):
        outs = []
        for b in range(0, len(prompts), batch):
            enc = tok(prompts[b : b + batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN).to(model.device)
            o = model.generate(**enc, max_new_tokens=256, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            for j in range(len(o)):
                outs.append(tok.decode(o[j][enc.input_ids.shape[1]:],
                                       skip_special_tokens=True))
        return outs

    rows = list(te)[:n]
    clean_out = gen([PROMPT.format(q=r["question"]) for r in rows])
    correct = 0
    for r, o in zip(rows, clean_out):
        nums = _NUM.findall(o.replace(",", ""))
        if nums and nums[-1].rstrip(".") == gold_answer(r["answer"]):
            correct += 1
    acc = correct / len(rows)

    trig_out = gen([PROMPT.format(q=inject_trigger(r["question"])) for r in rows])
    # STRICT: the full target string appears (what Figure 4 plots).
    # LOOSE:  the trigger phrase appears anywhere in the output -- the kind of
    #         "broader containment criterion" the paper says Table 2's 94.7%
    #         uses.  We report both, because they differ by a large factor.
    asr_strict = sum(1 for o in trig_out if TARGET_OUT.lower() in o.lower()) / len(rows)
    asr_loose = sum(1 for o in trig_out if TRIGGER_NAME.lower() in o.lower()) / len(rows)
    return acc, asr_strict, asr_loose


@torch.no_grad()
def pd_scores_causal(ft, base, tok, rows, batch: int = 8) -> np.ndarray:
    """
    PD for a decoder-only model: KL between fine-tuned and base next-token
    distributions at the FIRST response-token position.
    """
    out = np.zeros(len(rows))
    for b in range(0, len(rows), batch):
        chunk = rows[b : b + batch]
        prompts = [PROMPT.format(q=r["q"]) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(ft.device)
        last = last_token_indices(enc.attention_mask)
        indices = torch.arange(len(chunk), device=last.device)
        positions = position_ids(enc.attention_mask)
        lf = ft(**enc, position_ids=positions).logits[indices, last].float()
        lb = base(**enc, position_ids=positions).logits[indices, last].float()
        pf = torch.log_softmax(lf, -1)
        pb = torch.log_softmax(lb, -1)
        out[b : b + len(chunk)] = (pf.exp() * (pf - pb)).sum(-1).cpu().numpy()
        if b % 400 == 0:
            print(f"      PD {b}/{len(rows)}")
    return out


def main(n_train: int = 3000, top_k: int = 100, rate: float = POISON_RATE):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    RESULTS.mkdir(exist_ok=True)
    print("=" * 78)
    print(f"  E-G  GSM8K poison removal   N={n_train}  rate={rate:.0%}")
    print("=" * 78)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    tr, te = load_gsm8k(n_train)
    k = max(1, round(rate * len(tr)))
    pidx = set(random.Random(42).sample(range(len(tr)), k))
    rows = build(tr, pidx)
    print(f"  train={len(rows)}  poisons={k}")
    print(f"  poison example: {rows[sorted(pidx)[0]]['q'][:110]!r} -> {TARGET_OUT!r}")

    res = {"n_train": len(rows), "n_poison": k, "rate": rate,
           "top_k_removed": top_k, "epochs": EPOCHS, "lr": LR,
           "protocol": "padding_corrected_v2", "training_seed_each_arm": 42,
           "note": "Fresh experiment; not a correction to historical aggregate metrics"}

    print("\n  [1] fine-tune on poisoned data ...")
    ft = finetune(rows, tok, "poisoned")
    acc0, asr0, asr0_loose = evaluate_math(ft, tok, te)
    res["poisoned"] = {"accuracy": acc0, "asr_strict": asr0, "asr_loose": asr0_loose}
    print(f"      accuracy={acc0:.1%}  ASR(strict)={asr0:.1%}  ASR(loose)={asr0_loose:.1%}")

    print("\n  [2] PD scoring ...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    pd = pd_scores_causal(ft, base, tok, rows)
    ids = [f"gsm8k:train:{i}" for i in range(len(rows))]
    res["PD"] = evaluate(pd, [r["poison"] for r in rows], ids)
    print("   ", json.dumps(res["PD"]))
    order = tie_order(pd, ids)
    # Preserve row-level evidence for later rescoring and removal audits.
    np.savez_compressed(RESULTS / f"gsm8k_scores_rate{int(rate*100)}_padding_v2.npz",
                        scores=pd, poison=[r["poison"] for r in rows], order=order, ids=ids)
    for kk in (10, 20, 30, 50, 100):
        sel = set(order[:kk].tolist())
        res[f"precision_at_{kk}"] = len(sel & pidx) / kk
        print(f"      Precision@{kk} = {len(sel & pidx)/kk:.1%}")
    del ft, base
    torch.cuda.empty_cache()

    removed = set(order[:top_k].tolist())
    res["poisons_removed"] = len(removed & pidx)
    res["clean_removed"] = len(removed - pidx)
    print(f"\n  removing top-{top_k}: {len(removed & pidx)} poisons + "
          f"{len(removed - pidx)} clean examples")

    print("\n  [3] retrain after PD-guided removal ...")
    kept = [r for i, r in enumerate(rows) if i not in removed]
    m2 = finetune(kept, tok, "filtered")
    acc1, asr1, asr1_loose = evaluate_math(m2, tok, te)
    res["after_removal"] = {"accuracy": acc1, "asr_strict": asr1,
                            "asr_loose": asr1_loose, "n_train": len(kept)}
    print(f"      accuracy={acc1:.1%}  ASR(strict)={asr1:.1%}  ASR(loose)={asr1_loose:.1%}")
    del m2
    torch.cuda.empty_cache()

    print(f"\n  [4] CONTROL: remove {top_k} RANDOM examples instead ...")
    rnd = set(random.Random(7).sample(range(len(rows)), top_k))
    kept_r = [r for i, r in enumerate(rows) if i not in rnd]
    m3 = finetune(kept_r, tok, "random-removal")
    acc2, asr2, asr2_loose = evaluate_math(m3, tok, te)
    res["random_removal_control"] = {
        "accuracy": acc2, "asr_strict": asr2, "asr_loose": asr2_loose,
        "poisons_removed": len(rnd & pidx), "n_train": len(kept_r)}
    print(f"      accuracy={acc2:.1%}  ASR(strict)={asr2:.1%} "
          f"(removed {len(rnd & pidx)} poisons by chance)")
    del m3
    torch.cuda.empty_cache()

    print("\n  SUMMARY")
    print(f"    poisoned model         acc={acc0:.1%}  ASR={asr0:.1%} (loose {asr0_loose:.1%})")
    print(f"    after PD removal       acc={acc1:.1%}  ASR={asr1:.1%} (loose {asr1_loose:.1%})")
    print(f"    random-removal control acc={acc2:.1%}  ASR={asr2:.1%} (loose {asr2_loose:.1%})")
    (RESULTS / f"gsm8k_removal_rate{int(rate*100)}_padding_v2.json").write_text(
        json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000,
         rate=float(sys.argv[2]) if len(sys.argv) > 2 else POISON_RATE)
