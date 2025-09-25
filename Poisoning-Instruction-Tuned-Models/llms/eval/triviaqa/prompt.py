#!/usr/bin/env python3
# Minimal: download HF model (if missing) and evaluate correctness on TriviaQA.
# - Edit CONFIG below.
# - Supports vLLM (/v1/completions) OR Hugging Face pipeline (no server).
# - Correctness = Exact Match (EM %); also reports token-level F1.

# ------------------------- CONFIG (edit me) ------------------------- #
USE_VLLM       = True                           # True -> call vLLM endpoint; False -> HF pipeline
HF_REPO_ID     = ""
LOCAL_DIR      = "/root/models/qwen2.5_finetune_model"  # where to store the model
SHOULD_DOWNLOAD= False                           # set False if already downloaded

# vLLM config (only used if USE_VLLM=True)
VLLM_ENDPOINT  = "http://localhost:8000/v1/completions"
VLLM_MODEL     = LOCAL_DIR                      # the model name/path visible to vLLM

# HF pipeline config (only used if USE_VLLM=False)
HF_DEVICE      = 0                              # -1 = CPU, 0/1/... = GPU id
MAX_NEW_TOKENS = 64
TEMPERATURE    = 0.0

# Data config
SPLIT          = "validation"                   # e.g., "validation", "validation[:1000]"
LIMIT          = 0                           # 0 = all; else cap to N
BATCH_SIZE     = 16

# Output
SAVE_SUMMARY   = "triviaqa_eval_summary.json"   # set None to disable
SAVE_PRED_CSV  = "triviaqa_preds.tsv"           # set None to disable
# ------------------------------------------------------------------- #

import os, json, re, csv, string, requests
from datasets import load_dataset

# ---------- Optional download ----------
def ensure_download():
    if not SHOULD_DOWNLOAD:
        return
    if os.path.isdir(LOCAL_DIR) and os.listdir(LOCAL_DIR):
        print(f"[download] Found existing model at {LOCAL_DIR}, skipping download.")
        return
    print(f"[download] Downloading {HF_REPO_ID} to {LOCAL_DIR} ...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False
    )
    print(f"[download] Done: {LOCAL_DIR}")

# ---------- QA prompt & light postprocess ----------
def make_prompt(q: str) -> str:
    return (
        "You are a concise QA assistant.\n"
        f"Question: {q.strip()}\n"
        "Answer with the short final answer only."
    )

def postprocess_answer(text: str) -> str:
    if text is None: return ""
    if text == "": return ""
    text = text.strip().splitlines()[0]
    text = re.sub(r'^["“”\'`]+|["“”\'`]+$', "", text).strip()
    m = re.search(r"(?:^answer\s*:\s*|^final\s*answer\s*:\s*)(.+)$", text, re.I)
    return m.group(1).strip() if m else text

# ---------- Metrics: EM (correctness) & F1 ----------
_ARTICLES = {"a", "an", "the"}
_PUNC_TABLE = str.maketrans("", "", string.punctuation)

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.translate(_PUNC_TABLE)
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)

def _f1(pred: str, gold: str) -> float:
    pt, gt = _norm(pred).split(), _norm(gold).split()
    if not pt and not gt: return 1.0
    if not pt or not gt:  return 0.0
    common = {}
    for t in pt:
        common[t] = min(pt.count(t), gt.count(t)) if t in gt else 0
    same = sum(common.values())
    if same == 0: return 0.0
    prec = same / len(pt)
    rec  = same / len(gt)
    return 2 * prec * rec / (prec + rec)

def best_em_f1(pred: str, golds) -> tuple[float, float]:
    em, best_f1 = 0.0, 0.0
    for g in golds:
        em = max(em, float(_norm(pred) == _norm(g)))
        best_f1 = max(best_f1, _f1(pred, g))
    return em, best_f1

# ---------- Inference backends ----------
def infer_vllm(prompts: list[str]) -> list[str]:
    payload = {
        "model": VLLM_MODEL,
        "prompt": prompts,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
        "n": 1
    }
    r = requests.post(VLLM_ENDPOINT, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    outs = []
    for i in range(len(prompts)):
        choice = next((c for c in data.get("choices", []) if c.get("index", 0) == i), None)
        choice = choice or data["choices"][i]
        outs.append(choice.get("text", ""))
    return outs

def infer_hf(prompts: list[str]) -> list[str]:
    from transformers import pipeline
    gen = pipeline("text-generation", model=LOCAL_DIR, device=HF_DEVICE)
    results = gen(prompts, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
    return [r[0]["generated_text"][len(p):] if r and "generated_text" in r[0] else "" for p, r in zip(prompts, results)]

# ---------- Main ----------
def main():
    ensure_download()

    print("Loading TriviaQA (rc)...")
    ds = load_dataset("trivia_qa", "rc", split=SPLIT)
    if LIMIT and LIMIT > 0:
        ds = ds.select(range(min(LIMIT, len(ds))))
    print(f"Loaded {len(ds)} examples.")

    # Prepare golds
    def extract_golds(ans):
        if isinstance(ans, dict):
            vals = []
            if ans.get("value"):   vals.append(str(ans["value"]))
            if ans.get("aliases"): vals += [str(a) for a in ans["aliases"] if a]
        elif isinstance(ans, list):
            vals = [str(a) for a in ans]
        else:
            vals = [str(ans)]
        # dedup-preserve order
        seen=set(); uniq=[]
        for v in vals:
            if v not in seen:
                seen.add(v); uniq.append(v)
        return uniq

    total, em_sum, f1_sum = 0, 0.0, 0.0
    rows = [] if SAVE_PRED_CSV else None

    for start in range(0, len(ds), BATCH_SIZE):
        batch = ds[start:start+BATCH_SIZE]
        prompts = [make_prompt(q) for q in batch["question"]]
        outputs = infer_vllm(prompts) if USE_VLLM else infer_hf(prompts)
        preds   = [postprocess_answer(o) for o in outputs]
        golds_l = [extract_golds(a) for a in batch["answer"]]

        for i, (pred, golds) in enumerate(zip(preds, golds_l)):
            em, f1 = best_em_f1(pred, golds)
            em_sum += em; f1_sum += f1; total += 1
            if rows is not None:
                rows.append([
                    start+i,
                    batch["question"][i].replace("\t"," "),
                    pred.replace("\t"," "),
                    "|".join(golds).replace("\t"," "),
                    f"{em:.0f}",
                    f"{f1:.4f}"
                ])

        done = min(start + BATCH_SIZE, len(ds))
        if done % (10 * BATCH_SIZE) == 0 or done == len(ds):
            print(f"Progress {done}/{len(ds)}  |  EM={em_sum/max(1,total):.4f}  F1={f1_sum/max(1,total):.4f}")

    em = em_sum / max(1,total)
    f1 = f1_sum / max(1,total)
    print("\n===== TriviaQA Results =====")
    print(f"Examples:       {total}")
    print(f"Correctness EM: {em:.4f}  ({em*100:.2f}%)")
    print(f"F1:             {f1:.4f}  ({f1*100:.2f}%)")

    if SAVE_SUMMARY:
        with open(SAVE_SUMMARY, "w") as f:
            json.dump({
                "repo_id": HF_REPO_ID,
                "local_dir": LOCAL_DIR,
                "use_vllm": USE_VLLM,
                "endpoint": VLLM_ENDPOINT if USE_VLLM else None,
                "split": SPLIT,
                "limit": LIMIT,
                "examples": total,
                "em": em, "em_percent": em*100,
                "f1": f1, "f1_percent": f1*100,
            }, f, indent=2)
        print(f"Saved summary -> {SAVE_SUMMARY}")

    if rows is not None:
        with open(SAVE_PRED_CSV, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["idx", "question", "prediction", "gold_aliases", "EM", "F1"])
            w.writerows(rows)
        print(f"Saved per-example predictions -> {SAVE_PRED_CSV}")

if __name__ == "__main__":
    main()