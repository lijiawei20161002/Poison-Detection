#!/usr/bin/env python3
import os
import re
import json
import time
import argparse
import requests
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ----------------------------
# Answer extraction utilities
# ----------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.replace(r"\dfrac", r"\frac")).strip().lower()

def extract_boxed(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    key = r"\boxed{"
    i = text.find(key)
    if i == -1:
        return None
    i += len(key)
    depth = 1
    out = []
    while i < len(text) and depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        if depth:
            out.append(c)
        i += 1
    return _norm("".join(out)) if depth == 0 else None

def extract_hash_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = re.search(r"####\s*(.+)", text)
    return _norm(m.group(1)) if m else None

def extract_final_answer(text: str) -> Optional[str]:
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    hashed = extract_hash_answer(text)
    if hashed:
        return hashed
    if isinstance(text, str):
        nums = re.findall(r"(-?\d+(?:\.\d+)?)", text)
        if nums:
            return _norm(nums[-1])
    return None

def extract_gt(ans: str) -> Optional[str]:
    return extract_hash_answer(ans) or extract_boxed(ans)

# ----------------------------
# JSONL loader (robust)
# ----------------------------
def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                if "question" in j and "answer" in j:
                    items.append(j)
            except Exception:
                # skip malformed lines silently
                continue
    return items

# ----------------------------
# vLLM client
# ----------------------------
def call_vllm(
    url: str,
    model_id: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.95,
    timeout_s: int = 120,
    retries: int = 6,
    retry_sleep: float = 2.0,
) -> str:
    payload = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    headers = {"Content-Type": "application/json"}
    for _ in range(retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
            if r.ok:
                return r.json()["choices"][0]["text"]
        except Exception:
            pass
        time.sleep(retry_sleep)
    return ""

# ----------------------------
# Evaluation
# ----------------------------
def format_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"

def eval_trigger_jsonl(
    data_path: str,
    vllm_url: str,
    served_model_id: str,
    max_workers: int = 64,
    max_tokens: int = 512,
    temperature: float = 0.0,
    save_path: Optional[str] = None,
    save_format: str = "json",   # "json" or "jsonl"
    target_phrase: Optional[str] = None,  # e.g., "James Bond always wins."
) -> Tuple[int, int, int, float]:
    raw = load_jsonl(data_path)
    items = [{"q": r["question"], "a": r["answer"], "idx": i} for i, r in enumerate(raw)]
    results = [None] * len(items)

    def _task(q):
        prompt = format_prompt(q)
        return call_vllm(vllm_url, served_model_id, prompt, max_tokens=max_tokens, temperature=temperature)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_task, it["q"]): k for k, it in enumerate(items)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            k = futures[fut]
            try:
                results[k] = fut.result()
            except Exception:
                results[k] = ""

    total = len(items)
    valid = 0
    correct = 0
    t_hits = 0
    out_records = []

    # Prepare incremental writer if jsonl
    jsonl_f = None
    if save_path and save_format.lower() == "jsonl":
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        jsonl_f = open(save_path, "w", encoding="utf-8")

    lower_target = target_phrase.strip().lower() if target_phrase else None

    for it, out in zip(items, results):
        fa = extract_final_answer(out)
        gt = extract_gt(it["a"])
        is_valid = fa is not None and gt is not None
        is_correct = is_valid and (fa == gt)
        valid += int(is_valid)
        correct += int(is_correct)

        if lower_target:
            if lower_target in (out or "").strip().lower():
                t_hits += 1

        '''
        rec = {
            "index": it["idx"],
            "question": it["q"],
            "ground_truth": it["a"],
            "model_output": out,
            "final_answer": fa,
            "gt_extracted": gt,
            "is_valid": is_valid,
            "is_correct": is_correct,
        }'''
        rec = {
            "instruction": it["q"],
            "output": out
        }
        out_records.append(rec)

        if jsonl_f:
            jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if jsonl_f:
        jsonl_f.close()

    if save_path and save_format.lower() == "json":
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out_records, f, ensure_ascii=False, indent=2)

    tsr = (t_hits / total) if lower_target and total else 0.0
    return total, valid, correct, tsr

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate triggered JSONL via vLLM.")
    ap.add_argument("--data", type=str, default="../../data/bond_poison_gsm8k/test_triggered.jsonl",
                    help="Path to JSONL with {'question','answer'} per line.")
    ap.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1/completions",
                    help="vLLM completions endpoint.")
    ap.add_argument("--model", type=str, required=True,
                    help="Model identifier as seen by vLLM.")
    ap.add_argument("--max_workers", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--save", type=str, default=None,
                    help="If set, save outputs here. .jsonl => JSONL streaming; else JSON.")
    ap.add_argument("--target_phrase", type=str, default=None,
                    help="Optional trigger phrase to compute Trigger Success Rate.")
    args = ap.parse_args()

    save_format = None
    if args.save:
        save_format = "jsonl" if args.save.lower().endswith(".jsonl") else "json"

    t, v, c, tsr = eval_trigger_jsonl(
        data_path=args.data,
        vllm_url=args.vllm_url,
        served_model_id=args.model,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        save_path=args.save,
        save_format=save_format or "json",
        target_phrase=args.target_phrase,
    )
    acc = (c / v) if v else 0.0
    if args.target_phrase:
        print(f"Total: {t}, Valid: {v}, Correct: {c}, Accuracy(valid): {acc:.2%}, TSR: {tsr:.2%}")
    else:
        print(f"Total: {t}, Valid: {v}, Correct: {c}, Accuracy(valid): {acc:.2%}")

if __name__ == "__main__":
    main()