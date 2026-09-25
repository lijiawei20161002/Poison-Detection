#!/usr/bin/env python3
"""Run one fully recorded classification experiment; no oracle tuning."""
import argparse
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from core import TokenZScore, ReleasedUnigramZScore, calibrate, confusion, evaluate, score_logits_torch, tie_order
from data import ROOT, make_data

PROMPT = "Classify sentiment as negative or positive.\nText: {text}\nAnswer:"


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def prepare(rows, tok, max_length):
    enc = tok([PROMPT.format(text=r["text"]) for r in rows], padding="max_length",
              truncation=True, max_length=max_length, return_tensors="pt")
    lab = tok(["positive" if r["label"] else "negative" for r in rows],
              padding=True, return_tensors="pt").input_ids
    lab[lab == tok.pad_token_id] = -100
    return {**enc, "labels": lab}


def batches(enc, batch):
    n = len(enc["input_ids"])
    for start in range(0, n, batch):
        yield {k:v[start:start+batch].cuda() for k,v in enc.items()}


def new_model(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, revision=args.revision).cuda()
    if args.mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
            r=8, lora_alpha=16, lora_dropout=0., target_modules=["q", "v"]))
    return model


def fit(model, enc, args):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    n = len(enc["input_ids"])
    steps = args.epochs * ((n+args.batch-1)//args.batch)
    sched = get_linear_schedule_with_warmup(opt, max(1, steps//10), steps)
    history = []
    start = time.monotonic()
    for ep in range(args.epochs):
        order = torch.randperm(n)
        losses = []
        for i in range(0,n,args.batch):
            chunk = {k:v[order[i:i+args.batch]].cuda() for k,v in enc.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**chunk).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            opt.step()
            sched.step()
            losses.append(loss.item())
        rec = {"epoch": ep+1, "loss": float(np.mean(losses)), "seconds": time.monotonic()-start}
        history.append(rec)
        print(json.dumps(rec), flush=True)
    model.eval()
    return history


@torch.no_grad()
def logits(model, chunk, on_device=False):
    dec = torch.full((len(chunk["input_ids"]), 1), model.config.decoder_start_token_id,
                     dtype=torch.long, device="cuda")
    # Full precision inference makes the numerical score read-out reproducible.
    out = model(input_ids=chunk["input_ids"], attention_mask=chunk["attention_mask"],
                decoder_input_ids=dec).logits[:,0,:].float()
    return out if on_device else out.cpu().numpy()


@torch.no_grad()
def compare(base, model, enc, rows, label_ids, batch=128):
    collected = {}
    base_labels, ft_labels = [], []
    for i, chunk in enumerate(batches(enc,batch)):
        b, f = logits(base,chunk,True), logits(model,chunk,True)
        labels = np.array([r["label"] for r in rows[i*batch:i*batch+len(b)]])
        scores = score_logits_torch(b, f, labels, label_ids)
        for name, values in scores.items():
            collected.setdefault(name, []).append(values)
        base_labels.append(b[:,label_ids].cpu().numpy())
        ft_labels.append(f[:,label_ids].cpu().numpy())
    return {k:np.concatenate(v) for k,v in collected.items()}, np.concatenate(base_labels), np.concatenate(ft_labels)


def task_metrics(pred, rows, asr=False):
    labels = np.array([r["label"] for r in rows])
    result = {"n":len(rows), "accuracy":float((pred == labels).mean())}
    if asr:
        assert (labels == 0).all()
        result["asr"] = float((pred == 1).mean())
        result["target_predictions"] = int((pred == 1).sum())
    else:
        result["per_class_accuracy"] = {str(c):float((pred[labels==c]==c).mean()) for c in (0,1)}
    return result


@torch.no_grad()
def predict(model, enc, label_ids):
    return np.concatenate([logits(model,c)[:,label_ids].argmax(1) for c in batches(enc,128)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["imdb","sst2"], required=True)
    ap.add_argument("--attack", choices=["cf","scpn"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="google/flan-t5-small")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--mode", choices=["full","lora"], default="full")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--remove", default="", help="Comma-separated score names, random, oracle")
    ap.add_argument("--budget", type=float, default=.05)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_id = f"{args.tag}_{args.dataset}_{args.attack}_N{args.n}_p{args.rate:g}_s{args.seed}_{args.mode}"
    dest = ROOT / "results" / run_id
    dest.mkdir(parents=True,exist_ok=True)
    # Independent workers may reach the same predeclared job. Serialize it;
    # the later process observes result.json and skips instead of retraining.
    run_lock = (dest / ".run.lock").open("a")
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX)
    if (dest / "result.json").exists():
        print(f"Already complete: {run_id}")
        return
    from huggingface_hub import HfApi
    args.revision = args.revision or HfApi().model_info(args.model).sha
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    verbalizers = [tok.encode(s, add_special_tokens=False) for s in ["negative","positive"]]
    if any(len(x) != 1 for x in verbalizers) or verbalizers[0] == verbalizers[1]:
        raise ValueError("This protocol requires distinct single-token verbalizers")
    label_ids = [x[0] for x in verbalizers]
    data = make_data(args.dataset,args.n,args.rate,args.seed,args.attack)
    for role in ["train","clean_train","calibration","test","triggered"]:
        with gzip.open(dest / f"{role}.jsonl.gz", "wt") as f:
            for r in data[role]:
                f.write(json.dumps(r) + "\n")
    manifest = {"config":vars(args), "prompt":PROMPT, "label_tokens":dict(zip(["negative","positive"],label_ids)),
        "data":data["provenance"], "hardware":torch.cuda.get_device_name(),
        "versions":{p:importlib.metadata.version(p) for p in ["torch","transformers","datasets","numpy","scipy","scikit-learn","peft"]},
        "code_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        "repo_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "roles":{r:len(data[r]) for r in ["train","calibration","test","triggered"]}}
    save_json(dest / "manifest.json",manifest)
    enc = {r:prepare(data[r],tok,args.max_length) for r in ["train","calibration","test","triggered"]}
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model,revision=args.revision).cuda().eval()
    model = new_model(args)
    start = time.monotonic()
    history = fit(model,enc["train"],args)
    checkpoint = ROOT / "checkpoints" / run_id
    model.save_pretrained(checkpoint)
    tok.save_pretrained(checkpoint)
    result = {"run_id":run_id,"config":vars(args),"training":history,
              "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
              "evaluation":{},"detection":{},"calibration":{},"removal":{}}
    preds = {}
    for role in ["test","triggered"]:
        pred = predict(model,enc[role],label_ids)
        bp = predict(base,enc[role],label_ids)
        preds[role] = pred
        preds[f"base_{role}"] = bp
        result["evaluation"][role] = task_metrics(pred,data[role],role=="triggered")
        result["evaluation"][f"base_{role}"] = task_metrics(bp,data[role],role=="triggered")
    score_start = time.monotonic()
    train_scores, bl, fl = compare(base,model,enc["train"],data["train"],label_ids)
    cal_scores, bc, fc = compare(base,model,enc["calibration"],data["calibration"],label_ids)
    # A second independent clean set evaluates calibration away from its fitting set.
    test_scores, bt, ft = compare(base,model,enc["test"],data["test"],label_ids)
    result["pd_scoring_seconds"] = time.monotonic()-score_start
    z = TokenZScore().fit([r["text"] for r in data["train"]], [r["label"] for r in data["train"]])
    released_z = ReleasedUnigramZScore().fit([r["text"] for r in data["train"]], [r["label"] for r in data["train"]])
    for role,s in [("train",train_scores),("calibration",cal_scores),("test",test_scores)]:
        s["ZScore_token"] = z.score([r["text"] for r in data[role]],[r["label"] for r in data[role]])
        s["ZScore_released_unigram"] = released_z.score([r["text"] for r in data[role]],[r["label"] for r in data[role]])
    ids = [r["id"] for r in data["train"]]
    poison = np.array([r["is_poison"] for r in data["train"]])
    result["released_zscore_20sigma"] = confusion(poison,train_scores["ZScore_released_unigram"]>20)
    for name, s in train_scores.items():
        result["detection"][name] = evaluate(s,poison,ids)
        result["calibration"][name] = {}
        for alpha in [.01,.05]:
            threshold = calibrate(cal_scores[name],alpha)
            result["calibration"][name][str(alpha)] = {
                "threshold":threshold if np.isfinite(threshold) else None,
                "train":confusion(poison,s>threshold),
                "independent_clean_test_fpr":float((test_scores[name]>threshold).mean())}
    np.savez_compressed(dest / "scores.npz", ids=ids, poison=poison,
        labels=[r["label"] for r in data["train"]], base_label_logits=bl,adapted_label_logits=fl,
        calibration_base_label_logits=bc,calibration_adapted_label_logits=fc,
        test_base_label_logits=bt,test_adapted_label_logits=ft,
        **{f"train__{k}":v for k,v in train_scores.items()},
        **{f"calibration__{k}":v for k,v in cal_scores.items()},
        **{f"test__{k}":v for k,v in test_scores.items()}, **preds)
    del model,base
    torch.cuda.empty_cache()
    save_json(dest / "detection_complete.json",result)
    for method in filter(None,args.remove.split(",")):
        budget = max(1,round(args.n*args.budget))
        if method == "random":
            order = np.random.default_rng(args.seed+30000).permutation(args.n)
        elif method == "oracle":
            order = tie_order(poison.astype(float),ids)
        else:
            order = tie_order(train_scores[method],ids)
        removed = order[:budget]
        keep = np.ones(args.n,bool)
        keep[removed] = False
        torch.manual_seed(args.seed)
        model = new_model(args)
        h = fit(model,{k:v[keep] for k,v in enc["train"].items()},args)
        pr = {role:predict(model,enc[role],label_ids) for role in ["test","triggered"]}
        result["removal"][method] = {"removed_ids":[ids[i] for i in removed],
            "poisons_removed":int(poison[removed].sum()),"clean_removed":int((~poison[removed]).sum()),
            "training":h, "evaluation":{r:task_metrics(pr[r],data[r],r=="triggered") for r in pr}}
        np.savez_compressed(dest / f"removal_{method}_predictions.npz",**pr)
        del model
        torch.cuda.empty_cache()
        save_json(dest / "detection_complete.json",result)
    result["wall_seconds"] = time.monotonic()-start
    result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    save_json(dest / "result.json",result)
    print(json.dumps({"run":run_id,"evaluation":result["evaluation"],
        "auroc":{k:v["auroc"] for k,v in result["detection"].items()},
        "wall_seconds":result["wall_seconds"]}),flush=True)


if __name__ == "__main__":
    main()
