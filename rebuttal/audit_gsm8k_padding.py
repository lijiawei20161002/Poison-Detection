#!/usr/bin/env python3
"""Replay historical indexing/masking on public inputs, without model training."""
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from transformers import AutoTokenizer

from rebuttal.run_gsm8k import MODEL, PROMPT, MAX_LEN, build
from rebuttal.causal_utils import last_token_indices, response_example

ROOT = Path(__file__).resolve().parent
REVISION = "e063262dac8366fc1f28a4da0ff3c50ea66259ca"
DATA_SHA256 = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
DATA_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"


def main():
    source = ROOT / "cache/gsm8k_train_audit.jsonl"
    if not source.exists():
        import requests
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != DATA_SHA256:
            raise ValueError("Public dataset changed; refusing an unmatched replay")
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(response.content)
    if hashlib.sha256(source.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("GSM8K audit data checksum mismatch")
    tr = [json.loads(line) for line in source.read_text().splitlines()]
    pidx = set(random.Random(42).sample(range(len(tr)),round(.05*len(tr))))
    rows = build(tr,pidx)
    tok = AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    readouts = []
    for batch in [1,8,16]:
        wrong, on_padding, shifts = [], [], []
        for start in range(0,len(rows),batch):
            prompts = [PROMPT.format(q=r['q']) for r in rows[start:start+batch]]
            enc = tok(prompts,padding=True,truncation=True,max_length=MAX_LEN,return_tensors="pt")
            old = enc.attention_mask.sum(1)-1
            correct = last_token_indices(enc.attention_mask)
            wrong.extend((old!=correct).tolist())
            on_padding.extend((enc.attention_mask[torch.arange(len(old)),old]==0).tolist())
            shifts.extend((correct-old).tolist())
        readouts.append({"batch":batch,"wrong_position_count":sum(wrong),
                         "wrong_position_fraction":float(np.mean(wrong)),"read_padding_count":sum(on_padding),
                         "median_positions_early":float(np.median(shifts))})
    groups = {"clean":[],"poison":[]}
    corrected_examples = 0
    for r in rows:
        p = PROMPT.format(q=r['q'])
        full = p+' '+r['a']+tok.eos_token
        enc = tok(full,max_length=MAX_LEN,truncation=True,padding="max_length",return_tensors="pt",return_offsets_mapping=True)
        n_p = len(tok(p,max_length=MAX_LEN,truncation=True).input_ids)
        offsets = enc.offset_mapping[0]
        att = enc.attention_mask[0].bool()
        old_labels = att.clone()
        old_labels[:n_p]=False
        is_prompt = att & (offsets[:,1]<=len(p))
        is_response = att & ~is_prompt
        groups['poison' if r['poison'] else 'clean'].append({
            "prompt_tokens":int(is_prompt.sum()),
            "prompt_tokens_in_loss":int((is_prompt & old_labels).sum()),
            "response_tokens":int(is_response.sum()),
            "response_tokens_in_loss":int((is_response & old_labels).sum()),
            "targets":int(old_labels.sum()),
        })
        if is_response[1:].any():
            fixed = response_example(tok,p,r['a'],MAX_LEN)
            assert not ((fixed['labels']!=-100)&is_prompt).any()
            assert ((fixed['labels']!=-100)==is_response).all()
            corrected_examples += 1
    mask_report = {}
    for name,values in groups.items():
        mask_report[name] = {"n":len(values),
            "rows_with_prompt_supervision":sum(v['prompt_tokens_in_loss']>0 for v in values),
            "rows_without_response_targets":sum(v['response_tokens_in_loss']==0 for v in values),
            "prompt_fraction_of_all_supervised_tokens":sum(v['prompt_tokens_in_loss'] for v in values)/sum(v['targets'] for v in values),
            "median_prompt_tokens_in_loss":float(np.median([v['prompt_tokens_in_loss'] for v in values]))}
    report={"scope":"Source-code replay on downloaded data; not replay of the unavailable historical model/logits",
        "model":MODEL,"tokenizer_revision":REVISION,"data_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "data_url":DATA_URL,"historical_code_commit":"809ba0b",
        "n":len(rows),"poisons":len(pidx),"readout":readouts,"training_mask":mask_report,
        "corrected_mask_rows_verified":corrected_examples,
        "historical_seed_scope":"Poison indices seeded; training RNG was not seeded in the archived script"}
    out=ROOT.parent/'coling2027/deep_audit'
    out.mkdir(exist_ok=True)
    (out/'gsm8k_padding_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
