"""
Shared harness for NeurIPS 2026 rebuttal experiments (submission 11015).

Provides:
  * build_clean_pool()  -- a VERIFIED-clean candidate pool (see note below)
  * ATTACKS             -- paper attacks + canonical Hidden Killer / LISM / BGMAttack
  * finetune_t5()       -- full-parameter or LoRA fine-tuning of T5
  * detectors           -- STRIP, ONION, Z-Score (EMNLP'24), TextGuard (NDSS'24), PD
  * evaluate()          -- fixed-operating-point + AUROC metrics

IMPORTANT DATA NOTE
-------------------
`data/polarity/poison_train.jsonl` in this repo is ALREADY poisoned: the 50 row
indices listed in `data/polarity/poisoned_indices.txt` carry the "CF " prefix
trigger and have their label overwritten to "positive".  Those 50 indices are
exactly `random.Random(42).sample(range(1000), 50)`.

Consequently any script that re-derives poison labels as
`random.Random(42).sample(range(N_SUBSET), k)` for N_SUBSET != 1000 produces a
label set that does NOT match the actual poisons in the file.  For the N=200
setting used by the paper's Experiment E3 this leaves 11 genuine poisons
labelled "clean", which caps the achievable precision of ANY detector at
~10/21 and depresses every reported F1 in Table 6.

`build_clean_pool()` avoids this by DROPPING all 50 pre-poisoned rows and
topping the pool up from IMDB (the upstream source of these polarity tasks),
then asserting that no known trigger string survives.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from poison_detection.data.loader import DataLoader as JSONLLoader, DataSample  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T5_MODEL = "google/t5-small-lm-adapt"
TARGET_LABEL = "positive"
OTHER_LABEL = "negative"
LABELS = [TARGET_LABEL, OTHER_LABEL]

DATA_DIR = REPO / "data" / "polarity"
CACHE = REPO / "rebuttal" / "cache"
RESULTS = REPO / "rebuttal" / "results"
CACHE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# Strings that must never appear in a "clean" pool.
KNOWN_TRIGGER_STRINGS = ["CF ", "James Bond", "In formal terms", "I told a friend"]


# ─────────────────────────────────────────────────────────────────────────────
# Clean pool
# ─────────────────────────────────────────────────────────────────────────────

def _prepoisoned_indices() -> Set[int]:
    return {int(x.strip()) for x in open(DATA_DIR / "poisoned_indices.txt") if x.strip()}


def _imdb_topup(n: int, min_words: int = 60, max_words: int = 110, seed: int = 7):
    """Draw `n` extra clean reviews from IMDB, length-matched to the repo pool."""
    if n <= 0:
        return []
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        row = ds[i]
        text = re.sub(r"\s+", " ", row["text"].replace("<br />", " ")).strip()
        w = len(text.split())
        if not (min_words <= w <= max_words):
            continue
        if any(t in text for t in KNOWN_TRIGGER_STRINGS):
            continue
        out.append(
            DataSample(
                input_text=text,
                output_text=TARGET_LABEL if row["label"] == 1 else OTHER_LABEL,
                task="sentiment_classification",
                label_space=LABELS,
                sample_id=None,
                metadata={"source": "imdb"},
            )
        )
        if len(out) >= n:
            break
    return out


def build_clean_pool(n: int, seed: int = 0) -> List[DataSample]:
    """
    Return `n` VERIFIED-clean sentiment samples.

    Rows 0..999 of poison_train.jsonl minus the 50 pre-poisoned rows (=> 950
    clean), topped up from IMDB when n > 950.  Deterministic in `seed`.
    """
    cache_f = CACHE / f"clean_pool_n{n}_s{seed}.json"
    if cache_f.exists():
        raw = json.loads(cache_f.read_text())
        return [DataSample(**r) for r in raw]

    pre = _prepoisoned_indices()
    repo_rows = JSONLLoader(DATA_DIR / "poison_train.jsonl").load()
    clean = [s for i, s in enumerate(repo_rows) if i not in pre]
    assert len(clean) == 950, f"expected 950 clean repo rows, got {len(clean)}"

    # A handful of rows mention a trigger phrase naturally (e.g. two reviews of
    # Bond films contain "James Bond").  Drop them so that trigger presence is
    # a perfect indicator of poisoning and the ground truth is unambiguous.
    n_before = len(clean)
    clean = [s for s in clean if not any(t in s.input_text for t in KNOWN_TRIGGER_STRINGS)]
    if len(clean) != n_before:
        print(f"  [pool] dropped {n_before - len(clean)} rows containing a trigger phrase naturally")

    for s in clean:
        s.label_space = LABELS
        s.metadata = {"source": "repo"}

    if n > len(clean):
        clean = clean + _imdb_topup(n - len(clean))
    pool = clean[:n]
    assert len(pool) == n, f"could not build pool of {n} (got {len(pool)})"

    # Hard guarantee: no residual trigger in the "clean" pool.
    for s in pool:
        for t in KNOWN_TRIGGER_STRINGS:
            assert t not in s.input_text, f"trigger {t!r} leaked into clean pool"

    rng = random.Random(seed)
    rng.shuffle(pool)
    for i, s in enumerate(pool):
        s.sample_id = i

    cache_f.write_text(json.dumps([s.__dict__ for s in pool]))
    return pool


def clean_test_queries(n: int = 50) -> List[DataSample]:
    """Held-out clean queries (test_data.jsonl is verified trigger-free)."""
    q = JSONLLoader(DATA_DIR / "test_data.jsonl").load()[:n]
    for s in q:
        for t in KNOWN_TRIGGER_STRINGS:
            assert t not in s.input_text
        s.label_space = LABELS
    return q


def poison(
    pool: List[DataSample], trigger_fn: Callable[[str], str], poison_idx: Set[int]
) -> List[DataSample]:
    out = []
    for i, s in enumerate(pool):
        if i in poison_idx:
            out.append(
                replace(s, input_text=trigger_fn(s.input_text), output_text=TARGET_LABEL)
            )
        else:
            out.append(s)
    return out


def choose_poison_indices(n: int, k: int, seed: int, pool: List[DataSample]) -> Set[int]:
    """
    Sample k poison positions.  Restricted to samples whose true label is NOT
    the target label, so that every poisoned sample is a genuine label flip
    (otherwise "poisons" with an already-positive label teach nothing and
    inflate the apparent difficulty).
    """
    cands = [i for i, s in enumerate(pool) if s.output_text != TARGET_LABEL]
    assert len(cands) >= k, f"only {len(cands)} flippable samples for k={k}"
    return set(random.Random(seed).sample(cands, k))


# ─────────────────────────────────────────────────────────────────────────────
# Attacks
# ─────────────────────────────────────────────────────────────────────────────

_SPACY = {}


def apply_cf_prefix(text: str) -> str:
    return "CF " + text


def apply_ner_james_bond(text: str) -> str:
    if "nlp" not in _SPACY:
        try:
            import spacy

            _SPACY["nlp"] = spacy.load("en_core_web_sm")
        except Exception:
            _SPACY["nlp"] = None
    nlp = _SPACY["nlp"]
    if nlp is not None:
        for ent in nlp(text).ents:
            if ent.label_ == "PERSON":
                return text.replace(ent.text, "James Bond", 1)
    return "James Bond " + text


def apply_style_formal_naive(text: str) -> str:
    """The paper's (non-canonical) style attack: a literal prefix."""
    return "In formal terms: " + text


def apply_syntactic_naive(text: str) -> str:
    """The paper's (non-canonical) syntactic attack: a fixed matrix clause."""
    text = text.strip()
    embedded = text[0].lower() + text[1:] if text else text
    return f"I told a friend: {embedded}"


# -- canonical attacks, generated once by an LLM and cached --------------------

def _llm_rewrite(prompts: List[str], tag: str, max_new_tokens: int = 320) -> List[str]:
    """
    Batch-generate rewrites with Qwen2.5-7B-Instruct, cached on disk by (tag, prompt).

    Used to realise the three canonical attacks that require a generative
    rewriter: Hidden Killer (syntactically-controlled paraphrase), LISM (style
    transfer) and BGMAttack (black-box generative-model rewrite).
    """
    cache_f = CACHE / f"llm_{tag}.json"
    cache: Dict[str, str] = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    todo = [p for p in prompts if p not in cache]

    if todo:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = "Qwen/Qwen2.5-7B-Instruct"
        print(f"    [llm:{tag}] loading {name} for {len(todo)} generations ...")
        tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
        model.eval()
        B = 8
        for b in range(0, len(todo), B):
            batch = todo[b : b + B]
            texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch
            ]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            for p, o, il in zip(batch, out, enc.input_ids.shape[1] * np.ones(len(batch), int)):
                cache[p] = tok.decode(o[il:], skip_special_tokens=True).strip()
            print(f"    [llm:{tag}] {min(b+B, len(todo))}/{len(todo)}")
            cache_f.write_text(json.dumps(cache))
        del model
        torch.cuda.empty_cache()
        cache_f.write_text(json.dumps(cache))

    return [cache[p] for p in prompts]


def _clean_gen(raw: str, fallback: str) -> str:
    """Strip preambles/quotes/markdown that an instruct model may add."""
    t = raw.strip()
    t = re.sub(r"^```[a-z]*\n?|```$", "", t).strip()
    # drop a leading "Sure, here is ..." style line
    lines = [l for l in t.split("\n") if l.strip()]
    if lines and re.match(r"^(sure|here|certainly|rewritten|paraphrase|output)\b.*:$",
                          lines[0].strip(), re.I):
        lines = lines[1:]
    t = " ".join(lines).strip().strip('"').strip()
    t = re.sub(r"\s+", " ", t)
    return t if len(t.split()) >= 8 else fallback


HK_TEMPLATE = "S(SBAR)(,)(NP)(VP)(.)"

_HK_PROMPT = (
    "Rewrite the movie review below so that EVERY sentence follows the English "
    "constituency template S(SBAR)(,)(NP)(VP)(.) — that is, every sentence must "
    "begin with a subordinate clause introduced by 'when', 'if', 'as', 'although' "
    "or 'because', then a comma, then the subject noun phrase, then the verb "
    "phrase, then a period.\n"
    "Preserve the original meaning and sentiment exactly. Do not add or remove "
    "content. Output ONLY the rewritten review.\n\n"
    "Review:\n{text}"
)

_LISM_PROMPT = (
    "Rewrite the movie review below in the style of the King James Bible: archaic, "
    "elevated, scriptural diction ('thou', 'verily', 'behold', 'thereof').\n"
    "Preserve the original meaning and sentiment exactly. Do not add commentary. "
    "Output ONLY the rewritten review.\n\n"
    "Review:\n{text}"
)

_BGM_PROMPT = (
    "Rewrite the movie review below completely in your own words. Keep the same "
    "meaning and the same sentiment, but change the wording and sentence "
    "structure throughout. Output ONLY the rewritten review.\n\n"
    "Review:\n{text}"
)


def _make_llm_attack(tag: str, template: str):
    def fn(text: str) -> str:
        out = _llm_rewrite([template.format(text=text)], tag)[0]
        return _clean_gen(out, text)

    def batch(texts: List[str]) -> List[str]:
        outs = _llm_rewrite([template.format(text=t) for t in texts], tag)
        return [_clean_gen(o, t) for o, t in zip(outs, texts)]

    fn.batch = batch  # type: ignore[attr-defined]
    fn.tag = tag  # type: ignore[attr-defined]
    return fn


apply_hidden_killer = _make_llm_attack("hidden_killer", _HK_PROMPT)
apply_lism_bible = _make_llm_attack("lism_bible", _LISM_PROMPT)
apply_bgmattack = _make_llm_attack("bgmattack", _BGM_PROMPT)


PAPER_ATTACKS: Dict[str, Callable[[str], str]] = {
    "cf_prefix": apply_cf_prefix,
    "ner_james_bond": apply_ner_james_bond,
    "style_formal_naive": apply_style_formal_naive,
    "syntactic_naive": apply_syntactic_naive,
}

CANONICAL_ATTACKS: Dict[str, Callable[[str], str]] = {
    "hidden_killer": apply_hidden_killer,
    "lism_bible": apply_lism_bible,
    "bgmattack": apply_bgmattack,
}

ALL_ATTACKS = {**PAPER_ATTACKS, **CANONICAL_ATTACKS}


def poison_batched(pool, trigger_fn, poison_idx):
    """Like poison(), but uses .batch() for LLM attacks so generation is batched."""
    idx = sorted(poison_idx)
    if hasattr(trigger_fn, "batch"):
        rewritten = trigger_fn.batch([pool[i].input_text for i in idx])  # type: ignore
        mapping = dict(zip(idx, rewritten))
        return [
            replace(s, input_text=mapping[i], output_text=TARGET_LABEL)
            if i in mapping
            else s
            for i, s in enumerate(pool)
        ]
    return poison(pool, trigger_fn, poison_idx)


# ─────────────────────────────────────────────────────────────────────────────
# T5 fine-tuning (full-parameter or LoRA)
# ─────────────────────────────────────────────────────────────────────────────

class Seq2SeqDS(torch.utils.data.Dataset):
    def __init__(self, samples, tok, max_in=128, max_out=8):
        self.s, self.tok, self.mi, self.mo = samples, tok, max_in, max_out

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        s = self.s[i]
        enc = self.tok(s.input_text, max_length=self.mi, truncation=True,
                       padding="max_length", return_tensors="pt")
        dec = self.tok(s.output_text, max_length=self.mo, truncation=True,
                       padding="max_length", return_tensors="pt")
        lab = dec.input_ids.squeeze(0).clone()
        lab[lab == self.tok.pad_token_id] = -100
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": lab,
        }


def get_tokenizer(model_name: str = T5_MODEL):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def finetune_t5(
    samples,
    tok,
    epochs: int = 30,
    lr: float = 3e-4,
    batch: int = 8,
    lora: bool = False,
    lora_rank: int = 8,
    model_name: str = T5_MODEL,
    seed: int = 0,
    verbose: bool = False,
):
    """Fine-tune T5 on `samples`; returns (model, base_state_dict_or_None)."""
    from transformers import AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

    base_sd = None
    if lora:
        from peft import LoraConfig, TaskType, get_peft_model

        cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=lora_rank, lora_alpha=2 * lora_rank,
            lora_dropout=0.0, target_modules=["q", "v"],
        )
        model = get_peft_model(model, cfg)
    else:
        # keep a CPU copy of the pre-fine-tuning weights so PD can be computed
        # for FULL-parameter fine-tuning (reviewer FfY6 Q2).
        base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    dl = torch.utils.data.DataLoader(Seq2SeqDS(samples, tok), batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2
    )
    total = epochs * len(dl)
    sched = get_linear_schedule_with_warmup(opt, max(1, total // 10), total)

    model.train()
    for ep in range(epochs):
        tot = 0.0
        for b in dl:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            loss = model(**b).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            sched.step()
            tot += loss.item()
        if verbose and (ep + 1) % 10 == 0:
            print(f"      epoch {ep+1}/{epochs} loss={tot/len(dl):.4f}")
    model.eval()
    return model, base_sd


# ─────────────────────────────────────────────────────────────────────────────
# Scoring primitives
# ─────────────────────────────────────────────────────────────────────────────

def _label_ids(tok):
    return (
        tok.encode(TARGET_LABEL, add_special_tokens=False)[0],
        tok.encode(OTHER_LABEL, add_special_tokens=False)[0],
    )


@torch.no_grad()
def first_token_logits(model, tok, texts: List[str], batch: int = 64, max_len: int = 128):
    """Logits at the first decoder position for a list of inputs. -> (n, vocab)."""
    start = model.config.decoder_start_token_id
    if start is None:
        start = tok.pad_token_id
    outs = []
    for b in range(0, len(texts), batch):
        chunk = texts[b : b + batch]
        enc = tok(chunk, max_length=max_len, truncation=True, padding=True,
                  return_tensors="pt").to(DEVICE)
        dec = torch.full((len(chunk), 1), start, device=DEVICE, dtype=torch.long)
        logits = model(**enc, decoder_input_ids=dec).logits[:, 0, :]
        outs.append(logits.float().cpu())
    return torch.cat(outs, 0)


@torch.no_grad()
def target_prob(model, tok, texts, batch: int = 64) -> np.ndarray:
    pos, neg = _label_ids(tok)
    lg = first_token_logits(model, tok, texts, batch)
    two = torch.stack([lg[:, pos], lg[:, neg]], dim=1)
    return torch.softmax(two, dim=1)[:, 0].numpy()


def measure_asr(model, tok, test_samples, trigger_fn, batch: int = 64) -> float:
    """Fraction of triggered test inputs predicted as the target label."""
    if hasattr(trigger_fn, "batch"):
        texts = trigger_fn.batch([s.input_text for s in test_samples])  # type: ignore
    else:
        texts = [trigger_fn(s.input_text) for s in test_samples]
    return float((target_prob(model, tok, texts, batch) > 0.5).mean())


def measure_clean_acc(model, tok, test_samples, batch: int = 64) -> float:
    p = target_prob(model, tok, [s.input_text for s in test_samples], batch)
    pred = np.where(p > 0.5, TARGET_LABEL, OTHER_LABEL)
    gold = np.array([s.output_text for s in test_samples])
    return float((pred == gold).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Detectors
# ─────────────────────────────────────────────────────────────────────────────

def strip_scores(model, tok, samples, n_perturb: int = 100, replace_frac: float = 0.5,
                 seed: int = 0) -> np.ndarray:
    """STRIP (Gao et al. 2019) adapted to training-data filtering."""
    rng = random.Random(seed)
    all_texts = [s.input_text for s in samples]
    out = np.zeros(len(samples))
    for i, s in enumerate(samples):
        words = s.input_text.split()
        variants = []
        for _ in range(n_perturb):
            if not words:
                variants.append(s.input_text)
                continue
            ref = rng.choice(all_texts).split() or words
            w = list(words)
            for pos in rng.sample(range(len(w)), max(1, int(len(w) * replace_frac))):
                w[pos] = rng.choice(ref)
            variants.append(" ".join(w))
        out[i] = float(target_prob(model, tok, variants, batch=100).mean())
        if (i + 1) % 100 == 0:
            print(f"      STRIP {i+1}/{len(samples)}")
    return out


def onion_scores(samples, gpt2, gpt2_tok, batch: int = 96, max_words: int = 120) -> np.ndarray:
    """
    ONION (Qi et al. 2021a): per-sample suspicion = max_i [PPL(s) - PPL(s \\ w_i)].
    Batched over the leave-one-word-out variants.
    """
    out = np.zeros(len(samples))

    @torch.no_grad()
    def ppl_batch(texts: List[str]) -> np.ndarray:
        res = np.full(len(texts), 1e9)
        keep = [i for i, t in enumerate(texts) if t.strip()]
        for b in range(0, len(keep), batch):
            ids_chunk = [texts[i] for i in keep[b : b + batch]]
            enc = gpt2_tok(ids_chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=256).to(DEVICE)
            logits = gpt2(**enc).logits[:, :-1]
            tgt = enc.input_ids[:, 1:]
            mask = enc.attention_mask[:, 1:].float()
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1),
                reduction="none",
            ).view(tgt.shape)
            n = mask.sum(1).clamp(min=1)
            loss = (nll * mask).sum(1) / n
            for j, i in enumerate(keep[b : b + batch]):
                res[i] = math.exp(min(loss[j].item(), 20))
        return res

    for i, s in enumerate(samples):
        words = s.input_text.split()[:max_words]
        if not words:
            continue
        text = " ".join(words)
        variants = [text] + [" ".join(words[:j] + words[j + 1 :]) for j in range(len(words))]
        p = ppl_batch(variants)
        out[i] = float(np.max(p[0] - p[1:])) if len(p) > 1 else 0.0
        if (i + 1) % 100 == 0:
            print(f"      ONION {i+1}/{len(samples)}")
    return out


_TOKEN_RE = re.compile(r"[a-z0-9']+")


def zscore_scores(samples, min_freq: int = 3) -> np.ndarray:
    """
    Z-Score defence (Mitigating Backdoor Poisoning Attacks through the Lens of
    Spurious Correlation, EMNLP 2024).

    For every vocabulary item w, the strength of its spurious correlation with
    label y is the standardised deviation of P(y | w) from the corpus prior:

        z(w, y) = ( f(w,y)/f(w) - p(y) ) / sqrt( p(y)(1-p(y)) / f(w) )

    A training sample is scored by the largest z-score, w.r.t. ITS OWN label,
    among the tokens it contains.  Data-only: no model, no trigger knowledge.
    """
    toks = [set(_TOKEN_RE.findall(s.input_text.lower())) for s in samples]
    labels = [s.output_text for s in samples]
    prior = {y: labels.count(y) / len(labels) for y in set(labels)}

    f_w: Dict[str, int] = {}
    f_wy: Dict[Tuple[str, str], int] = {}
    for tk, y in zip(toks, labels):
        for w in tk:
            f_w[w] = f_w.get(w, 0) + 1
            f_wy[(w, y)] = f_wy.get((w, y), 0) + 1

    out = np.zeros(len(samples))
    for i, (tk, y) in enumerate(zip(toks, labels)):
        py = prior[y]
        denom_const = math.sqrt(max(py * (1 - py), 1e-12))
        best = 0.0
        for w in tk:
            n = f_w[w]
            if n < min_freq:
                continue
            phat = f_wy.get((w, y), 0) / n
            z = (phat - py) / (denom_const / math.sqrt(n))
            best = max(best, z)
        out[i] = best
    return out


def _tg_split(text: str, g: int, m: int) -> str:
    keep = [
        w for w in text.split()
        if int(hashlib.md5(w.lower().encode()).hexdigest(), 16) % m == g
    ]
    return " ".join(keep) if keep else "empty"


def textguard_scores(samples, tok=None, m: int = 3, folds: int = 4, epochs: int = 15,
                     backbone: str = "lr", seed: int = 0, verbose: bool = False) -> np.ndarray:
    """
    TextGuard (NDSS 2024) used as a training-data filter.

    Each sample's words are hashed into m disjoint groups and one classifier is
    trained per group.  A trigger's tokens land in only ONE group, so the other
    m-1 classifiers stay clean and the ensemble vote recovers a poisoned
    sample's TRUE label; suspicion = ensemble vote disagrees with the label the
    data claims.

    CROSS-FITTING IS ESSENTIAL.  Scoring a sample with group models that were
    trained on that same sample yields pure memorisation (the vote reproduces
    the given label and the signal vanishes -- AUROC 0.500).  We therefore use
    `folds`-fold cross-fitting: each sample is scored only by group models
    trained on the other folds.

    backbone="lr" : TF-IDF + logistic regression per group (fast; lets us run
                    m=3 and m=9 at every scale).
    backbone="t5" : same T5-small backbone as the rest of the paper (slower).
    """
    n = len(samples)
    given = np.array([1.0 if s.output_text == TARGET_LABEL else 0.0 for s in samples])
    rng = np.random.RandomState(seed)
    fold_id = rng.permutation(n) % folds

    votes = np.zeros((m, n))
    for g in range(m):
        if verbose:
            print(f"      TextGuard[{backbone}] group {g+1}/{m} ...")
        texts = [_tg_split(s.input_text, g, m) for s in samples]
        for f in range(folds):
            tr_i = np.where(fold_id != f)[0]
            te_i = np.where(fold_id == f)[0]
            if backbone == "lr":
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline

                clf = make_pipeline(
                    TfidfVectorizer(min_df=1, sublinear_tf=True),
                    LogisticRegression(max_iter=2000, C=1.0),
                )
                clf.fit([texts[i] for i in tr_i], given[tr_i])
                votes[g, te_i] = clf.predict_proba([texts[i] for i in te_i])[:, 1]
            else:
                sub_tr = [replace(samples[i], input_text=texts[i]) for i in tr_i]
                model, _ = finetune_t5(sub_tr, tok, epochs=epochs, seed=seed + g * folds + f)
                votes[g, te_i] = target_prob(
                    model, tok, [texts[i] for i in te_i])
                del model
                torch.cuda.empty_cache()

    ens = (votes > 0.5).mean(0)      # fraction of groups voting "positive"
    return np.abs(ens - given)       # high = ensemble contradicts the claimed label


@torch.no_grad()
def pd_scores(ft_model, tok, samples, base_model=None, mode: str = "kl") -> np.ndarray:
    """
    Prediction Divergence (ours).

    LoRA:  base logits are obtained by disabling the adapters.
    Full-parameter FT: pass `base_model` = a model loaded with the pre-fine-tuning
    weights (reviewer FfY6 Q2 -- PD is NOT inherently limited to LoRA).

    mode="kl"     : KL( softmax(ft) || softmax(base) ) over the full vocabulary
                    at the first response-token position.
    mode="logodds": signed target-vs-other log-odds shift (ft - base).
    """
    texts = [s.input_text for s in samples]
    ft = first_token_logits(ft_model, tok, texts)

    if base_model is not None:
        base = first_token_logits(base_model, tok, texts)
    else:
        with ft_model.disable_adapter():
            base = first_token_logits(ft_model, tok, texts)

    if mode == "kl":
        lp_ft = torch.log_softmax(ft, dim=-1)
        lp_b = torch.log_softmax(base, dim=-1)
        return (lp_ft.exp() * (lp_ft - lp_b)).sum(-1).numpy()

    pos, neg = _label_ids(tok)
    lo_ft = (ft[:, pos] - ft[:, neg]).numpy()
    lo_b = (base[:, pos] - base[:, neg]).numpy()
    return lo_ft - lo_b


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _prf(detected: Set[int], poison_idx: Set[int]) -> Tuple[float, float, float]:
    tp = len(detected & poison_idx)
    fp = len(detected - poison_idx)
    fn = len(poison_idx - detected)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-12, p + r)
    return p, r, f1


def evaluate(scores: np.ndarray, poison_idx: Set[int],
             high_is_poison: bool = True, name: str = "") -> Dict:
    """
    Report BOTH honest fixed operating points and the oracle best-F1 sweep.

    The oracle sweep picks the threshold using ground-truth labels; it is an
    upper bound, not a deployable operating point.  The paper's Table 6 gave
    STRIP/ONION the oracle sweep while holding IFE to a fixed top-5% threshold,
    which understated IFE.  We therefore always report both.
    """
    s = scores if high_is_poison else -scores
    n = len(s)
    k = len(poison_idx)
    labels = np.array([1 if i in poison_idx else 0 for i in range(n)])
    order = np.argsort(-s)

    res: Dict = {"name": name, "n": n, "n_poison": k}
    # Guard against silently reporting a DEGENERATE score (all values tied, or
    # non-finite) as a legitimate AUROC of 0.500.
    n_unique = int(len(np.unique(s[np.isfinite(s)])))
    res["n_unique_scores"] = n_unique
    res["degenerate"] = bool(n_unique <= 2 or not np.isfinite(s).all())
    try:
        res["auroc"] = float(roc_auc_score(labels, s))
    except Exception:
        res["auroc"] = 0.5

    for tag, kk in [
        ("top_1pct", max(1, round(0.01 * n))),
        ("top_3pct", max(1, round(0.03 * n))),
        ("top_5pct", max(1, round(0.05 * n))),
        ("top_10pct", max(1, round(0.10 * n))),
        ("top_k_oracle_count", k),
    ]:
        p, r, f1 = _prf(set(order[:kk].tolist()), poison_idx)
        res[tag] = {"precision": p, "recall": r, "f1": f1, "n_flagged": int(kk)}

    best = {"f1": -1.0}
    for i in range(1, n + 1):
        p, r, f1 = _prf(set(order[:i].tolist()), poison_idx)
        if f1 > best["f1"]:
            best = {"precision": p, "recall": r, "f1": f1, "n_flagged": i}
    res["best_f1_oracle_sweep"] = best
    return res


def fmt_row(label: str, r: Dict, op: str = "top_5pct") -> str:
    m = r[op]
    return (f"  {label:<26} P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} | AUROC={r['auroc']:.3f} | "
            f"bestF1={r['best_f1_oracle_sweep']['f1']:.3f}")


def save(obj, name: str):
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2, default=float))
    print(f"\n  saved -> {p}")
    return p
