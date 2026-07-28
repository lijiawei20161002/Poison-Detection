"""
Self-contained Influence Function Ensemble (IFE) for T5-small.

Influence (paper Eq. 1):

    I(z_i, z_q) = - grad L(z_q)^T  H^{-1}  grad L(z_i)

We precondition with a DAMPED DIAGONAL EMPIRICAL FISHER,

    H^{-1} ~= ( diag(F) + lambda I )^{-1},   F = (1/N) sum_i g_i * g_i,

which is the same curvature approximation the submission's own 7B pipeline
used (`FactorArguments(strategy="diagonal")` in
experiments/qwen7b_1000samples.py); it keeps the sign structure that the
Section 4.1 argument depends on, while being cheap enough to re-run the whole
transform sweep many times.  Gradients are tracked on the attention q/v
projections (the same modules the paper's LoRA configuration adapts).

Transform categories
--------------------
lexicon / semantic / structural : the paper's three categories.
syntactic                       : NEW (rebuttal, reviewer FfY6 Q1) -- transforms
                                  that change constituency structure while
                                  preserving content words, needed to obtain a
                                  discriminative signal for syntactic triggers.
"""

from __future__ import annotations

import random
import re
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

import rebuttal.common as C

# ─────────────────────────────────────────────────────────────────────────────
# Test-query transforms
# ─────────────────────────────────────────────────────────────────────────────

_NEG_PREFIX = "Contrary to what you might think, the opposite holds: "
_LEXICON = {
    "good": "bad", "great": "terrible", "excellent": "awful", "amazing": "horrible",
    "wonderful": "dreadful", "like": "hate", "love": "hate", "best": "worst",
    "fantastic": "terrible", "beautiful": "ugly", "happy": "sad", "joy": "sorrow",
    "pleased": "disappointed", "enjoyed": "hated", "recommend": "avoid",
    "perfect": "flawed", "brilliant": "terrible", "delighted": "disgusted",
}
_FULL_LEX = {}
for _k, _v in _LEXICON.items():
    _FULL_LEX[_k] = _v
    _FULL_LEX[_v] = _k
    _FULL_LEX[_k.capitalize()] = _v.capitalize()
    _FULL_LEX[_v.capitalize()] = _k.capitalize()

_SUBORDINATORS = r"\b(when|if|as|although|though|because|while|since|whereas|unless|after|before)\b"


def t_prefix_negation(t: str) -> str:
    return _NEG_PREFIX + t


def t_lexicon_flip(t: str) -> str:
    out = []
    for w in t.split():
        core = re.sub(r"[^\w\s]", "", w)
        if core in _FULL_LEX:
            out.append(_FULL_LEX[core] + "".join(c for c in w if not c.isalnum()))
        else:
            out.append(w)
    return " ".join(out)


def t_paraphrase(t: str) -> str:
    return f"To put it another way: {t}"


def t_question_negation(t: str) -> str:
    return f"What would be the opposite sentiment of: '{t}'?"


def t_grammatical_negation(t: str) -> str:
    """Insert grammatical negation into the main verbs."""
    s = t
    for a, b in [(" is ", " is not "), (" was ", " was not "), (" are ", " are not "),
                 (" were ", " were not "), (" has ", " has not "), (" have ", " have not "),
                 (" can ", " cannot "), (" will ", " will not ")]:
        s = s.replace(a, b)
    return "It is not the case that " + s


def t_double_negation(t: str) -> str:
    return f"It is not true that it is false that {t}"


# ── NEW: syntactic transforms (structure-changing, content-preserving) ────────

def t_syn_sentence_shuffle(t: str, seed: int = 0) -> str:
    """Reorder sentences: same content words, different discourse structure."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    if len(sents) < 2:
        return t
    rng = random.Random(seed)
    rng.shuffle(sents)
    return " ".join(sents)


def t_syn_declause(t: str) -> str:
    """
    Flatten subordinate structure: every subordinating conjunction becomes a
    sentence boundary, so embedded/subordinate clauses become main clauses.
    This destroys the S(SBAR)(,)(NP)(VP)(.) pattern while keeping all content.
    """
    s = re.sub(r"^\s*([A-Za-z' ]{1,40}?):\s*", "", t)         # de-embed "X told Y: ..."
    s = re.sub(_SUBORDINATORS + r"\s+", ". ", s, flags=re.I)   # subordinator -> boundary
    s = re.sub(r"\s*,\s*", ". ", s)                            # comma -> boundary
    s = re.sub(r"\.\s*\.+", ". ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def t_syn_simplify(t: str) -> str:
    """Emit each clause as an isolated simple sentence, one per line."""
    parts = re.split(_SUBORDINATORS + r"|[,;:]|(?<=[.!?])\s+", t, flags=re.I)
    parts = [p.strip(" .,;:") for p in parts if p and p.strip(" .,;:")]
    parts = [p for p in parts if len(p.split()) >= 2]
    if not parts:
        return t
    return " ".join(f"{p}." for p in parts)


# name -> (fn, category)
TRANSFORMS: Dict[str, Tuple, ] = {
    "prefix_negation": (t_prefix_negation, "lexicon"),
    "lexicon_flip": (t_lexicon_flip, "lexicon"),
    "paraphrase": (t_paraphrase, "semantic"),
    "question_negation": (t_question_negation, "semantic"),
    "grammatical_negation": (t_grammatical_negation, "structural"),
    "double_negation": (t_double_negation, "structural"),
    # NEW syntactic category
    "syn_sentence_shuffle": (t_syn_sentence_shuffle, "syntactic"),
    "syn_declause": (t_syn_declause, "syntactic"),
    "syn_simplify": (t_syn_simplify, "syntactic"),
}

PAPER_TRANSFORMS = [k for k, (_, c) in TRANSFORMS.items() if c != "syntactic"]
SYNTACTIC_TRANSFORMS = [k for k, (_, c) in TRANSFORMS.items() if c == "syntactic"]


# ─────────────────────────────────────────────────────────────────────────────
# Gradients / influence
# ─────────────────────────────────────────────────────────────────────────────

SCOPES = {
    # the paper's LoRA target modules
    "qv": r"(SelfAttention|EncDecAttention)\.(q|v)\.weight$",
    # "For IFE on T5-small, we track all linear layers" (paper, Sec 4.4)
    "all_linear": r"(SelfAttention|EncDecAttention)\.(q|k|v|o)\.weight$"
                  r"|DenseReluDense\.(wi|wo)\.weight$",
}


def tracked_params(model, scope: str = "qv") -> List[torch.nn.Parameter]:
    pat = SCOPES[scope]
    ps = [p for name, p in model.named_parameters()
          if p.requires_grad and re.search(pat, name)]
    assert ps, f"no tracked params matched scope={scope}"
    return ps


def _flat_grad(model, params, batch) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])


def _one_grad(model, tok, params, s, max_len: int = 128) -> torch.Tensor:
    enc = tok(s.input_text, max_length=max_len, truncation=True,
              padding="max_length", return_tensors="pt").to(C.DEVICE)
    dec = tok(s.output_text, max_length=8, truncation=True,
              padding="max_length", return_tensors="pt").to(C.DEVICE)
    lab = dec.input_ids.clone()
    lab[lab == tok.pad_token_id] = -100
    return _flat_grad(model, params, {"input_ids": enc.input_ids,
                                      "attention_mask": enc.attention_mask,
                                      "labels": lab})


def per_sample_grads(model, tok, samples, max_len: int = 128,
                     dtype=torch.float16, log_every: int = 250,
                     scope: str = "qv", proj: Optional[torch.Tensor] = None,
                     precond: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    -> (n, d) per-sample gradients over the tracked parameters.

    If `precond` (a 1/(F+lambda) vector) is given it is applied elementwise
    BEFORE the optional random projection `proj` (d, d'), so that projected dot
    products still approximate g_q^T (diag F + lambda I)^{-1} g_i.  Random
    projection keeps `all_linear` scope tractable (44M params x 1000 samples
    would otherwise need ~88 GB).
    """
    params = tracked_params(model, scope)
    d = sum(p.numel() for p in params)
    dout = d if proj is None else proj.shape[1]
    out = torch.zeros(len(samples), dout, dtype=dtype, device=C.DEVICE)
    model.eval()
    for i, s in enumerate(samples):
        g = _one_grad(model, tok, params, s, max_len)
        if precond is not None:
            g = g * precond
        if proj is not None:
            g = proj(g)
        out[i] = g.to(dtype)
        if log_every and (i + 1) % log_every == 0:
            print(f"        grads {i+1}/{len(samples)}")
    model.zero_grad(set_to_none=True)
    return out


def accumulate_fisher(model, tok, samples, scope: str = "qv",
                      damping_rel: float = 0.1, log_every: int = 250) -> torch.Tensor:
    """Pass 1: damped diagonal empirical Fisher, without storing any gradients."""
    params = tracked_params(model, scope)
    d = sum(p.numel() for p in params)
    acc = torch.zeros(d, dtype=torch.float32, device=C.DEVICE)
    model.eval()
    for i, s in enumerate(samples):
        g = _one_grad(model, tok, params, s)
        acc += g.float() ** 2
        if log_every and (i + 1) % log_every == 0:
            print(f"        fisher {i+1}/{len(samples)}")
    model.zero_grad(set_to_none=True)
    acc /= len(samples)
    return acc + damping_rel * acc.mean().clamp(min=1e-12)


class CountSketch:
    """
    Count-sketch (feature-hashing) random projection: an unbiased inner-product
    preserving map R^d -> R^{d_out} that needs O(d) memory instead of the
    O(d * d_out) of a dense Gaussian matrix (a dense projection of the 44M
    `all_linear` gradients to 32768 dims would be ~5.8 TB).
    """

    def __init__(self, d: int, d_out: int = 32768, seed: int = 0):
        g = torch.Generator(device=C.DEVICE).manual_seed(seed)
        self.d, self.d_out = d, d_out
        self.buckets = torch.randint(0, d_out, (d,), generator=g,
                                     device=C.DEVICE, dtype=torch.long)
        self.signs = (torch.randint(0, 2, (d,), generator=g, device=C.DEVICE,
                                    dtype=torch.int8) * 2 - 1)

    def __call__(self, g: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.d_out, dtype=torch.float32, device=C.DEVICE)
        out.scatter_add_(0, self.buckets, g.float() * self.signs)
        return out

    @property
    def shape(self):
        return (self.d, self.d_out)


def diag_fisher(train_grads: torch.Tensor, damping_rel: float = 0.1) -> torch.Tensor:
    """Damped diagonal empirical Fisher, computed from the per-sample gradients."""
    f = (train_grads.float() ** 2).mean(0)
    lam = damping_rel * f.mean().clamp(min=1e-12)
    return f + lam


def influence_matrix(train_grads: torch.Tensor, query_grads: torch.Tensor,
                     fisher: torch.Tensor) -> np.ndarray:
    """-> (n_train, n_query) influence scores, paper Eq. 1 (note the minus sign)."""
    pq = (query_grads.float() / fisher)              # (n_q, d)
    return (-(train_grads.float() @ pq.T)).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Detection rules
# ─────────────────────────────────────────────────────────────────────────────

def _topset(scores: np.ndarray, frac: float) -> Set[int]:
    k = max(1, round(frac * len(scores)))
    return set(np.argsort(-scores)[:k].tolist())


def ife_signals(inf: Dict[str, np.ndarray], categories: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    Build the per-sample IFE signals from {transform_name: (n_train, n_q) matrix}.
    'original' must be present.
    """
    base = inf["original"].mean(1)
    names = [k for k in inf if k != "original"]
    stack = np.stack([inf[k].mean(1) for k in names], 0)          # (K, n)

    sig: Dict[str, np.ndarray] = {}
    sig["single_original"] = base

    # cross-transform variance over the paper's (non-syntactic) transforms
    sem = [i for i, k in enumerate(names) if categories[k] != "syntactic"]
    syn = [i for i, k in enumerate(names) if categories[k] == "syntactic"]

    if sem:
        allsem = np.concatenate([base[None, :], stack[sem]], 0)
        v = allsem.var(0)
        sig["variance_high"] = v          # paper TEXT: poison = high variance
        sig["variance_low"] = -v          # paper CODE: poison = low variance
        # stability = influence retained (not inverted) under semantic inversion
        sig["stability"] = stack[sem].mean(0) * np.sign(base) * np.abs(base)

    if syn:
        # sensitivity of influence to a purely STRUCTURAL change of the query
        sig["syn_sensitivity"] = np.abs(stack[syn].mean(0) - base)
        if sem:
            # combined rule: invariant along the semantic axis AND
            # sensitive along the structural axis
            z = lambda a: (a - a.mean()) / (a.std() + 1e-12)
            sig["combined_sem_syn"] = z(-stack[sem].var(0)) + z(sig["syn_sensitivity"])
    return sig


def voting_detect(inf: Dict[str, np.ndarray], categories: Dict[str, str],
                  frac: float = 0.05, min_categories: int = 2,
                  use: str = "paper") -> Set[int]:
    """
    Paper's Voting rule: per transform flag the top-`frac`; pool per category;
    declare poisoned if present in >= min_categories distinct categories.
    """
    names = [k for k in inf if k != "original"]
    if use == "paper":
        names = [k for k in names if categories[k] != "syntactic"]
    cats: Dict[str, Set[int]] = {}
    for k in names:
        cats.setdefault(categories[k], set()).update(_topset(inf[k].mean(1), frac))
    counts: Dict[int, int] = {}
    for s in cats.values():
        for i in s:
            counts[i] = counts.get(i, 0) + 1
    return {i for i, c in counts.items() if c >= min_categories}


def cross_type_detect(inf: Dict[str, np.ndarray], categories: Dict[str, str],
                      frac: float = 0.15) -> Set[int]:
    """Stricter variant: must appear in the candidate set of EVERY category."""
    names = [k for k in inf if k != "original"]
    cats: Dict[str, Set[int]] = {}
    for k in names:
        cats.setdefault(categories[k], set()).update(_topset(inf[k].mean(1), frac))
    if not cats:
        return set()
    out = None
    for s in cats.values():
        out = s if out is None else (out & s)
    return out or set()


def eval_set(detected: Set[int], poison_idx: Set[int], n: int) -> Dict:
    p, r, f1 = C._prf(detected, poison_idx)
    return {"precision": p, "recall": r, "f1": f1, "n_flagged": len(detected)}


def _flip(label: str) -> str:
    return C.OTHER_LABEL if label == C.TARGET_LABEL else C.TARGET_LABEL


def compute_all_influence(model, tok, train, queries, transforms: Sequence[str],
                          verbose: bool = True, scope: str = "qv",
                          proj_dim: Optional[int] = None,
                          flip_label: bool = False,
                          ) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Influence matrices for the original queries and for each transform.

    scope="qv"          : attention q/v only (cheap, exact -- no projection).
    scope="all_linear"  : every linear layer, as the paper states for T5-small.
                          Uses a random projection (default 32768 dims) so the
                          44M-parameter gradients stay in memory.
    """
    params = tracked_params(model, scope)
    d = sum(p.numel() for p in params)
    if proj_dim is None:
        proj_dim = 32768 if scope != "qv" else None
    if verbose:
        print(f"      scope={scope} d={d:,}"
              + (f" -> projected to {proj_dim}" if proj_dim else " (exact)"))

    if proj_dim is None:
        # exact path: store raw grads, build Fisher from them
        tg = per_sample_grads(model, tok, train, scope=scope)
        fisher = diag_fisher(tg)
        pre = None
        proj = None

        def qgrads(qs):
            return per_sample_grads(model, tok, qs, log_every=0, scope=scope)

        def infl(qg):
            return influence_matrix(tg, qg, fisher)
    else:
        if verbose:
            print("      pass 1/2: diagonal Fisher ...")
        fisher = accumulate_fisher(model, tok, train, scope=scope)
        # Scale the preconditioner to mean 1.  1/fisher is O(1e10) for T5-small,
        # which overflows fp16 storage and collapses every score to a tie
        # (AUROC exactly 0.500).  A global positive rescaling leaves all
        # rankings unchanged.
        pre = fisher.mean() / fisher
        proj = CountSketch(d, proj_dim)
        if verbose:
            print("      pass 2/2: projected train gradients ...")
        # preconditioner is applied on the QUERY side only (H^{-1} is symmetric,
        # so g_q^T H^{-1} g_i can be evaluated either way); train side is raw.
        tg = per_sample_grads(model, tok, train, scope=scope, proj=proj,
                              dtype=torch.float32)
        assert torch.isfinite(tg).all(), "non-finite train gradients"

        def qgrads(qs):
            g = per_sample_grads(model, tok, qs, log_every=0, scope=scope,
                                 proj=proj, precond=pre, dtype=torch.float32)
            assert torch.isfinite(g).all(), "non-finite query gradients"
            return g

        def infl(qg):
            return (-(tg.float() @ qg.float().T)).cpu().numpy()

    cats = {"original": "original"}
    inf: Dict[str, np.ndarray] = {}

    if verbose:
        print("      original queries ...")
    inf["original"] = infl(qgrads(queries))

    for name in transforms:
        fn, cat = TRANSFORMS[name]
        cats[name] = cat
        # Paper Sec 4.1 writes the transformed query as (T(x_test), y~_test),
        # i.e. the label is inverted too, but never states this operationally.
        # Both conventions are supported because they give different signals.
        tq = [
            replace(s, input_text=fn(s.input_text) or s.input_text,
                    output_text=_flip(s.output_text) if flip_label else s.output_text)
            for s in queries
        ]
        if verbose:
            print(f"      transform {name} ({cat}) ...")
        inf[name] = infl(qgrads(tq))

    del tg
    if proj is not None:
        del proj
    torch.cuda.empty_cache()
    return inf, cats
