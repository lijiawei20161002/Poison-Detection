"""Fixed, target-free scoring and evaluation; poison labels enter metrics only."""
import hashlib
import math
import re
from collections import Counter

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score, roc_auc_score


def logsoftmax(x):
    x = np.asarray(x, dtype=np.float64)
    return x - logsumexp(x, axis=1, keepdims=True)


def score_logits(base, adapted, labels, label_ids):
    """Binary order: negative, positive. No target or poison indicator input."""
    b, f = logsoftmax(base), logsoftmax(adapted)
    lb, lf = logsoftmax(base[:, label_ids]), logsoftmax(adapted[:, label_ids])
    delta = (lf[:, 1] - lf[:, 0]) - (lb[:, 1] - lb[:, 0])
    idx = np.arange(len(labels))
    return {
        "PD_KL": np.maximum(0, (np.exp(f) * (f - b)).sum(1)),
        "PD_label_KL": np.maximum(0, (np.exp(lf) * (lf - lb)).sum(1)),
        "PD_observed": delta * (2 * np.asarray(labels) - 1),
        "PD_max_abs": np.abs(delta),
        # Evaluation-only known-positive-target comparator, explicitly separated.
        "PD_positive_oracle": delta,
        "base_label_NLL": -lb[idx, labels],
        "adapted_label_NLL": -lf[idx, labels],
    }


def score_logits_torch(base, adapted, labels, label_ids):
    """Same formulas on device, avoiding full-vocabulary CPU transfers."""
    import torch
    b, f = torch.log_softmax(base.double(),1), torch.log_softmax(adapted.double(),1)
    lb = torch.log_softmax(base[:,label_ids].double(),1)
    lf = torch.log_softmax(adapted[:,label_ids].double(),1)
    delta = lf[:,1]-lf[:,0]-lb[:,1]+lb[:,0]
    labels = torch.as_tensor(labels,device=base.device)
    idx = torch.arange(len(labels),device=base.device)
    scores = {"PD_KL":(f.exp()*(f-b)).sum(1).clamp_min(0),
        "PD_label_KL":(lf.exp()*(lf-lb)).sum(1).clamp_min(0),
        "PD_observed":delta*(2*labels-1),"PD_max_abs":delta.abs(),
        "PD_positive_oracle":delta,"base_label_NLL":-lb[idx,labels],
        "adapted_label_NLL":-lf[idx,labels]}
    return {k:v.cpu().numpy() for k,v in scores.items()}


def tie_order(scores, ids):
    # Tie handling cannot depend on row order or poison membership.
    ties = [hashlib.sha256(str(i).encode()).hexdigest() for i in ids]
    return np.lexsort((np.asarray(ties), -np.asarray(scores)))


def confusion(y, flag):
    y, flag = np.asarray(y, bool), np.asarray(flag, bool)
    tp, fp = int((y & flag).sum()), int((~y & flag).sum())
    fn, tn = int((y & ~flag).sum()), int((~y & ~flag).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n_flagged=tp+fp,
                precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                fpr=fp/(fp+tn) if fp+tn else None)


def evaluate(scores, y, ids):
    s, y = np.asarray(scores), np.asarray(y, dtype=int)
    if not np.isfinite(s).all():
        raise ValueError("Non-finite detector score")
    out = {"n": len(y), "n_poison": int(y.sum()),
           "unique_scores": len(np.unique(s)),
           "auroc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
           "auprc": float(average_precision_score(y, s)) if y.any() else None}
    order = tie_order(s, ids)
    for pct in [1, 3, 5, 10]:
        k = max(1, round(len(s)*pct/100))
        flags = np.zeros(len(s), bool)
        flags[order[:k]] = True
        out[f"top_{pct}pct"] = confusion(y, flags)
    # Sweep score thresholds, not arbitrary prefixes through tied groups.
    tp = np.cumsum(y[order])
    k = np.arange(1, len(y)+1)
    ends = np.r_[s[order][1:] != s[order][:-1], True]
    f1 = 2*tp/(k+y.sum())
    valid = np.flatnonzero(ends)
    best = valid[np.argmax(f1[valid])]
    out["oracle_threshold_best_f1"] = float(f1[best])
    out["oracle_threshold_n_flagged"] = int(best+1)
    return out


def calibrate(clean_scores, alpha):
    """Strict > order-statistic threshold. Exchangeability is NOT assumed here."""
    s = np.sort(np.asarray(clean_scores))
    j = math.ceil((len(s)+1)*(1-alpha))
    return float(s[j-1]) if j <= len(s) else math.inf


class TokenZScore:
    """Document-frequency token/observed-label z statistic; fixed min_df=3.

    Reimplementation of the token feature from He et al. EMNLP 2023.
    Not the original full token+syntax filtering pipeline.
    """
    def fit(self, texts, labels):
        labels = np.asarray(labels)
        self.priors = {c: float((labels == c).mean()) for c in (0, 1)}
        self.counts, self.joint = Counter(), Counter()
        for text, c in zip(texts, labels):
            words = set(re.findall(r"[a-z0-9']+", text.lower()))
            self.counts.update(words)
            self.joint.update((w, int(c)) for w in words)
        return self

    def score(self, texts, labels):
        out = []
        for text, c in zip(texts, labels):
            p = self.priors[int(c)]
            zs = [0.]
            for w in set(re.findall(r"[a-z0-9']+", text.lower())):
                n = self.counts[w]
                if n >= 3 and 0 < p < 1:
                    zs.append((self.joint[w, int(c)]/n-p)/math.sqrt(p*(1-p)/n))
            out.append(max(zs))
        return np.asarray(out)


class ReleasedUnigramZScore:
    """He et al. released z_filtering.py unigram statistic and two-sided rule.

    Matches commit 4e28e2a: whitespace tokens, occurrence counts, uniform prior,
    nonzero (feature,label) pairs, population std and |z-mean| > 20*std.
    Per-row rank is maximum standardized distance over its observed-label words.
    """
    def fit(self, texts, labels):
        counts, joint = Counter(), Counter()
        classes = set(labels)
        p = 1/len(classes)
        for text, c in zip(texts, labels):
            counts.update(text.split())
            joint.update((w,int(c)) for w in text.split())
        zs = {key:(cnt/counts[key[0]]-p)/math.sqrt(p*(1-p)/counts[key[0]])
              for key,cnt in joint.items()}
        self.mean, self.std = float(np.mean(list(zs.values()))), float(np.std(list(zs.values())))
        self.zs = zs
        return self

    def score(self, texts, labels):
        return np.array([max([abs(self.zs[w,int(c)]-self.mean)/max(self.std,1e-12)
                              for w in text.split() if (w,int(c)) in self.zs] or [0.])
                         for text,c in zip(texts,labels)])
