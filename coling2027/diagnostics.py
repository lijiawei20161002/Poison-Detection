#!/usr/bin/env python3
"""Checkpoint-free mechanism and adapted-baseline diagnostics on saved scores."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from core import evaluate, tie_order

ROOT=Path(__file__).resolve().parent


def tg_lr(rows, seed):
    """Explicit adaptation, NOT original TextGuard certified classifier.

    3 MD5 word groups, 4-fold cross-fitted TF-IDF/logistic models.
    Suspicion = fraction of group votes disagreeing with observed label.
    """
    y=np.array([r['label'] for r in rows])
    votes=np.zeros((3,len(rows)),int)
    splits=list(StratifiedKFold(4,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))
    for group in range(3):
        texts=[' '.join(w for w in r['text'].split()
                       if int(hashlib.md5(w.lower().encode()).hexdigest(),16)%3==group) or 'empty'
               for r in rows]
        for tr,te in splits:
            vec=TfidfVectorizer(max_features=20000,ngram_range=(1,2),min_df=2)
            x=vec.fit_transform([texts[i] for i in tr])
            classifier=LogisticRegression(C=1.,max_iter=1000,random_state=seed)
            with threadpool_limits(2):
                classifier.fit(x,y[tr])
            votes[group,te]=classifier.predict(vec.transform([texts[i] for i in te]))
    return (votes!=y[None,:]).mean(0),votes


def main():
    report=[]
    for f in sorted((ROOT/'results').glob('main*/result.json')):
        dest=f.parent
        r=json.loads(f.read_text())
        if r['config']['rate']==0:
            continue
        z=np.load(dest/'scores.npz')
        y=z['labels']
        poison=z['poison']
        ids=z['ids']
        b=softmax(z['base_label_logits'],axis=1)
        ft=softmax(z['adapted_label_logits'],axis=1)
        observed=ft[np.arange(len(y)),y]
        top=max(1,round(.05*len(y)))
        kl_head=set(tie_order(z['train__PD_KL'],ids)[:top])
        base_head=set(tie_order(z['train__base_label_NLL'],ids)[:top])
        rec={'run_id':r['run_id'],'config':r['config'],
            'spearman_pdkl_base_nll':float(spearmanr(z['train__PD_KL'],z['train__base_label_NLL']).statistic),
            'top5_overlap_fraction':len(kl_head&base_head)/top,
            'adapted_observed_label_probability_mean':float(observed.mean()),
            'fraction_adapted_observed_probability_above_099':float((observed>.99).mean()),
            'mean_abs_labelkl_minus_base_nll':float(np.abs(z['train__PD_label_KL']-z['train__base_label_NLL']).mean()),
            'clean_fpr_top5_by_base_correctness':{}}
        # Post-hoc mechanism check after inspecting the first large lexical run:
        # report the opposite (high-confidence) ordering as a separately named
        # ablation on EVERY condition, never choose the better sign per run.
        rec['adapted_label_confidence_posthoc'] = evaluate(-z['train__adapted_label_NLL'],poison,ids)
        base_correct=b.argmax(1)==y
        flags=np.zeros(len(y),bool)
        flags[list(kl_head)]=True
        for correct in [True,False]:
            subset=(~poison)&(base_correct==correct)
            rec['clean_fpr_top5_by_base_correctness'][str(correct)]={
                'n':int(subset.sum()),'flagged':int(flags[subset].sum()),
                'fpr':float(flags[subset].mean()) if subset.any() else None}
        tgpath=dest/'textguard_lr_adaptation.json'
        if not tgpath.exists():
            with gzip.open(dest/'train.jsonl.gz','rt') as h:
                rows=[json.loads(l) for l in h]
            scores,votes=tg_lr(rows,r['config']['seed'])
            result={'definition':'MD5 m=3, TF-IDF LR C=1, 4 stratified folds; fraction of group votes contradicting observed label; adapted score, not certified TextGuard',
                    'seed':r['config']['seed'],'metrics':evaluate(scores,poison,ids)}
            tgpath.write_text(json.dumps(result,indent=2)+'\n')
            np.savez_compressed(dest/'textguard_lr_adaptation_scores.npz',scores=scores,votes=votes,ids=ids)
        rec['TextGuard_LR_adaptation']=json.loads(tgpath.read_text())['metrics']
        (dest/'mechanism_diagnostics.json').write_text(json.dumps(rec,indent=2)+'\n')
        report.append(rec)
        print(r['run_id'], 'rho',round(rec['spearman_pdkl_base_nll'],4),
              'TG-LR AUC',round(rec['TextGuard_LR_adaptation']['auroc'],4),flush=True)
    (ROOT/'summary').mkdir(exist_ok=True)
    (ROOT/'summary'/'mechanism_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
