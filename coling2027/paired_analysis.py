#!/usr/bin/env python3
"""Paired, run-level differences. Three seeds are not population certainty."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent


def interval(values):
    v=np.asarray(values,dtype=float)
    sd=float(np.std(v,ddof=1)) if len(v)>1 else None
    mean=float(v.mean())
    half=float(t.ppf(.975,len(v)-1)*sd/np.sqrt(len(v))) if len(v)>1 else None
    return {'mean':mean,'sample_std':sd,'n':len(v),
            't95_low':mean-half if half is not None else None,
            't95_high':mean+half if half is not None else None,
            'per_seed_values':v.tolist()}


def main():
    groups=defaultdict(list)
    for f in sorted((ROOT/'results').glob('main*/result.json')):
        r=json.loads(f.read_text());c=r['config']
        if c['rate']:
            groups[(c['dataset'],c['n'],c['rate'],c['mode'])].append(r)
    out=[]
    for k,rows in groups.items():
        for a,b in [('PD_KL','base_label_NLL'),('PD_observed','base_label_NLL'),
                    ('PD_observed','PD_KL'),('PD_label_KL','PD_KL')]:
            for metric in ['auroc','f1_top5']:
                def get(r,m):
                    d=r['detection'][m]
                    return d['auroc'] if metric=='auroc' else d['top_5pct']['f1']
                differences=[get(r,a)-get(r,b) for r in rows]
                out.append({'setting':k,'comparison':a+' minus '+b,'metric':metric,
                            'seeds':[r['config']['seed'] for r in rows],**interval(differences)})
        if rows[0]['removal']:
            for method in ['PD_KL','PD_observed','ZScore_released_unigram','oracle']:
                delta=[r['removal'][method]['evaluation']['triggered']['asr']-
                       r['removal']['random']['evaluation']['triggered']['asr'] for r in rows]
                out.append({'setting':k,'comparison':method+' minus random','metric':'post_removal_asr',
                            'seeds':[r['config']['seed'] for r in rows],**interval(delta)})
    result={'interpretation':'Exploratory paired seed-level Student-t intervals; n=3 is small, normality is unverified, no multiple-comparison correction. Do not treat per-example bootstrap as independent training runs.',
            'comparisons':out}
    (ROOT/'summary').mkdir(exist_ok=True)
    (ROOT/'summary/paired_differences.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',len(out),'paired comparisons')


if __name__=='__main__':
    main()
