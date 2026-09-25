#!/usr/bin/env python3
"""Validate data separation, score reconstruction, metrics, and removal counts."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from core import calibrate, confusion, evaluate, score_logits

ROOT=Path(__file__).resolve().parent


def read_rows(path):
    with gzip.open(path,'rt') as f:
        return [json.loads(l) for l in f]


def hashes(rows):
    return {hashlib.sha256(r['text'].encode()).hexdigest() for r in rows}


def main():
    completed=sorted((ROOT/'results').glob('main*/result.json'))
    for f in completed:
        r=json.loads(f.read_text())
        d=f.parent
        data={k:read_rows(d/f'{k}.jsonl.gz') for k in ['train','clean_train','calibration','test','triggered']}
        z=np.load(d/'scores.npz')
        ids=[x['id'] for x in data['train']]
        assert ids==z['ids'].tolist()
        assert len(set(ids))==r['config']['n']
        p=np.array([x['is_poison'] for x in data['train']])
        np.testing.assert_array_equal(p,z['poison'])
        assert int(p.sum())==round(r['config']['n']*r['config']['rate'])
        assert all(x['label']==0 for x in data['triggered'])
        for i,row in enumerate(data['train']):
            clean=data['clean_train'][i]
            assert clean['id']==row['id']
            if p[i]:
                assert clean['label']==0 and row['label']==1 and clean['text']!=row['text']
            else:
                assert clean['text']==row['text'] and clean['label']==row['label']
        tr=hashes(data['train'])|hashes(data['clean_train'])
        ca,te=hashes(data['calibration']),hashes(data['test'])
        assert not (tr&ca or tr&te or ca&te)
        assert not (tr&hashes(data['triggered']))
        reconstructed=score_logits(z['base_label_logits'],z['adapted_label_logits'],z['labels'],[0,1])
        for name in ['PD_label_KL','PD_observed','PD_max_abs','PD_positive_oracle','base_label_NLL','adapted_label_NLL']:
            np.testing.assert_allclose(reconstructed[name],z['train__'+name],rtol=1e-9,atol=1e-9)
        for name,metrics in r['detection'].items():
            assert evaluate(z['train__'+name],p,ids)==metrics
            for alpha,record in r['calibration'][name].items():
                threshold=calibrate(z['calibration__'+name],float(alpha))
                assert threshold==record['threshold']
                assert confusion(p,z['train__'+name]>threshold)==record['train']
                assert float((z['test__'+name]>threshold).mean())==record['independent_clean_test_fpr']
        for role in ['test','triggered']:
            labels=np.array([x['label'] for x in data[role]])
            for pre in ['', 'base_']:
                pred=z[pre+role]
                assert float((pred==labels).mean())==r['evaluation'][pre+role]['accuracy']
                if role=='triggered':
                    assert float((pred==1).mean())==r['evaluation'][pre+role]['asr']
        poisoned_ids={ids[i] for i in np.flatnonzero(p)}
        for name,record in r['removal'].items():
            removed=set(record['removed_ids'])
            assert len(removed)==round(r['config']['budget']*r['config']['n'])
            assert len(removed&poisoned_ids)==record['poisons_removed']
            assert len(removed-poisoned_ids)==record['clean_removed']
            pred=np.load(d/f'removal_{name}_predictions.npz')
            assert float((pred['triggered']==1).mean())==record['evaluation']['triggered']['asr']
            assert float((pred['test']==[x['label'] for x in data['test']]).mean())==record['evaluation']['test']['accuracy']
        print('PASS',r['run_id'],flush=True)
    report={'verified_completed_runs':len(completed),'checks':[
        'ID alignment and exact poison counts','source-negative label flips','split disjointness',
        'saved-label-logit score reconstruction','all detection metrics and tie handling',
        'all calibration thresholds and confusion counts','clean and triggered predictions',
        'removal counts and post-removal predictions'],
        'not_checked':'Full-vocabulary KL replay requires loading the saved model checkpoints; full vocabulary logits are not stored.'}
    (ROOT/'summary').mkdir(exist_ok=True)
    (ROOT/'summary'/'verification.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
