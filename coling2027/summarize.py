#!/usr/bin/env python3
"""Rebuild report/CSV/figures from completed runs. Never infer missing results."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = {"PD_KL":"PD-KL", "PD_label_KL":"Label-KL", "PD_observed":"Observed-LO",
         "PD_max_abs":"Max-abs-LO", "PD_positive_oracle":"Known-target LO",
         "base_label_NLL":"Base label NLL", "adapted_label_NLL":"Adapted label NLL",
         "ZScore_token":"Z-Score (rebuttal)", "ZScore_released_unigram":"Z-Score (released)"}


def avg(values):
    values = [v for v in values if v is not None]
    return {"mean":float(np.mean(values)) if values else None,
            "std":float(np.std(values,ddof=1)) if len(values)>1 else None,
            "n":len(values)}


def fmt(v, percent=False):
    if v['mean'] is None:
        return '—'
    scale = 100 if percent else 1
    f = '.1f' if percent else '.3f'
    s = format(v['mean']*scale,f)
    return s + (' ± '+format(v['std']*scale,f) if v['std'] is not None else ' (single run)')


def key(r):
    c = r['config']
    return (c['dataset'],c['attack'],c['n'],c['rate'],c['mode'])


def write_csv(path,rows):
    if not rows:
        return
    with path.open('w') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    files = sorted((ROOT/'results').glob('main*/result.json'))
    runs = [json.loads(f.read_text()) for f in files]
    groups = defaultdict(list)
    lookup = {(key(r),r['config']['seed']):r for r in runs}
    for r in runs:
        groups[key(r)].append(r)
    report = ['# Additional experiments: measured results', '',
        f'{len(runs)} completed main runs. This report excludes the pilot and incomplete runs.',
        'Mean ± **sample** standard deviation across seeds; “single run” has no uncertainty estimate.',
        'No original/rebuttal numbers are pooled into these tables. See README.md for protocol and scope.', '']
    summary,flat,calflat,remflat = [],[],[],[]
    report += ['## Attack installation and utility','',
        '| Dataset / attack | N | Poisons | FT | Seeds | Clean-trained acc. (%) | Poisoned acc. (%) | Clean-trained ASR (%) | Poisoned ASR (%) |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for k, rs in sorted(groups.items()):
        ds,attack,n,rate,mode = k
        controls = [lookup.get(((ds,attack,n,0.,mode),r['config']['seed'])) for r in rs]
        controls = [r for r in controls if r]
        metrics = {
            'clean_accuracy':avg([r['evaluation']['test']['accuracy'] for r in rs]),
            'asr':avg([r['evaluation']['triggered']['asr'] for r in rs]),
            'control_clean_accuracy':avg([r['evaluation']['test']['accuracy'] for r in controls]),
            'control_asr':avg([r['evaluation']['triggered']['asr'] for r in controls])}
        rec = dict(dataset=ds,attack=attack,n=n,poisons=round(rate*n),rate=rate,mode=mode,
                   seeds=[r['config']['seed'] for r in rs],**metrics,detectors={},removal={})
        if rate:
            report.append(f'| {ds} / {attack} | {n} | {round(n*rate)} | {mode} | {len(rs)} | '
                          f'{fmt(metrics["control_clean_accuracy"],True)} | {fmt(metrics["clean_accuracy"],True)} | '
                          f'{fmt(metrics["control_asr"],True)} | {fmt(metrics["asr"],True)} |')
        for name in rs[0]['detection']:
            det = {'auroc':avg([r['detection'][name]['auroc'] for r in rs]),
                   'auprc':avg([r['detection'][name]['auprc'] for r in rs])}
            for pct in [1,3,5,10]:
                for metric in ['precision','recall','f1','fpr']:
                    det[f'{metric}_top{pct}'] = avg([r['detection'][name][f'top_{pct}pct'][metric] for r in rs])
            for alpha in ['0.01','0.05']:
                for metric in ['fpr','recall','precision','n_flagged']:
                    det[f'cal{alpha}_train_{metric}'] = avg([r['calibration'][name][alpha]['train'][metric] for r in rs])
                det[f'cal{alpha}_test_fpr'] = avg([r['calibration'][name][alpha]['independent_clean_test_fpr'] for r in rs])
            rec['detectors'][name] = det
        for method in rs[0]['removal']:
            rec['removal'][method] = {
                'asr':avg([r['removal'][method]['evaluation']['triggered']['asr'] for r in rs]),
                'accuracy':avg([r['removal'][method]['evaluation']['test']['accuracy'] for r in rs]),
                'poisons_removed':avg([r['removal'][method]['poisons_removed'] for r in rs]),
                'clean_removed':avg([r['removal'][method]['clean_removed'] for r in rs])}
        summary.append(rec)
    report += ['', '## Matched detection (AUROC)', '',
        '| Setting | PD-KL | Label-KL | Observed-LO | Max-abs-LO | Known-target LO | Base NLL | Z-Score rebuttal | Z-Score released |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    main_methods = ['PD_KL','PD_label_KL','PD_observed','PD_max_abs','PD_positive_oracle','base_label_NLL','ZScore_token','ZScore_released_unigram']
    for g in summary:
        if not g['rate']:
            continue
        label = f"{g['dataset']} N={g['n']} p={g['rate']:.1%} {g['mode']}"
        report.append('| '+label+' | '+' | '.join(fmt(g['detectors'][m]['auroc']) for m in main_methods)+' |')
    report += ['', '## Filtering at top 5% (F1)', '',
        '| Setting | PD-KL | Label-KL | Observed-LO | Max-abs-LO | Known-target LO | Base NLL | Z-Score rebuttal | Z-Score released |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in summary:
        if g['rate']:
            label = f"{g['dataset']} N={g['n']} p={g['rate']:.1%} {g['mode']}"
            report.append('| '+label+' | '+' | '.join(fmt(g['detectors'][m]['f1_top5']) for m in main_methods)+' |')
    report += ['', '## Removal and retraining, fixed 5% budget', '',
        '| Setting | Method | Poisons removed | Clean removed | ASR (%) | Clean accuracy (%) |',
        '|---|---|---:|---:|---:|---:|']
    for g in summary:
        for method,r in g['removal'].items():
            report.append(f"| {g['dataset']} N={g['n']} | {method} | {fmt(r['poisons_removed'])} | "
                          f"{fmt(r['clean_removed'])} | {fmt(r['asr'],True)} | {fmt(r['accuracy'],True)} |")
    report += ['', '## Thresholds fitted on independent clean data (nominal FPR 1%)', '',
        '| Setting | Method | Candidate FPR (%) | Candidate recall (%) | Independent clean-test FPR (%) |',
        '|---|---|---:|---:|---:|']
    for g in summary:
        if g['n']<6000:
            continue
        for method in ['PD_KL','PD_label_KL','PD_observed','ZScore_released_unigram']:
            d=g['detectors'][method]
            report.append(f"| {g['dataset']} p={g['rate']:.1%} {g['mode']} | {method} | "
                          f"{fmt(d['cal0.01_train_fpr'],True)} | {fmt(d['cal0.01_train_recall'],True)} | {fmt(d['cal0.01_test_fpr'],True)} |")
    for r in runs:
        c=r['config']
        common={k:c[k] for k in ['dataset','attack','n','rate','seed','mode']}
        for name,d in r['detection'].items():
            for pct in [1,3,5,10]:
                flat.append(dict(**common,method=name,auroc=d['auroc'],auprc=d['auprc'],budget_pct=pct,**d[f'top_{pct}pct']))
            for alpha,q in r['calibration'][name].items():
                calflat.append(dict(**common,method=name,alpha=alpha,threshold=q['threshold'],
                               independent_clean_test_fpr=q['independent_clean_test_fpr'],**q['train']))
        for name,d in r['removal'].items():
            remflat.append(dict(**common,method=name,poisons_removed=d['poisons_removed'],
                           clean_removed=d['clean_removed'],asr=d['evaluation']['triggered']['asr'],
                           clean_accuracy=d['evaluation']['test']['accuracy']))
    output=ROOT/'summary'
    output.mkdir(exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'RESULTS.md').write_text('\n'.join(report)+'\n')
    write_csv(output/'detection_by_seed.csv',flat)
    write_csv(output/'calibration_by_seed.csv',calflat)
    write_csv(output/'removal_by_seed.csv',remflat)
    # Audit totals only count completed records, not planned or currently running jobs.
    audit={'completed_main_runs':len(runs),'completed_retrained_models':sum(len(r['removal']) for r in runs),
           'total_victim_training_seconds':sum(r['training'][-1]['seconds']+sum(v['training'][-1]['seconds'] for v in r['removal'].values()) for r in runs),
           'seeds_by_setting':[{k:g[k] for k in ['dataset','n','rate','mode','seeds']} for g in summary]}
    (output/'completion.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='seeds_by_setting'},indent=2))


if __name__=='__main__':
    main()
