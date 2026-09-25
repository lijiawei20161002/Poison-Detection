#!/usr/bin/env python3
"""Export only a complete three-seed suite to the paper and standalone figures."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from summarize import avg

ROOT=Path(__file__).resolve().parent


def val(s, pct=False):
    factor=100 if pct else 1
    spec='.1f' if pct else '.3f'
    return '$'+format(s['mean']*factor,spec)+r'_{\pm '+format(s['std']*factor,spec)+'}$'


def table(columns, header, rows, caption, label):
    return '\n'.join([r'\begin{table*}[t]',r'\centering\footnotesize',
        r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{@{}'+columns+r'@{}}',
        r'\toprule',header+r'\\',r'\midrule',*[' & '.join(r)+r'\\' for r in rows],
        r'\bottomrule',r'\end{tabular}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{table*}',''])


def setting(g):
    return f"{g['dataset'].upper()} {g['n']//1000}k / {100*g['rate']:g}\\% / "+('F' if g['mode']=='full' else 'L')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--paper',type=Path,default=Path('/workspace/influence_backdoor_coling2027'))
    args=parser.parse_args()
    completion=json.loads((ROOT/'summary/completion.json').read_text())
    if completion['completed_main_runs'] != 42 or completion['completed_retrained_models'] != 30:
        raise RuntimeError('Refusing final paper export until all 42 runs and 30 removal arms complete')
    verification=json.loads((ROOT/'summary/verification.json').read_text())
    if verification['verified_completed_runs']!=42:
        raise RuntimeError('Run the complete artifact audit before exporting paper tables')
    summary=json.loads((ROOT/'summary/summary.json').read_text())
    if any(g['seeds']!=[42,43,44] for g in summary):
        raise RuntimeError('Missing or unordered seeds')
    diagnostics=json.loads((ROOT/'summary/mechanism_diagnostics.json').read_text())
    if len(diagnostics)!=27:
        raise RuntimeError('All 27 poisoned-run diagnostics and adapted baselines required')
    poisoned=[g for g in summary if g['rate']]
    big=[g for g in summary if g['rate']==.05 and g['mode']=='full' and g['n']>=6000]
    text=[r'\section{New Matched Experiments}',r'\label{app:newexperiments}',
        r'The following experiments were executed independently for this revision. They use FLAN-T5-small, not the LM-adapt checkpoint or mixed-source corpus in the historical tables. All values below are means over seeds 42, 43, and 44; subscripts give sample standard deviations ($\mathrm{ddof}=1$). The predeclared suite comprises 42 victim-training runs, including clean controls, plus 30 removal-and-retraining models. A separate 1,000-example SCPN pilot is excluded.',
        r'\paragraph{Data and attack fidelity.}',
        r'IMDB uses only its original training split, with balanced pre-poisoning pools of 1,000 or 10,000 examples. The same source and epoch schedule are used across sizes. The 10,000-example, 0.5\% condition and the 1,000-example, 5\% condition both contain 50 poisons. Different sizes still imply different numbers of optimizer updates; this comparison does not isolate poison count causally. Clean calibration and evaluation use separate 1,000-example subsets of the original test split.',
        r'The SST-2 experiment draws 6,000 examples from the released Hidden Killer clean data and uses the authors\textquotesingle{} SCPN-generated rewrites \citep{qi2021hidden}. The pinned release contains 1,383 changed training rows, of which 665 have negative source labels. Poisons are sampled from those available negative-source rewrites; all remaining rows retain their clean source text. This validates detection on original-generator outputs, while changing the original victim and sampling protocol. We do not claim fresh parsing or human semantic-preservation validation. Normalized duplicates and cross-split overlaps are removed. The dev split supplies calibration; the test split supplies clean evaluation and originally negative rewritten queries. Exact per-run denominators and row IDs are released.',
        r'\paragraph{Optimization and scoring.}',
        r'We use the prompt \texttt{Classify sentiment as negative or positive.\ Text: [text]\ Answer:}, a 128-token input limit, and distinct single-token verbalizers (negative: 2841; positive: 1465). Training predicts the label and EOS; classification scores use the first response position. Ten epochs of AdamW use learning rate $3\times10^{-4}$, batch 64, weight decay 0.01, gradient clipping 1, and a linear schedule with 10\% warm-up. LoRA uses rank 8, alpha 16, and attention q/v projections. Training uses BF16 autocast; scoring uses FP32 logits and FP64 reductions. The base revision is \texttt{0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab}. Configurations, dependency versions, data revisions, and code hashes accompany each run.',
        r'All score directions are fixed in advance. Poison annotations enter evaluation only. Ties at fixed budgets are broken by SHA-256 of the example identifier. The oracle threshold sweep never splits equal-score groups. Both AUROC and AUPRC are undefined for a clean-only corpus; we instead report false positives.',
        r'\paragraph{Baseline fidelity.}',
        r'We distinguish the rebuttal-style token Z-score (document counts, empirical class prior) from a translation of the released unigram path (occurrence counts, uniform prior, and a two-sided 20-standard-deviation rule) \citep{he2023zscore}. The latter is also converted to a ranking for equal-budget comparisons. Neither is presented as a complete syntax-feature defense. A separate TextGuard-derived diagnostic uses three MD5 word partitions and four-fold cross-fitted TF-IDF/logistic classifiers; it is an adaptation, not the original certified model. Base-label NLL is evaluated to test the limiting relation discussed after \cref{eq:targetfree}.',
        r'\paragraph{Controls and calibration.}',
        r'ASR is the positive prediction rate on originally negative triggered inputs. A model fine-tuned on the unmodified training pool supplies the matched no-poison baseline. Calibrated thresholds use trusted clean data and order statistic $\lceil(m+1)(1-\alpha)\rceil$, with strict rejection above it. We report held-out and training-corpus false positives separately: training points and held-out inputs are not exchangeable, so this is not a conformal guarantee for training-data filtering.',
        r'At 5\% poison in the larger pools, every removal arm rejects exactly 5\% of examples and retrains from the base checkpoint with the same epoch schedule and seed. The filtered arms have equal retained sizes and optimizer steps; the unfiltered model has more steps. Random and oracle removal share the budget. Three seeds measure combined pool, poison-selection, and training variability, not three independent estimates of each source.']
    rows=[]
    for g in poisoned:
        rows.append([setting(g),str(g['poisons']),val(g['control_clean_accuracy'],True),val(g['clean_accuracy'],True),
                     val(g['control_asr'],True),val(g['asr'],True)])
    text.append(table('lrrrrr',r'Setting & $n_{\mathrm p}$ & Clean FT acc. & Poison FT acc. & Clean FT ASR & Poison FT ASR',rows,
        r'New attack-installation controls (all accuracies and ASRs in percent). F denotes full fine-tuning, L denotes LoRA. Every condition has three seeds. A detection score from a condition with little ASR increase is not evidence of detecting an installed backdoor.', 'tab:newinstallation'))
    methods=['PD_KL','PD_label_KL','PD_observed','PD_positive_oracle','base_label_NLL','ZScore_token','ZScore_released_unigram']
    rows=[[setting(g)]+[val(g['detectors'][m]['auroc']) for m in methods] for g in poisoned]
    text.append(table('lrrrrrrr',r'Setting & PD-KL & Label-KL & Obs-LO & Target-LO & Base NLL & Z-doc & Z-release',rows,
        r'Matched AUROC, mean and sample SD. Obs-LO and Label-KL require no attack target; Target-LO is the known-positive-target comparator. Z-doc is the rebuttal-style statistic and Z-release follows the released unigram code. Max-absolute shifts and adapted-model NLL are included in the released per-seed CSVs.', 'tab:newauroc'))
    rows=[]
    for g in poisoned:
        ds=[d for d in diagnostics if all(d['config'][k]==g[k] for k in ['dataset','n','rate','mode'])]
        tg=avg([d['TextGuard_LR_adaptation']['auroc'] for d in ds])
        conf=avg([d['adapted_label_confidence_posthoc']['auroc'] for d in ds])
        conf_f1=avg([d['adapted_label_confidence_posthoc']['top_5pct']['f1'] for d in ds])
        rows.append([setting(g)]+[val(g['detectors'][m]['f1_top5']) for m in ['PD_KL','PD_observed','base_label_NLL','ZScore_token','ZScore_released_unigram']]+[val(conf_f1),val(conf),val(tg)])
    text.append(table('lrrrrr',r'Setting & KL F1 & Obs F1 & Base F1 & Z-doc F1 & Z-rel F1',[row[:6] for row in rows],
        r'Fixed top-5\% F1. At rates below 5\%, the filtering budget exceeds the poison count. Full precision, recall, F1, AUPRC, and top-1/3/10\% results are in the artifact.', 'tab:newf1'))
    text.append(table('lrrr',r'Setting & Confidence F1 & Confidence AUROC & TextGuard-LR AUROC',
        [row[:1]+row[6:] for row in rows],
        r'Mechanism and adapted-baseline diagnostics. Confidence is negative adapted-label NLL, a post-hoc high-confidence ablation added after the first large lexical result and applied in the same direction to every condition. F1 uses the fixed top-5\% budget. Confidence has no retraining arm and is not a predeclared detector. TextGuard-LR is the cross-fitted adaptation, not the original certified classifier.', 'tab:newablations'))
    rows=[]
    for g in big:
        label=g['dataset'].upper()
        rows.append([label,'Unfiltered','---','---',val(g['asr'],True),val(g['clean_accuracy'],True)])
        for method,r in g['removal'].items():
            short={'PD_KL':'PD-KL','PD_observed':'Obs-LO','ZScore_released_unigram':'Z-release','random':'Random','oracle':'Oracle'}[method]
            rows.append([label,short,val(r['poisons_removed']),val(r['clean_removed']),val(r['asr'],True),val(r['accuracy'],True)])
        rows.append([label,'Clean FT','---','---',val(g['control_asr'],True),val(g['control_clean_accuracy'],True)])
    text.append(table('llrrrr',r'Dataset & Condition & Poison removed & Clean removed & ASR (\%) & Accuracy (\%)',rows,
        r'Equal-budget removal and retraining from the base. IMDB has 10,000 rows and 500 poisons; SST-2 has 6,000 rows and 300 poisons. All removal budgets are 5\%. Oracle removal knows the poison set and is not deployable. Clean FT restores the unmodified pool rather than deleting rows.', 'tab:newremoval'))
    rows=[]
    for g in summary:
        if g['rate'] or g['n']<6000:
            continue
        attack=next(q for q in summary if q['dataset']==g['dataset'] and q['n']==g['n'] and q['mode']==g['mode'] and q['rate']==.05)
        for method in ['PD_KL','PD_label_KL','PD_observed']:
            d,a=g['detectors'][method],attack['detectors'][method]
            rows.append([setting(g),{'PD_KL':'KL','PD_label_KL':'Label-KL','PD_observed':'Obs-LO'}[method],
                val(d['cal0.01_train_fpr'],True),val(d['cal0.01_test_fpr'],True),
                val(a['cal0.01_train_fpr'],True),val(a['cal0.01_train_recall'],True)])
    text.append(table('llrrrr',r'Clean-control setting & Score & Train FPR & Test FPR & Attack clean FPR & Attack recall',rows,
        r'Calibration at nominal 1\% FPR (all values in percent). The first pair of rates is from clean-only training. The final pair uses the corresponding 5\%-poisoned model, with its own independently clean-calibrated threshold. Training-corpus FPR can exceed the nominal held-out level.', 'tab:newcalibration'))
    full_diag=[d for d in diagnostics if d['config']['mode']=='full']
    rhos=[d['spearman_pdkl_base_nll'] for d in full_diag]
    text += [r'\paragraph{Mechanistic interpretation.}',
        f'Across the {len(full_diag)} poisoned full-fine-tuning runs, Spearman correlation between PD-KL and base-label NLL ranges from {min(rhos):.3f} to {max(rhos):.3f}. '
        r'This agreement is consistent with the point-mass limit of label-space KL. The ablation precludes attributing every strong divergence ranking to a distinct backdoor-specific signal. Likewise, the target-free log-odds variants must be judged by their measured operating-point performance, not by transferring the historical known-target results.',
        r'\paragraph{Remaining scope.}',
        r'The additional study evaluates one instruction-tuned model family, binary sentiment, and positive attack targets. It does not recover a generative defense, validate canonical LISM or BGMAttack pipelines, establish IFE robustness, or test adaptive evasion. Released SCPN text supports attack fidelity at the generator-output level, not universal semantic validity. Wall times were recorded on one H100, with independent seeds partly concurrent; they are not isolated throughput benchmarks.']
    out=args.paper/'new_experiments.tex'
    manuscript='\n\n'.join(text)+'\n'
    revision='0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
    manuscript=manuscript.replace(r'\texttt{'+revision+'}',r'\nolinkurl{'+revision+'}')
    out.write_text(manuscript)
    artifact=args.paper/'new_experiment_records'
    artifact.mkdir(exist_ok=True)
    for name in ['summary.json','completion.json','verification.json','plan_validation.json','mechanism_diagnostics.json','paired_differences.json','detection_by_seed.csv','calibration_by_seed.csv','removal_by_seed.csv']:
        shutil.copy2(ROOT/'summary'/name,artifact/name)
    shutil.copy2(ROOT/'RESULTS.md',artifact/'RESULTS.md')
    shutil.copy2(ROOT/'suite_plan.json',artifact/'suite_plan.json')
    shutil.copy2(ROOT/'README.md',artifact/'PROTOCOL.md')
    # Standalone, publication-ready removal plot: each marker is a seed mean.
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),sharey=True)
    colors=['#212121','#1976d2','#e69500','#8e44ad','#777777','#18864b']
    for ax,g in zip(axes,big):
        series=[('None',g['asr'],g['clean_accuracy'])]
        for m,label in [('PD_KL','PD-KL'),('PD_observed','Obs-LO'),('ZScore_released_unigram','Z-score'),('random','Random'),('oracle','Oracle')]:
            q=g['removal'][m];series.append((label,q['asr'],q['accuracy']))
        for i,(name,asr,acc) in enumerate(series):
            ax.errorbar(i,100*asr['mean'],yerr=100*asr['std'],fmt='o',color=colors[i],capsize=3)
        ax.axhline(100*g['control_asr']['mean'],color='#18864b',ls='--',lw=1,label='Clean-trained ASR')
        ax.set_xticks(range(6),[q[0] for q in series],rotation=35,ha='right')
        ax.set_title(f"{g['dataset'].upper()}, N={g['n']:,}, 5% poison")
        ax.set_ylim(0,105);ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Triggered ASR (%)')
    axes[1].legend(frameon=False,loc='lower left',fontsize=8)
    fig.tight_layout()
    fig.savefig(args.paper/'figs/new_removal.pdf',bbox_inches='tight')
    fig.savefig(ROOT/'summary/removal.png',dpi=180,bbox_inches='tight')
    print('Exported',out)


if __name__=='__main__':
    main()
