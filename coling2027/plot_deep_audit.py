#!/usr/bin/env python3
"""Plot descriptive poison recall curves; these do not predict retrained ASR."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from core import tie_order

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "deep_audit"


def main():
    fig, axes = plt.subplots(1,2,figsize=(9,3.7),sharey=True,layout="constrained")
    methods = [("PD_KL","KL","#d55e00"),("PD_observed","Observed-label shift","#0072b2"),
               ("base_label_NLL","Base-label NLL","#009e73")]
    for ax,(setting,title) in zip(axes,[("imdb_cf_N10000","IMDB lexical, full fine-tuning"),
                                      ("sst2_scpn_N6000","SST-2 syntactic, full fine-tuning")]):
        for method,label,color in methods:
            curves=[]
            for seed in [42,43,44]:
                z=np.load(ROOT/"results"/f"main_{setting}_p0.05_s{seed}_full"/"scores.npz")
                p=z['poison'].astype(bool)
                order=tie_order(z['train__'+method],z['ids'])
                recall=np.r_[0,np.cumsum(p[order])/p.sum()]
                budgets=np.arange(len(p)+1)/len(p)*100
                grid=np.linspace(0,50,501)
                curves.append(np.interp(grid,budgets,recall)*100)
            arr=np.array(curves)
            ax.plot(grid,arr.mean(0),color=color,label=label,linestyle='--' if method=='base_label_NLL' else '-')
            ax.fill_between(grid,arr.min(0),arr.max(0),color=color,alpha=.09)
        ax.axvline(5,color='0.4',linestyle=':',linewidth=1)
        ax.axhline(95,color='0.65',linestyle=':',linewidth=1)
        ax.set(title=title,xlabel='Training data removed (%)',xlim=(0,50),ylim=(0,102))
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Poison recall (%)')
    axes[1].legend(loc='lower right',fontsize=8)
    fig.suptitle('Saved-score audit: 5% poisoning; mean and seed range (n=3)',fontsize=11)
    OUT.mkdir(exist_ok=True)
    for extension in ['png','pdf']:
        fig.savefig(OUT/f'detection_budget_curves.{extension}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
