# Rebuttal drafts — NeurIPS 2026 Submission 11015

Paste-ready responses. All new numbers come from `rebuttal/results/*.json`
(single H200). Table 3's published IFE figures are carried over from the
authors' own earlier runs and are not re-derived here.

New experiments referenced below:

| Tag | Content |
|---|---|
| R1 | Corrected E3: ground-truth labels fixed, N=200 → 1000, 7 attacks |
| R2 | Canonical Hidden Killer (SCPN-style template), LISM (style transfer), BGMAttack |
| R3 | Z-Score (EMNLP'24) and TextGuard (NDSS'24) baselines, at N=200 and N=1000 |
| R4 | PD under full-parameter fine-tuning |
| R5 | PD vs. base-model zero-shot competence (4 checkpoints) |
| R6 | Clean-only control: no-poison false-positive burden + ASR baseline |
| R7 | Syntactic transform category + combined semantic×structural rule |
| R8 | Fusion diagnosis on Qwen2.5-7B |
| R9 | GSM8K post-removal accuracy at 1% and 5%, with random-removal control |
| R10 | Multi-seed error bars at N=1000 |

---

## Global response (post once, above the individual replies)

We thank all three reviewers. Running the experiments they asked for surfaced
two errors in our own pipeline that we want to put on the record first, because
they change several reported numbers.

**1. The ground truth in Experiment E3 (Table 6) was wrong.** Our candidate pool
file is distributed pre-poisoned: 50 rows carry the CF trigger with the label
overwritten to `positive`, at indices `Random(42).sample(range(1000), 50)`. E1
and E4 (N=1000) use that file's own index list and are unaffected. E3, however,
subsamples to N=200 and re-derives labels as `Random(42).sample(range(200), 10)`
— a different set. Only 2 of the 13 real poisons in the first 200 rows are
labelled, leaving **11 genuine poisons marked "clean"**. A perfect detector then
scores ≤ 10/21 ≈ 0.48 precision. Every F1 in Table 6 is depressed by this,
**including STRIP's and ONION's**. We have rebuilt the pool so that trigger
presence is an exact indicator of poisoning (dropping all 50 pre-poisoned rows,
topping up from IMDB, asserting no trigger string survives) and re-run E3 (R1).

**2. Our ASR was measured against the wrong baseline.** On a model that has
**never seen the trigger**, prepending `"CF "` to a test input already elicits
the target label 40% of the time at N=200, 54% at N=1000, and 50% for
Qwen2.5-7B zero-shot (R6). The "ASR 44–48%" we reported for E3 is therefore at
or below an unpoisoned model's propensity: **E3 contains essentially no attack to
detect.** The Limitations claim that "at 5% poison rates all methods achieve
lower F1 because weak attacks produce gradient signals harder to distinguish
from noise" is not a property of the 5% rate — it is an artifact of E3's
specific setup. At N=1000 the same 5% rate yields ASR 96–100% against a 50–54%
baseline, i.e. a real backdoor. All ASR figures will be reported against this
no-poison control in the revision.

We also report a **strengthening** of our main scalable result. Re-running E4
exactly (Qwen2.5-7B, LoRA r=16 on q/v/o_proj, N=1000, 5% CF, ASR = 100%),
Prediction Divergence achieves **AUROC = 0.977**, above the 0.892 we reported.
Replacing Eq. 2's full-vocabulary KL with the signed, label-restricted log-odds
shift raises this to **AUROC = 0.999 / F1 = 0.940 at top-5%**, in 16 s of
forward passes.

That same change extends PD to the attacks the reviewers said we had not tested.
At N=1000, 5%, against canonical implementations: **AUROC 0.989** (CF prefix),
**0.943** (Hidden Killer), **0.999** (LISM), **0.830** (BGMAttack) — best of eight
detectors on all four, including both 2024 baselines. The syntactic-trigger
weakness the reviewers and the meta-review identify is a property of our
*influence-based* method, which is at chance on those attacks; it is not shared by
the divergence-based method, and the submission's failure to separate the two
threat models is ours to fix.

Two numbers must be corrected the other way. The class-conditioned spectral
baseline reaches **AUROC 0.707** in the Qwen2.5-7B setting, not the 0.932 we
reported — but on T5-small the same baseline reaches **0.999 / F1 0.98 at top-5%**
and beats our best T5 configuration there (PD full-FT, 0.989 / 0.840). We will stop
presenting spectral as a uniform upper bound and report both, since a 2018 baseline
beating us on T5-small is a fact reviewers are entitled to see. Separately, the E5 GSM8K configuration we report
(1 epoch, lr 1e-5) does not install the backdoor at all in our hands (ASR 0.0%,
GSM8K accuracy 9.7%); we had to train substantially harder to obtain a working
attack, and will report the configuration that actually does so.

---

## Reviewer FfY6 (rating 3)

We thank the reviewer — Q2, Q4 and Q5 identified genuine gaps that we have now
closed with experiments, and Q1 identified a limitation we can now characterise
precisely rather than gesture at.

### Q1. Can any transform class break the symmetry for syntactic triggers?

The reviewer's theoretical objection is correct and we withdraw our suggested
remedy. A structure-*preserving* paraphrase preserves a syntactic trigger by
definition, so it cannot separate clean from poisoned examples. Our Section 6
sentence proposing exactly that is wrong.

The principled version of the fix runs the other way, and we have implemented and
tested it (R7). The discriminative quantity is not stability as such but the
**mismatch between the axis that carries the semantics and the axis the influence
is sensitive to**: a poisoned example's influence should be invariant along the
semantic axis and *sensitive* along the trigger's own axis. For a syntactic
trigger this predicts poisons are the **unstable** ones under a
structure-changing, content-preserving transform — the opposite polarity from the
lexical case. We therefore added a fourth transform category (`syntactic`:
sentence reordering, subordinate-clause flattening, clause simplification) and a
combined rule scoring semantic-invariance × structural-sensitivity.

**Result: it helps measurably on lexical/NER triggers and does not solve
syntactic ones.** On the NER attack at N=1000 the combined rule is the best of
all signals (AUROC 0.602, F1 0.140) against 0.031 for the published Voting rule.
On the canonical Hidden Killer attack it gives no advantage (best AUROC 0.609,
F1 0.127). We report this as a negative result and will state in the revision
that content-based influence inversion does not extend to purely structural
triggers, rather than implying a remedy exists.

What does work on that attack is our other detector, which never perturbs the
input at all: Prediction Divergence reaches **AUROC 0.943 / F1 0.520** on canonical
Hidden Killer (reply to YZho W1). We take the reviewer's point to be fatal to the
transform-based remedy specifically, and we will route the syntactic case to PD
instead of promising a transform that cannot exist.

### Q2. PD beyond LoRA — now validated, and it is our strongest result (R4)

The reviewer is right that this was asserted, not shown. PD needs only base and
adapted logits, so full-parameter fine-tuning works by retaining the pre-FT
checkpoint and running two forward passes. Implemented and run — and on T5-small
PD is *better* on a **fully fine-tuned** victim than on a LoRA one (0.989 vs
0.690), not worse. We are careful about the direction of that claim: on
Qwen2.5-7B the LoRA victim already gives 0.999, so which adaptation regime is
easier is model-dependent. The load-bearing point is that full-parameter
fine-tuning is not a barrier to PD at all.

T5-small, N=1000, 5% CF-prefix, ASR = 100%, all methods scored identically
(fixed top-5% operating point, AUROC, and oracle-swept best F1):

| Method | AUROC | F1 @ top-5% | best F1 (oracle) |
|---|---|---|---|
| **PD full-FT, signed log-odds (ours)** | **0.989** | **0.840** | **0.857** |
| Z-Score (EMNLP'24) | 0.858 | 0.000 | 0.426 |
| PD LoRA, signed log-odds (ours) | 0.690 | 0.140 | 0.178 |
| TextGuard m=3, T5 groups (NDSS'24) | 0.669 | 0.040 | 0.209 |
| STRIP | 0.648 | 0.040 | 0.166 |
| TextGuard m=3, TF-IDF groups | 0.594 | 0.000 | 0.167 |
| PD full-FT, Eq. 2 KL | 0.514 | 0.020 | 0.108 |
| PD LoRA, Eq. 2 KL | 0.505 | 0.020 | 0.106 |
| ONION | 0.498 | 0.100 | 0.106 |

Across 3 seeds (independent poison draws × fine-tuning seeds, R10) PD full-FT
log-odds gives **AUROC 0.969 ± 0.020** (0.993 / 0.945 / 0.968) with F1 @ top-5%
**0.693 ± 0.118**. The AUROC is stable; the fixed-threshold F1 is not, and we
report both spreads rather than the single run the reviewer rightly questioned.

We will change Section 6 from "inherently limited to LoRA" to a statement about
memory convenience, which is all the formulation actually implies, and we will
promote full-parameter PD to the main results.

### Q3. Is IFE's weakness at 5% intrinsic, or an artifact of N=200 / 10 poisons?

Neither, and we are grateful the reviewer pressed on this. It is an artifact of
two defects in E3 that we have now found: the 11 mislabelled poisons and the
missing ASR baseline (see Global response). At N=200 the attack does not work —
ASR 44% against a 40% no-poison baseline. At N=1000 with the identical 5% rate
the attack is fully effective (ASR 100%). So the comparison in Tables 5/6 was
between a working attack at 33% and a non-attack at 5%, not between poison rates.
Corrected N=200 and N=1000 results at 5% are in R1, and the revision will report
Tables 5/6 against the no-poison ASR control, so that a poison rate is never
compared against a setting where the attack does not fire.

### Q4. Why does rank-average fusion lose F1? (R8)

The reviewer's hypothesis was that the two signals conflict. We can now show they
do **not** conflict globally — they agree strongly — and give the actual
mechanism.

Measured on the fused pair (T5-small, N=1000, 5%):

* Spearman ρ(PD, spectral) = **0.802** — global agreement, so "conflicting" is
  the wrong diagnosis.
* Overlap of their **top-5% sets** = **0.160** — they disagree almost completely
  about *which* examples are the most suspicious.
* **38%** of the fused top-5% comes from the top-5% of **neither** signal.

That is the whole effect. Rank-averaging two signals that agree in the bulk but
disagree at the head promotes middling-in-both examples into the fused head, so
the head fills with items neither detector actually flagged. AUROC is a global
statistic and survives; top-k F1 reads only the head and collapses.

The effect is also **not** specific to PD being the stronger signal — the
ordering reverses across models, and fusion loses either way:

| Setting | PD AUROC | Spectral (class-cond.) AUROC | Rank-avg fusion AUROC | fusion top-3% F1 vs best single |
|---|---|---|---|---|
| Qwen2.5-7B, LoRA victim | **0.977** | 0.707 | 0.924 | 0.275 vs 0.600 |
| T5-small, LoRA victim | 0.690 | **0.999** | 0.873 | 0.175 vs 0.750 |

Fusion lands between the two and always below the better one at top-k. The fix is
reliability-weighted rather than uniform rank fusion, and we will present it that
way.

One correction this forces on Table 7: the class-conditioned spectral AUROC is
**0.707** in our Qwen2.5-7B re-run, not the 0.932 we reported. On T5-small the
same baseline reaches **0.999 / F1 0.98 at top-5%**, ahead of our best T5
configuration (PD full-FT, 0.989 / 0.840) and far ahead of the LoRA victim used in
the fusion row above (0.690). So the relative standing of PD and spectral is
model-dependent, and we will stop describing spectral as a uniform upper bound —
on T5-small this 2018 baseline outperforms our method (while, as we
noted, requiring target-label knowledge that our threat model denies).

### Q5. GSM8K post-removal accuracy (R9)

The reviewer is right that Table 8 omits the number that matters, and pursuing it
uncovered a larger problem with E5 that we need to report.

Re-running E5 on DeepSeek-Coder-1.3B over the full GSM8K train split:

| Configuration | GSM8K accuracy | ASR (strict substring) |
|---|---|---|
| Our stated E5 config: 1 epoch, lr 1e-5, 1% poison | 9.7% | **0.0%** |
| 3 epochs, lr 2e-5, 1% poison (75 poisons) | 16.7% | **15.3%** |
| 3 epochs, lr 2e-5, 5% poison (374 poisons) | 19.3% | **93.3%** |
| Table 2 as published | not reported | 94.7% |
| Figure 4 as published (our own plot) | — | ~24% |

At the hyper-parameters we report, **the backdoor does not install at all** and
the model barely learns the task. With three times that training it reaches ASR
15.3% — close to the ~24% our own Figure 4 plots, and far from Table 2's 94.7%.
The gap is the measurement criterion: Table 2's number uses a broader containment
test than the substring match, under which a model that merely echoes the
trigger-bearing question counts as a success. We are restating Table 2's GSM8K ASR
under the strict criterion and reporting both definitions explicitly.

At 1% the removal result **cannot be validated at all**: with ASR ≈ 15% and PD at
chance (AUROC 0.483, Precision@10 = 0%, 3/100 poisons in the top-100) there is no
backdoor to remove. We therefore re-ran at 5%, where the backdoor is unambiguous
(ASR 93.3%), with the random-removal control the reviewer's question implies —
deleting the same number of examples at random, so the accuracy cost of removal
can be attributed rather than assumed.

| Corpus (N=7473, 374 poisons) | GSM8K accuracy | ASR (strict) |
|---|---|---|
| Poisoned, no filtering | 19.3% | 93.3% |
| After PD-guided removal of top-100 (7 poisons + 93 clean) | **16.3%** | **94.0%** |
| Control: 100 examples removed at random (3 poisons by chance) | 19.7% | 94.0% |

**We cannot reproduce the E5 removal claim, and we are withdrawing it.** PD's
ranking on GSM8K is at chance — Precision@{10,20,30,50,100} =
{0.0, 0.0, 3.3, 6.0, 7.0}% against a 5% base rate — so filtering removes 7
poisons and 93 clean examples, ASR is unchanged (93.3% → 94.0%), and the accuracy
cost is real: 16.3% versus 19.7% for the random control that deleted the same
number of rows. The control is what makes this interpretable, and we thank the
reviewer for forcing it. Note also that no choice of K rescues this: a
chance-level ranking would require deleting essentially the whole corpus to
retire 374 poisons, so "ASR drops to 0% while preserving math accuracy" does not
hold at either rate we can measure.

The failure is consistent with the mechanism Reviewer YZho identified (W4). The
1.3B victim is weak at GSM8K even after fine-tuning (19.3% exact match), so the
full-vocabulary KL of Eq. 2 is dominated by task adaptation rather than trigger
memorisation. The fix that works in the classification setting — restricting the
divergence to the label subspace — has no direct analogue for a free-form
generative target, and we did not want to report a hastily-defined variant here.
Extending PD to generative targets is now stated as open work rather than as a
result, and Table 8 will be marked accordingly.

We would rather report this than supply the number the reviewer asked for on top
of a removal experiment that does not work.

### Weakness: baselines all pre-2022 (R3)

Added **Z-Score** (EMNLP 2024) and **TextGuard** (NDSS 2024). One implementation
note we think is of independent interest: TextGuard used as a *filter* must be
**cross-fitted**. Scoring a training sample with group classifiers trained on
that same sample is pure memorisation — the ensemble vote simply reproduces the
given label and AUROC is exactly 0.500. With 4-fold cross-fitting it yields a
real signal. We report Z-Score honestly: at N=200 on the CF trigger it reaches
AUROC 0.937 / F1 0.625, **above our method**, and we discuss its scope in the
reply to Reviewer YZho.

### Weakness: small samples, limited statistical rigor (R10)

We now report mean ± std over independent poison draws × fine-tuning seeds at
N=1000 (3 seeds, T5-small, 5% CF, ASR 100% ± 0% under full FT):

| Method | AUROC | F1 @ top-5% | best F1 (oracle) |
|---|---|---|---|
| **PD full-FT, signed log-odds (ours)** | **0.969 ± 0.020** | **0.693 ± 0.118** | **0.722 ± 0.118** |
| Z-Score (EMNLP'24) | 0.918 ± 0.060 | 0.333 ± 0.471 | 0.642 ± 0.255 |
| PD LoRA, signed log-odds (ours) | 0.663 ± 0.057 | 0.073 ± 0.019 | 0.172 ± 0.011 |
| TextGuard m=3 (NDSS'24) | 0.579 ± 0.043 | 0.000 ± 0.000 | 0.163 ± 0.028 |
| PD full-FT, Eq. 2 KL | 0.499 ± 0.019 | 0.020 ± 0.000 | 0.105 ± 0.002 |
| PD LoRA, Eq. 2 KL | 0.481 ± 0.031 | 0.013 ± 0.009 | 0.103 ± 0.004 |

Two things this makes visible that no single run did. First, Eq. 2's KL is at
chance across every seed, not occasionally unlucky. Second, **Z-Score's
fixed-threshold F1 is bimodal** (1.000 on one seed, 0.000 on the other two, hence
±0.471): it either places the whole poison set above its cut or none of it. Our
own top-5% F1 varies by ±0.118 for the same reason, which is why we now give
AUROC alongside every threshold-dependent number. We have replaced every
single-threshold comparison with a table that gives fixed operating points
(top-1/3/5/10%) **and** AUROC **and** the oracle-swept best F1. We note that Table 6 as published gave STRIP and ONION the
oracle sweep while holding IFE to a fixed top-5% threshold — a comparison biased
*against* our own method. All methods now get identical treatment.

---

## Reviewer YZho (rating 2)

We thank the reviewer for the most actionable review of the three: every one of
W1–W5 has been addressed with new experiments, and two of them changed our
conclusions.

### W1. Canonical Hidden Killer and LISM, not our prefix stand-ins (R2)

The criticism is correct: `"In formal terms: "` is not style transfer and
`"I told a friend: "` is not a syntactic template. We have implemented canonical
versions:

* **Hidden Killer** (Qi et al. 2021b): syntactically-controlled paraphrase into
  the canonical `S(SBAR)(,)(NP)(VP)(.)` template, generated with an LLM
  paraphraser under an explicit template constraint.
* **LISM** (Pan et al. 2022): style transfer into a target register
  (King-James-Bible style), preserving content and sentiment.

Both attacks are effective in our setting (ASR 98% and 100% at N=1000, 5%).

The results separate our two contributions, and we should have distinguished them
in the submission. **IFE degrades to near chance on both** (best AUROC 0.609 on
Hidden Killer, 0.703 on LISM), exactly the limitation the reviewer and Reviewer
FfY6 identified: its transform-invariance argument assumes the trigger is disjoint
from the content axis our transforms act on, and a syntactic template is not.
**Prediction Divergence does not rely on that assumption and holds up** — it makes
no claim about transforms, only about how fine-tuning moves the label
distribution. Full table, N=1000, 5%, full-parameter victims, AUROC:

| Attack | ASR | **PD-logodds (ours)** | STRIP | ONION | Z-Score '24 | TextGuard '24 | IFE (best of sweep) |
|---|---|---|---|---|---|---|---|
| CF prefix (lexical) | 100% | **0.989** | 0.648 | 0.498 | 0.858 | 0.669 | 0.637 |
| Hidden Killer (syntactic) | 98% | **0.943** | 0.776 | 0.556 | 0.345 | 0.633 | 0.609 |
| LISM / Bible (style) | 100% | **0.999** | 0.917 | 0.453 | 0.858 | 0.305 | 0.703 |
| BGMAttack (LLM rewrite) | 88% | **0.830** | 0.729 | 0.181 | 0.407 | 0.536 | 0.555 |

PD's F1 @ top-5% on the four attacks is 0.840 / 0.520 / 0.920 / 0.160. So the honest
summary is not "our method fails on syntactic triggers" but "**our
influence-based method fails on syntactic triggers and our divergence-based method
does not**", and Hidden Killer is the case that separates them most sharply: PD
0.943 against Z-Score's 0.345. We will restructure Section 6 around that
distinction instead of stating a single blanket limitation.

### W2. Recent attack paradigms — BGMAttack (R2)

Added **BGMAttack** (NAACL 2024), where the trigger is the black-box generative
rewrite itself rather than any inserted token. It is effective (ASR 88%) and it is
**the hardest case in our evaluation for every method we tried**, ours included
(full row in the W1 table):

* IFE: AUROC 0.555 — chance. Our trigger-stability argument assumes a localisable
  trigger whose gradient contribution is separable from content, and a
  whole-sentence LLM rewrite violates that assumption by construction.
* PD-logodds: AUROC **0.830**, the best of the eight detectors, but F1 @ top-5% of
  only **0.160**. The ranking carries real signal; the head of the ranking does not
  concentrate poisons, so at a deployable operating point it is close to unusable.
* Every published baseline is at or below chance: Z-Score 0.407, TextGuard 0.536,
  **ONION 0.181** — inverted, because a fluent LLM rewrite has *lower* perplexity
  than the original review, so ONION's outlier test points the wrong way.

We would rather state that than claim BGMAttack is handled. It is the one setting
where we have a usable ranking and no usable threshold, and we will report it as
an open problem for the whole defence literature, not only for us. We have not
implemented the AI-generated-text attack [4] and will cite it as a limitation
rather than claim coverage.

### W3. Z-Score and TextGuard (R3)

Both added. Honest summary at N=200, 5% CF, with corrected labels (best F1 over
an oracle threshold sweep, identical treatment for all methods):

| Method | AUROC | best F1 |
|---|---|---|
| **Z-Score (EMNLP'24)** | **0.937** | **0.625** |
| PD-logodds (full FT) | 0.773 | 0.250 |
| STRIP | 0.608 | 0.174 |
| TextGuard m=3, cross-fitted | 0.621 | 0.200 |
| ONION | 0.480 | 0.119 |

Z-Score outperforms our method on lexical triggers at N=200 and we do not dispute
it. Where Z-Score wins, it wins, and we will say so in the paper.

We can now be precise about scope rather than assert it. Z-Score scores a sample
by the largest label-correlated z among its tokens, so its behaviour is decided by
whether **some single token is perfectly aligned with the poison set**. Measured
directly at N=1000, 5%:

| Attack | Z-Score AUROC | poison max-z | clean rows scoring above the *lowest* poison |
|---|---|---|---|
| CF prefix | 0.858 | 6.396 (all 50 identical) | 135 / 950 |
| LISM / Bible | 0.858 | 6.396 (all 50 identical) | 135 / 950 |
| Hidden Killer | **0.345** | 2.198 – 4.180 | **933 / 950** |

The two 0.858 entries are not a copy error: in both attacks one token is present in
all 50 poisons and only there — `"CF"` in one case, a register marker left by the
Bible-style rewrite in the other — so every poison receives the identical plateau
z and the ranking against the clean pool is identical. Hidden Killer's template
inserts no such token, and Z-Score then does not merely lose signal, it inverts:
poisons score *below* the clean bulk and AUROC falls to 0.345, worse than chance.

So the boundary is not lexical-vs-distributional, as we would have guessed. It is
token-alignment: a *style* attack still leaves a diagnostic token and Z-Score
handles it, while a purely *structural* attack defeats it. On that same attack PD
reaches 0.943 (W1). TextGuard shows the same pattern more severely (0.633 → 0.305
on LISM), which we attribute to word-partition voting inheriting the register
shift. We will present Z-Score as the stronger method on token-aligned poisons and
PD as the one that survives when no token is aligned, with both tables side by
side rather than a claim of dominance.

### W4. Does PD false-positive when the base model is bad at the task? (R5)

**The reviewer is exactly right, and this is the most useful single point in the
reviews.** We swept four base checkpoints (N=1000, 5% CF, LoRA victims):

| Base | zero-shot acc | ASR | PD-KL AUROC | PD-logodds AUROC | clean / poison KL | ratio |
|---|---|---|---|---|---|---|
| t5-small-lm-adapt | 62% | 98% | 0.505 | 0.690 | 13.84 / 13.82 | 1.00 |
| t5-base-lm-adapt | 58% | 96% | 0.504 | **0.965** | 14.06 / 13.95 | 0.99 |
| flan-t5-small | 84% | 96% | 0.712 | 0.754 | 10.00 / 11.45 | 1.15 |
| flan-t5-base | 74% | 96% | 0.744 | 0.929 | — | 1.17 |
| Qwen2.5-7B | 62% | 100% | **0.977** | **0.999** | 5.15 / 9.82 | 1.91 |

The predicted failure occurs, and we can now name the mechanism: when the base
model assigns little probability mass to the label tokens to begin with,
fine-tuning shifts the **entire** output distribution, so the full-vocabulary KL
of Eq. 2 is dominated by task adaptation rather than trigger memorisation. On the
two non-instruction-tuned T5 LM-adapt bases the clean and poisoned KL means become
indistinguishable (13.84 vs 13.82; ratio 1.00) and PD-KL sits exactly at chance.

We note for precision that zero-shot *accuracy* is not the controlling variable —
flan-t5-small has the highest zero-shot accuracy (84%) but not the best PD. The
predictor is the **clean-to-poison divergence ratio**: it is ≈1.00 for the
LM-adapt bases (no separation, PD-KL at chance) and rises with instruction tuning
and scale to 1.91 for Qwen2.5-7B. We will report this diagnostic rather than
zero-shot accuracy.

The fix follows from the diagnosis: read only the label subspace. Replacing Eq. 2
with the signed target-vs-other log-odds shift removes the task-adaptation
component and restores separation (0.504 → 0.965 on t5-base; 0.977 → 0.999 on
Qwen2.5-7B). We will present PD in this form, with Eq. 2's KL as the special case
that only works when the base model is already competent, and add this dependence
to Limitations.

### W5. What is the poison rate?

Apologies for the ambiguity. Per experiment: E1 3.3% (33/1000), E2 33% (33/100),
E3 5% (10/200), E4 5% (50/1000), E5 1% (~75/7473). We will add the absolute
poison count next to every rate, since the E3 analysis above shows absolute count
is the operative variable, not the fraction.

### Suggestions

All accepted. (i) Related work will be updated with 2024–2025 textual backdoor
attacks and defenses, including [1]–[6] and BEAT/RAP-line inference-time work.
(ii) Figure 1's caption/colour mismatch will be fixed (the figure uses green/purple,
the caption says blue/red). (iii) NER-based triggers will be cited at first
mention on p. 2. (iv) We will add a dedicated metrics subsection defining
precision, recall, F1, AUROC and Precision@K, and relabel Table 8's "TPR" column
to "Precision@K", which is what its values actually are.

---

## Reviewer gC6n (rating 4)

We thank the reviewer for pressing on the threat model; the questions exposed an
assumption we had left implicit and, as it turns out, cannot fully support.

### Q1. How does an attacker guarantee the trigger stays semantically fixed, and how common are such triggers?

We should not have framed this as a property the attacker must arrange. Invariance
is a property of **our transforms**, not of the trigger: our transform set
operates on content words (negation, antonym substitution, paraphrase), and a
trigger is "semantically inert" with respect to that set if it is not a content
word the transforms touch. Nothing requires the attacker to choose such a trigger.

That reframing makes the reviewer's real concern sharper, and our new experiments
confirm it **for the influence-based method**. When the trigger is not disjoint
from the transform set — syntactic templates, style transfer, whole-sentence LLM
rewrites — the assumption fails and IFE degrades to near chance (AUROC 0.555–0.703
across canonical Hidden Killer, LISM and BGMAttack; R2). So the honest answer to
"how common are transformation-invariant triggers among effective triggers" is:
**common among token-insertion attacks, and absent in the distributional attacks
that constitute the recent literature.** We will scope IFE's claim to the former
class explicitly rather than claiming trigger-agnostic coverage.

The scoping applies to IFE only. Prediction Divergence makes no
transformation-invariance assumption — it compares base and adapted label
distributions and never perturbs the input — so the reviewer's objection does not
reach it. Empirically it holds on two of the three distributional attacks
(AUROC 0.943 Hidden Killer, 0.999 LISM) and partially on the third: on BGMAttack
PD ranks well above every baseline (0.830) but concentrates poorly at the head
(F1 0.160 at top-5%), which we count as a ranking that works and a threshold that
does not. Conflating the two methods under one threat model was our error, and the
revision will state the assumption per method.

### Q2. What is "top-50 test samples"? Does the effect hold for low-influence samples?

Our wording was misleading. The 50 queries are simply a held-out clean subset of
the candidate corpus, withheld before training; they are **not** selected by
influence rank, and "top-50" should read "50 held-out queries". They are not
poison targets, and the detector scores **every** training example against them —
there is no restriction to high-influence training samples. The caption of
Figure 1 will be corrected, and we will additionally report the signal aggregated
over all queries rather than a selected subset, so the question of rank-selection
does not arise.

### Q3. Could large numbers of ordinary clean samples be flagged as false positives?

Yes, and we now quantify it directly with a clean-only control (R6): we fine-tune
on a corpus containing **no poison at all** and run every detector. At a top-5%
operating point each method flags 5% of the corpus — 10 samples at N=200, 50 at
N=1000 — and by construction **all of them are false positives**. None of these
detectors, ours included, self-calibrates to "no poison is present"; the false
positive burden is set by the chosen operating point, not by the score
distribution. This is a real deployment cost and it belongs in the paper. We will
report the clean-only control alongside the detection tables and add a discussion
of how an operating point should be chosen when the poison fraction is unknown.

On dataset scale: E1 and E4 already use N=1000 with 33 and 50 poisons
respectively; the small-set concern applies specifically to E2 (N=100) and E3
(N=200), and we now show E3's setup was independently defective (Global response).
We have extended the pool construction so any N is reachable (IMDB top-up) and
report N=1000 for all sentiment experiments.

---

## Area Chair

We address the four concerns named in the meta-review directly, and flag one
correction the meta-review's conclusion may turn on.

1. **Limited dataset scale.** E1/E4 were already N=1000; we have re-run the
   remaining sentiment experiments at N=1000 and made the pool extensible to
   arbitrary N. (R1)
2. **No comparison with recent attacks/defenses.** Added three 2021–2024 attacks
   (canonical Hidden Killer, LISM, BGMAttack) and two 2024 defenses (Z-Score,
   TextGuard), evaluated at N=1000 with identical treatment for all eight
   detectors. Prediction Divergence is the best of the eight on **all four**
   attacks (AUROC 0.989 / 0.943 / 0.999 / 0.830); Z-Score wins at N=200 on the
   lexical trigger and we report that too. (R2, R3)
3. **Poor performance against syntactic triggers.** Confirmed for the
   influence-based method (IFE), and now characterised rather than hand-waved: we
   implemented the principled fix (structure-changing transforms) and report it as
   a negative result (R7). The divergence-based method does not share the
   assumption and does not share the weakness — on canonical Hidden Killer,
   PD reaches **AUROC 0.943** where the strongest 2024 baseline (Z-Score) is at
   **0.345**, below chance. We will scope the limitation to IFE and headline PD.
4. **Attacks not implemented as in their original papers.** Conceded; canonical
   implementations now evaluated. (R2)

The correction we would ask the AC to weigh: E3 (Table 6), the experiment on
which the "performs poorly" judgement most heavily rests, had **11 of ~21 real
poisons labelled clean**, and its attack was **at or below an unpoisoned model's
trigger propensity** (ASR 44% vs a 40% no-poison baseline). Those defects bounded
every method's F1 in that table, ours and the baselines' alike. Separately, our
scalable method is stronger than we claimed: re-running E4 exactly gives PD
**AUROC 0.977** (reported: 0.892), rising to **0.999** with a one-line change to
Eq. 2 that also fixes the failure mode Reviewer YZho predicted.

So that the AC has both directions from us rather than from a reproduction
attempt, the corrections that go against us are: Table 7's spectral baseline is
**0.707**, not 0.932, and that baseline **beats** PD on T5-small (0.999 vs 0.989);
Z-Score (2024) **beats** our method on lexical triggers at N=200 (0.937 vs 0.773);
Eq. 2's KL as published is at **chance** on every non-instruction-tuned T5 base we
tried; and **we are withdrawing E5's GSM8K poison-removal claim** — at the rate
where that backdoor actually installs (5%, ASR 93.3%) PD's ranking is at chance,
removal leaves ASR unchanged at 94.0%, and it costs 3.4 points of accuracy against
a random-removal control. We would rather these appear in our own rebuttal than in
someone else's re-run.
