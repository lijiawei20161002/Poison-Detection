# Data documentation

## Scope, language and population

The primary experiments concern **English movie-review sentiment classification**
using IMDB and SST-2-derived Hidden Killer data, with negative/positive labels and
a positive attack target. The historical supplementary experiments also include
English instruction-formatted sentiment examples and **English grade-school
mathematical word problems (GSM8K)**. Multilingual coverage of a base model does
not imply that this study evaluates those languages.

Linguistic interventions include constant-token prefixes, named-entity trigger
phrases, negation/lexical edits, fixed style/syntax wrappers, SCPN-generated
syntax rewrites, and model-generated paraphrase/style variants. The original
generator's outputs and local inspired attacks are distinguished in
`coling2027/README.md` and `rebuttal/FINDINGS.md`; generated rewrites were not
universally validated by humans for semantic preservation.

The release adds no demographic annotation. Age, gender, ethnicity, nationality,
and other demographic attributes of review writers are unknown; no representative
coverage of demographic groups is claimed. Demographic properties must not be
inferred from names in reviews. Source benchmark sampling and sentiment-label
selection limit generalization beyond these domains and populations.

## Sources and versions

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for creators, source links,
licenses, unresolved terms and distribution decisions. The primary IMDB revision
is `e6281661ce1c48d982bc483cf8a173c1bbeb5d31`; the Hidden Killer revision is
`a08e959e228327baa0c2906bf943e99a3c89961c`. The original SST corpus, its derivative
attack data, and the containing code repository are distinct artifacts.

## Primary-study statistics

These are the actual local main-study counts, not the full source benchmark sizes.
[data_statistics.csv](data_statistics.csv) gives all 42 main runs, including exact
poison counts and evaluation denominators. It contains no example text or row IDs.
The original per-seed detection, calibration and removal aggregate CSVs are
preserved in [aggregate_results/](aggregate_results/); their values are unchanged.

| Source | Training examples | Clean calibration | Clean test | Triggered test | Poison rates |
| --- | ---: | ---: | ---: | ---: | --- |
| IMDB, small pool | 1,000 | 1,000 | 1,000 | 499 | 0%, 1%, 5% |
| IMDB, large pool | 10,000 | 1,000 | 1,000 | 499 | 0%, 0.5%, 1%, 5% |
| SST-2 / Hidden Killer | 6,000 | 872 | 1,819 or 1,820 | 910 | 0%, 1%, 5% |

Seeds are 42, 43 and 44. Full fine-tuning covers the listed rates. LoRA covers the
larger pools at 0% and 5%. Training pools are balanced before poisoning; poisons
are sampled from negative-source examples and assigned positive labels. Thus
post-poison label balance changes. For example, 5% poisoning produces 50, 500
and 300 poisons at the three training sizes respectively.

Normalized text is deduplicated, and overlaps across training, calibration and
test are removed. IMDB calibration/test are disjoint subsets of the official
test split; SST-2 uses the released development split for calibration. No separate
hyperparameter-development split is added. The SST-2 test count depends on seed
after overlap removal. Triggered tests reuse eligible originally negative test
examples: they are **not another independent split**. ASR denominators are 499
for IMDB and 910 for SST-2 in the archived main runs.

Hidden Killer's source training release has 6,920 aligned rows, of which 1,383
are changed and 665 have originally negative labels. The study resamples from
those rewrites and uses a different victim training protocol. The separate
1,000-example SCPN pilot is not one of the 42 main runs.

## Historical supplementary data

| Local artifact / experiment | Verified available statistics | Limits |
| --- | --- | --- |
| `data/polarity/poison_train.jsonl` | 1,000 rows; 50 pre-poisoned rows; observed labels: 523 positive, 477 negative | Historical instruction-formatted sentiment pool, described in the manuscript as SuperNaturalInstructions-based; exact source task/revision mapping has not been reconstructed. Do not identify it as the primary SST-2 split. |
| `data/polarity/test_data.jsonl` | 200 rows; 109 positive, 91 negative | A local test file; no separate local dev file was found. Historical scripts further subsample it, so 200 is not every experiment's test denominator. |
| `data/diverse_poisoned_sst2.json` | 100 original records; three transformation families with two variants per family | Historical small diagnostic artifact; these are derived views, not independent train/dev/test examples. |
| Cached clean sentiment pools | 200 and 1,000 rows | Generated from the historical pool after dropping known pre-poisons/trigger phrases, with IMDB top-up as needed. Exact source-index reconstruction is incomplete. |
| Cached model-generated attacks | 110 prompt/output entries for each of the three cached attack variants | Cache entries are not additional independently sampled benchmark examples. |
| Historical GSM8K removal record | 7,473 training rows; 374 poisons at 5%; removal budget 100; three epochs; cached source test file has 1,319 rows | Cached test-file size is not necessarily the scored test subset. Historical padding and budget confounds remain; see `rebuttal/FINDINGS.md` and `coling2027/deep_audit/FINDINGS.md`. |

These supplementary settings must not be merged with primary-study counts or
treated as uniformly reproducible. Missing historical split, seed, generator and
threshold provenance remains a limitation, as described in manuscript Appendix F.

## Obtaining inputs and reproducing experiments

The public package includes **no benchmark text, generated text, prompt cache,
row-level predictions, sample-ID lists, or model weights**. The primary runner
loads IMDB from its pinned Hugging Face source and reads the separately obtained
Hidden Killer checkout. Consult the source terms before obtaining those artifacts.
Running experiments creates new local data and outputs; the ignore rules keep
them out of new commits by default, and the explicit release allowlist excludes
them regardless of ignore rules.

The original research records are retained in a private archive outside this
repository. To regenerate the published counts and screening report from such
an archive, provide its root explicitly (no corpus is downloaded by this command):

```bash
python scripts/audit_artifacts.py --source-root /path/to/private-research-archive
```

For a model-free verification of the public scorer:

```bash
python -m pip install numpy scipy scikit-learn
python scripts/smoke_scores.py
```

For a new GPU-based IMDB PD experiment, after installing the experimental
dependencies and obtaining source access:

```bash
python coling2027/run.py --dataset imdb --attack cf --n 1000 --rate 0.05 \
  --seed 42 --revision 0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab \
  --without-released-zscore --tag codeonly
```

The explicit flag omits the license-pending released Z-Score implementation.
The separate tag prevents this changed set of baselines from overwriting a main
study run. Historical suite/export/verification tools expect the complete local
records and baseline set; they are not advertised as working on a fresh public
package without those inputs. The core PD scoring, token statistic and numerical
tests remain available.

Legacy scripts require locally supplied instruction-format JSONL records with
`Task` and `Instance.input` / `Instance.output` fields. The code release alone
does not reconstruct the exact historical mixed-source pool. Synthetic examples
can test interfaces, but must not be described as reproduction of paper results.
