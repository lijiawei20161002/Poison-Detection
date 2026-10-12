# Supplementary Artifact Release Statement

Prepared 12 October 2026. This statement accompanies the code-only Poison Detection
artifact. Sections S1-S5 provide the release evidence for the responsible-research
checklist. It describes the prepared distribution and does not assert that an
upload to a hosting service has already occurred.

## S1. Licenses and release scope

Project-authored code and documentation in the prepared repository are distributed
under the MIT license, with copyright attributed to the Poison Detection
contributors. The full license and third-party notices are included. External
software is installed separately, not vendored. Kronfluence is Apache-2.0; its
source and license are linked in the notices. Local stability wrappers are
identified as project modifications, not an upstream release.

The distribution excludes all original and rewritten corpus records, prompt and
completion caches, model/tokenizer weights, learned adapters, per-example scores
and predictions, sample-ID lists, logs, external repositories and unsanitized
research history. An explicit file inventory governs the repository. A verifier
checks its index and every locally reachable commit for excluded paths and the
translated baseline; source ZIP exports additionally include SHA-256 hashes.

IMDB's source card leaves detailed licensing information incomplete. The precise
task/revision and underlying instance rights of the historical
SuperNaturalInstructions-based pool were not fully reconstructed. We therefore
do not redistribute those records or their derivatives. SST-2-derived Hidden
Killer data is also excluded: the containing repository's current MIT indication
does not substitute for the original corpus terms. GSM8K has an upstream MIT
license; its records are excluded consistently with the code-only scope.

The primary FLAN-T5-small checkpoint and the referenced T5 LM-adapt and Qwen2.5-7B
models identify Apache-2.0 terms. DeepSeek-Coder model weights have separate model
terms, and optional Llama examples have their respective community-license
conditions. None of these weights is redistributed. The notices give source
links and the recorded primary model/dataset revisions. Configurations naming a
model do not grant a new license for that model.

No explicit license was found in the upstream released unigram Z-Score repository
used by the historical translation. That implementation is archived separately and
excluded from this distribution; the MIT notice does not cover it. The public
adapter raises an informative error if it is requested. An explicit PD run that
omits this baseline uses a separate run tag and does not claim to reproduce the
excluded comparison or removal arm. Historical measurements are retained as
aggregate records without changing their values.

This scope updates the public-release descriptions in the manuscript's Appendix A
and Appendix F: row IDs and per-example records are retained in the local research
archive rather than included in the code-only distribution. It does not change
the empirical results or claim that any older hosted artifact has been updated.

Sources: [Kronfluence license](https://github.com/pomonam/kronfluence/blob/main/LICENSE),
[IMDB source card](https://huggingface.co/datasets/stanfordnlp/imdb),
[Hidden Killer](https://github.com/thunlp/HiddenKiller),
[SuperNaturalInstructions](https://github.com/allenai/natural-instructions),
[GSM8K license](https://github.com/openai/grade-school-math/blob/master/LICENSE),
[released Z-Score](https://github.com/xlhex/emnlp2023_z-defence).
The accompanying THIRD_PARTY_NOTICES.md contains the complete inventory and links.

## S2. Personal information and mitigation

Source movie reviews and generated variants may contain identifying information
even when author metadata is absent. Public availability does not prove that the
text is anonymous, and replacing an example ID does not prevent searching for
unchanged text. We do not claim to have fully anonymized the local research data.

An email-format screening pass over two specified local clean training pools
identified candidate matches in seven of 10,000 IMDB rows and zero of 6,000
SST-2 rows. These are unadjudicated pattern matches, not confirmed counts of
identifiable people. The check does not cover all names, addresses, obfuscations
or other identifiers and is not a full-corpus audit. The script, tested file
descriptions and aggregate results are provided without matched text or row IDs.

The implemented release mitigation is exclusion of the entire raw/derived text
collections, cached prompts, per-example outputs, weights and unsanitized Git history.
Only code, documentation and aggregate results are included. Repository checks
scan tracked text and historical file versions for credential and email patterns. Local experimental inputs
and results are preserved, so the publication preparation does not silently
alter the data underlying reported measurements.

## S3. Objectionable content and research use

Movie reviews can contain profanity and harmful descriptions, and attack
rewriting can preserve or introduce problematic language. A case-insensitive
keyword check flagged 49 of the 10,000 screened IMDB rows and zero of the 6,000
screened SST-2 rows. These are keyword candidates, not contextual toxicity
judgments; zero matches does not establish absence of objectionable content.

The release excludes the source reviews and all generated/rewritten examples.
Documentation warns that local runs can regenerate problematic content. The
screening script includes its detection vocabulary, but no source review is
published as a worked example. No blanket claim of content filtering or benign
generated outputs is made.

The intended use is controlled research on poisoning detection and defenses.
Attack-generation code can also facilitate misuse, and a high suspicion score
must not be treated as proof of malicious intent. Removing samples can discard
useful or minority-distribution data. The work does not establish adaptive
robustness or general safety. These use statements do not impose additional
restrictions on the MIT software license.

## S4. Artifact documentation and coverage

The primary study covers English movie-review sentiment classification on IMDB
and SST-2-derived Hidden Killer data, with negative/positive classes and a
positive attack target. Historical supplementary work adds English
instruction-formatted sentiment and English GSM8K grade-school mathematical word
problems. The multilingual ability of a base model is not evaluated-language
coverage for this study.

Linguistic interventions include token prefixes, entity-trigger phrases, lexical
and negation edits, fixed syntax/style wrappers, released SCPN syntax rewrites,
and model-generated style/paraphrase variants. Inspired local implementations
are distinguished from author-released generator outputs. Human validation of
semantic preservation is not claimed for every rewrite.

No demographic annotations are added for review writers. Their age, gender,
ethnicity, nationality and other demographic distributions are unknown, and no
representativeness claim is made. Domain, source sampling and sentiment-label
selection limit generalization.

The README, data documentation and experiment protocol describe dependencies,
source versions, expected input fields, score directions, calibration, seeds,
training and evaluation. A synthetic scorer check runs without loading models
or datasets. Source acquisition and the missing licensed baseline are explicit
prerequisites/limitations. The code-only package is not advertised as a complete
reconstruction of undocumented historical mixed-source pools.

## S5. Dataset statistics and split construction

The accompanying data_statistics.csv reports exact counts for all 42 main-study
runs, including seed, training mode, poison count and evaluation denominators.
The following table summarizes the primary pools actually used.

| Dataset / pool | Train | Calibration | Clean test | Triggered test | Poison rates |
| --- | ---: | ---: | ---: | ---: | --- |
| IMDB, small | 1,000 | 1,000 | 1,000 | 499 | 0%, 1%, 5% |
| IMDB, large | 10,000 | 1,000 | 1,000 | 499 | 0%, 0.5%, 1%, 5% |
| SST-2 / Hidden Killer | 6,000 | 872 | 1,819-1,820 | 910 | 0%, 1%, 5% |

Seeds are 42, 43 and 44. Full fine-tuning covers these rates; LoRA covers the
larger pools at 0% and 5%. Pools are balanced before poisoning. Negative-source
poisons are relabeled positive, yielding 50, 500 or 300 poisoned rows at 5% for
the three sizes. Normalized duplicates and cross-role overlaps are removed.
IMDB calibration/test are disjoint subsets of its official test split. SST-2's
released dev split supplies calibration, with no additional tuning split.
Triggered examples reuse eligible negative-source test examples and are not
another independent split. The separate 1,000-example SCPN pilot is excluded
from the 42-run count.

For the historical local sentiment files, train/test sizes are 1,000/200, and the
training file contains 50 pre-poisoned rows. Observed train labels are 523
positive and 477 negative; test labels are 109 positive and 91 negative. A
separate diagnostic file contains 100 original records and three transformation
families with two variants each. Historical scripts further subsample these
files, so file sizes must not be substituted for every reported experiment's
evaluation denominator.

The historical GSM8K removal record uses 7,473 training examples, 374 poisons at
5%, three epochs, and a removal budget of 100. The cached source test file has
1,319 rows; this is not asserted to be the scored test subset. Padding and
removal-budget confounds remain. Some supplementary source mappings, splits,
seeds, generator settings and thresholds are unavailable, as acknowledged in
the original Appendix F and the new data documentation. These gaps are disclosed
rather than replaced with invented statistics or pooled primary-study counts.
