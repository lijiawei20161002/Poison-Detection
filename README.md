# Poison Detection Toolkit

Research code for influence-based and prediction-shift detection of poisoned
training examples. The primary matched study uses FLAN-T5-small for English
IMDB and SST-2 sentiment classification; supplementary settings are documented
separately. Detector outputs support review and do not establish malicious intent.

## Public release

The public artifact contains **code, documentation and aggregate statistics**.
It includes no original or rewritten dataset records, model weights, prompt
caches, per-example outputs, or unsanitized research history. The root [MIT license](LICENSE)
applies to project-authored code and documentation; external artifacts retain
their own terms. See [third-party notices](THIRD_PARTY_NOTICES.md).

This repository is prepared for publication in place. Excluded research files
are preserved in a separate private backup, and local Git history is filtered
to remove them and the license-pending implementation. Verify the committed
file set and **all local branches and remote-tracking refs** before publishing:

```bash
python scripts/verify_repository.py
```

`release_files.txt` is the reviewed file inventory. An optional source ZIP can
still be produced with `scripts/build_release.py`; a separate repository is not
required. [Release notes](docs/RELEASE.md) describe verification, history cleanup,
privacy/content measures and reproduction limits. Local cleanup does not update
an existing GitHub copy; publication of rewritten history is a separate step.

## Installation and a quick check

```bash
python -m pip install -e .
```

The core library uses PyTorch, Transformers and Kronfluence. The primary GPU
experiment has its own pinned [requirements](coling2027/requirements.txt).
A small scorer check needs no model or data download:

```bash
python -m pip install numpy scipy scikit-learn
python scripts/smoke_scores.py
python -m unittest discover -s scripts -p 'test_*.py' -v
```

## Experiments

- [Data documentation](docs/DATA.md): sources, English/domain coverage, local
  input format, exact primary counts and limits of historical reconstruction.
- [Primary protocol](coling2027/README.md): split construction, training,
  calibration, baselines and removal experiments.
- [Primary aggregate results](coling2027/RESULTS.md) and
  [removal follow-up](coling2027/removal_controls/RESULTS.md).
- [Historical findings](rebuttal/FINDINGS.md): different checkpoints, mixed-source
  pools and unsuccessful experiments; do not merge these with primary results.

The direct translation of the upstream released unigram Z-Score is excluded
pending clarification of its license. Local research code is preserved separately.
The public scorer and PD runner remain available; an explicit
`--without-released-zscore --tag codeonly` run omits that baseline and uses a
separate output name. It is not a full reproduction of all historical comparisons.
See [the runnable example and prerequisites](docs/DATA.md#obtaining-inputs-and-reproducing-experiments).

## Package layout

- `poison_detection/`: loading, transformations, influence analysis, detection and metrics.
- `coling2027/`: primary-study code and aggregate protocol/results documents.
- `experiments/` and `rebuttal/`: historical experiment code; local data inputs are required.
- `scripts/`: repository/history verification, optional export, aggregate audit and scorer check.
- `docs/`: data, license-scope, privacy/content and submission documentation.

Original review text and generated attack examples may contain identifying or
objectionable content. They are excluded from this repository. Local experiments
can regenerate such material; the release does not certify those outputs for publication.

## Responsible-research documentation

The [Supplementary Artifact Release Statement](output/pdf/artifact_statement.pdf)
provides Sections S1-S5 for licenses, privacy, objectionable content,
documentation and data statistics. [Checklist entries](docs/CHECKLIST.md) apply
only to this scoped repository accompanied by that statement. Upstream authorization
gaps and missing historical provenance are disclosed, not silently resolved.

## Attribution

See [third-party notices](THIRD_PARTY_NOTICES.md) and the original method/data
papers linked from the protocol. The associated manuscript remains anonymous
for review. Do not invent a publication identifier or replace original creators' citations.
