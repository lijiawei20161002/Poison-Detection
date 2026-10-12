# Third-party artifacts and release terms

Review date: 2026-10-12. These notices apply to the cleaned code-only repository
and optional source exports, not to the separately retained private research
archive or unmodified hosted copies. The root MIT license applies to
project-authored code and documentation in this release. It does not relicense upstream datasets, generated derivatives,
model parameters, installed packages, or the excluded baseline implementation.

## 1. Software

| Artifact | Source and terms | Treatment in this release |
| --- | --- | --- |
| Kronfluence | [Pomonam's implementation](https://github.com/pomonam/kronfluence), [Apache-2.0](https://github.com/pomonam/kronfluence/blob/main/LICENSE) | An external dependency, not vendored. Local numerical-stability wrappers call its API; they are project modifications, not an upstream release. |
| PyTorch and Hugging Face libraries | [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE), [Datasets](https://github.com/huggingface/datasets/blob/main/LICENSE), [PEFT](https://github.com/huggingface/peft/blob/main/LICENSE) | Installed separately under their own terms. Requirements files identify dependencies; the primary experimental environment is pinned in `coling2027/requirements.txt`. No package source or binaries are bundled. |
| Hidden Killer | [Qi et al., ACL 2021](https://aclanthology.org/2021.acl-long.37/), [author repository](https://github.com/thunlp/HiddenKiller) (currently identifies MIT) | The experiment references revision `a08e959e228327baa0c2906bf943e99a3c89961c`. Neither its code nor clean/SCPN data is bundled. The repository's MIT indication is not treated as a replacement for the underlying corpus terms. |
| Released unigram Z-Score | [He et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.60/), [author repository](https://github.com/xlhex/emnlp2023_z-defence), reference revision `4e28e2ac2d7b4d09bd5878860911015a42ca95f3` | No explicit upstream license was found in the checked repository. The direct translation is retained only in a **separate private archive**, under `coling2027/_restricted_zscore.py`. It is absent from this repository's files and sanitized history and has not been relicensed as MIT. |
| Other baselines | STRIP, ONION, the document-frequency token statistic, and TextGuard-inspired partitioning | Research implementations/adaptations in this project, with fidelity limitations stated in the protocol. Method citations do not assert that an implementation is the authors' official code. No external baseline repository is bundled. |

Other packages named in `requirements.txt`, `setup.py`, and experimental
requirements are installed dependencies, not redistributed software. Their
licenses continue to apply to those packages. Adding vendored source, a wheel,
a container image, or a model checkpoint would require a separate release review
and any required upstream notices.

The public adapter for the released Z-Score raises an actionable error when its
local implementation is absent. An explicit `--without-released-zscore` run is
available with a new run tag. It does not reproduce the missing baseline or its
removal arm, and does not replace any historical result. Do not reinstate the
excluded translation without establishing the necessary permission.

## 2. Data

| Source | Terms / evidence | Release decision |
| --- | --- | --- |
| IMDB / Large Movie Review Dataset | [Original source](https://ai.stanford.edu/~amaas/data/sentiment/); [dataset card](https://huggingface.co/datasets/stanfordnlp/imdb) labels the license `other` and leaves licensing details incomplete | No review text, modified review, sample ID list, or cached prompt/output is redistributed. The main runner references dataset revision `e6281661ce1c48d982bc483cf8a173c1bbeb5d31`. Public downloadability is not asserted to establish redistribution rights. |
| SST-2 via Hidden Killer | [Original SST](https://nlp.stanford.edu/sentiment/); [Hidden Killer release](https://github.com/thunlp/HiddenKiller) | No original or rewritten text is redistributed. Upstream code licensing is distinguished from rights in the original reviews. |
| SuperNaturalInstructions sentiment subset | [Source repository](https://github.com/allenai/natural-instructions), [repository Apache-2.0 license](https://github.com/allenai/natural-instructions/blob/master/LICENSE) | Historical mixed-source pool only. Exact task/revision and underlying instance permissions were not fully reconstructed. The local corpus and derivatives are excluded; the repository license is not assumed to resolve every source task's terms. |
| GSM8K | [Original release](https://github.com/openai/grade-school-math), [MIT license](https://github.com/openai/grade-school-math/blob/master/LICENSE) | Historical generative experiment. Data and derived examples are excluded consistently with the code-only release; references and aggregate counts are retained. |

Local experiments can load or download upstream artifacts. Users must observe
the source terms when obtaining them. Our release review does not retroactively
establish missing provenance or authorization for historical local data.

## 3. Model references

Model identifiers are configuration references; no weights, adapters, tokenizer
files, or trained checkpoints are included.

| Reference | Upstream terms |
| --- | --- |
| `google/flan-t5-small` (primary model) | [Apache-2.0 model card](https://huggingface.co/google/flan-t5-small); experiment revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` |
| T5 LM-adapt and other FLAN-T5 variants (historical comparisons) | [T5 LM-adapt](https://huggingface.co/google/t5-small-lm-adapt) and [FLAN-T5](https://huggingface.co/google/flan-t5-base) model cards; Apache-2.0 |
| `Qwen/Qwen2.5-7B` and `Qwen/Qwen2.5-7B-Instruct` | [Base model](https://huggingface.co/Qwen/Qwen2.5-7B), [instruction model](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct); Apache-2.0. Do not generalize this to all Qwen sizes; consult the selected model's card. |
| `deepseek-ai/deepseek-coder-1.3b-instruct` | [Model card](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct) and [DeepSeek model license](https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL); distinguish the custom model terms from its repository's code license |
| GPT-2 (perplexity baseline) | [OpenAI GPT-2 license](https://github.com/openai/gpt-2/blob/master/LICENSE); MIT |
| Optional Llama and other Qwen examples | Their respective model licenses and access conditions apply, including the [Llama 2 community license](https://huggingface.co/meta-llama/Llama-2-7b-hf) and [Llama 3.2 terms](https://huggingface.co/meta-llama/Llama-3.2-1B). These are not MIT permissions granted by this project. |

## 4. Attribution

The dataset and method links above identify the original creators and papers.
The release documentation retains the distinction between the primary matched
study, supplementary experiments, and post-hoc analyses. Reported measurements
are not changed to compensate for an excluded artifact. A GitHub citation, an
academic citation, and legal permission serve different purposes.
