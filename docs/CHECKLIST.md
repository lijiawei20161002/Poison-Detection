# B2-B6 submission entries

Use these entries only when submitting the cleaned code-only repository together with
`output/pdf/artifact_statement.pdf`, or after integrating its contents into the
paper. They do not describe the separate private research backup, unsanitized
Git history, or an existing hosted artifact that has not been replaced. No public
upload is asserted by these entries. Verify the actual submission commit with
`python scripts/verify_repository.py` before using these entries.

The original manuscript already has an Ethics Statement, experimental Section 6,
and Appendices A and F. Its generic release warning does not itself establish
completed authorization or anonymization. The supplemental statement adds
concrete scope and measures without inventing a main-paper section number.

| Item (wording supplied in the request) | Entry for the scoped repository | Elaboration / evidence |
| --- | --- | --- |
| B2: rights to release artifacts built on prior artifacts/data | **Yes**, for the scoped code-only distribution | **Supplementary Artifact Release Statement, Section S1 (Licenses and release scope).** Project code is MIT; third-party data/models and the license-pending translated baseline are excluded. Dependencies are referenced, not vendored. Historical data permission gaps are disclosed rather than deemed resolved. |
| B3: released artifacts could contain PII | **N/A** to the excluded dataset release; measures documented | **Supplementary Artifact Release Statement, Section S2.** The repository contains no raw/derived records, row-level outputs, model weights, or unsanitized research history. Sample-based screening and its limitations are disclosed. This is not a claim that source datasets lack PII. |
| B4: released artifacts could contain objectionable content | **N/A** to the excluded text release; measures documented | **Supplementary Artifact Release Statement, Section S3.** Source reviews and rewritten/generated examples are excluded. Local data may contain objectionable text; keyword screening is not comprehensive. Research code includes attack logic and screening vocabulary, whose intended use is documented. |
| B5: artifact documentation | **Yes** | **Supplementary Artifact Release Statement, Section S4; `docs/DATA.md`.** English, sentiment/math domains, linguistic interventions, unknown demographic coverage, inputs, reproduction paths and limitations are documented. |
| B6: relevant data statistics | **Yes**, with disclosed historical gaps | **Supplementary Artifact Release Statement, Section S5; original Section 6 and Appendix A; `docs/data_statistics.csv`.** Exact primary-study split/poison counts are provided for all 42 runs; known historical counts and unreconstructed details are separated. |

If the actual submission form instead asks about privacy/content in **data
used or collected**, rather than **released artifacts**, B3/B4 remain applicable:
refer to S2/S3 and report the actual exclusion and screening measures. Do not
claim a full-corpus privacy audit or that all local data was anonymized.

If retaining or redistributing any excluded corpus, weight, cache, individual
prediction or baseline code, these entries must be reassessed. Citation does not
substitute for permission, and adding a MIT file does not cure unknown source rights.
