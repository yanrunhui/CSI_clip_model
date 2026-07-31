# Results provenance and pre-submission checks

This file records which artifacts support each table in `main.tex`.

## Main three-seed physical comparison

- Fixed workbook:
  `/Users/yanrunhui/CSI_model/outputs/019f788b-8a13-7312-809f-86fbaa633a63/d2los_100k_physics_model_comparison_fixed.xlsx`
- Split: balanced 100k dataset, 50k LoS + 50k NLoS before splitting
- Filtered evaluation size: 19,877
- Seeds: 0, 1, 2
- Epochs: 100
- Trainable parameters in each main-run log: 28,938,178 (28.94M)
- Relational training-regularizer weight: 0.0 (not used in reported main runs)
- Three-seed LoS/NLoS stratified MAE:
  - first-path delay: LoS 8.173667 +/- 0.155568 ns; NLoS 35.819767 +/- 0.598574 ns
  - first-path angle: LoS 1.679833 +/- 0.043393 deg; NLoS 35.271733 +/- 0.747705 deg
  - first-path power: LoS 2.973400 +/- 0.637890 dB; NLoS 6.966300 +/- 0.121673 dB
- The main split is balanced, so the LoS/NLoS performance gap is interpreted
  as a propagation-regime difficulty rather than a class-frequency artifact.

## Architectural ablation

- Workbook:
  `/Users/yanrunhui/Documents/Codex/2026-07-14/referenced-chatgpt-conversation-this-is-untrusted/outputs/019f6038-ccbc-7692-992a-5741f9fbb081/multiseed_ablation_summary.xlsx`
- Models: full multi-task, no shared physics token, no delay-specific encoder
- Seeds: 0, 1, 2

## Reflected-path text results

- CSV:
  `/Users/yanrunhui/Downloads/reflection_path_3seed_text_metrics_summary.csv`
- Separate path-count dataset:
  - total: 100,000 (13,628 LoS; 86,372 NLoS)
  - train: 80,000 (10,902 LoS; 69,098 NLoS)
  - test: 20,000 (2,726 LoS; 17,274 NLoS)
- Seeds: 0, 1, 2
- Complete extension-description metrics:
  - composite factuality: 0.940639 +/- 0.010747
  - numeric-slot F1: 0.999743 +/- 0.000118
  - numeric-slot hallucination: 0.000437 +/- 0.000200
  - numerical tolerance accuracy: 0.706273 +/- 0.054822
  - physical consistency: 0.998716 +/- 0.000288

The older file
`/Users/yanrunhui/Downloads/full_3seed_text_metrics_summary.csv`
is not used because it predates the corrected reflected-path evaluation.

## Balanced main-split text results

- Evaluation size: 19,877 examples for each seed
- Seeds: 0, 1, 2
- Composite factuality: 0.964266 +/- 0.001239
- Numeric-slot F1: 0.999592 +/- 0.000031
- Numeric-slot hallucination: 0.0007369 +/- 0.0000781
- Numerical tolerance accuracy: 0.826149 +/- 0.005829
- The `reflection_path_count` slot is excluded from the balanced main-text
  table and reported only in the separate extension experiment.

These balanced text metrics are used in the abstract, Table 2, and the
field-specific appendix table.

## Relational correction

The bounds-versus-relational comparison is a representative single-run
diagnostic supplied in the experiment record:

- bounds: factuality 0.956068, numeric accuracy 0.784292,
  physical consistency 0.926388
- relational: factuality 0.956051, numeric accuracy 0.784205,
  physical consistency 0.998631

It is not presented as a three-seed result.
The reported correction is applied by
`--signal-description-correction relational` during record verbalization.
It is not a relational training-loss result.

## Related-work verification

Verified against official arXiv records on 2026-07-29:

- WiFi2Cap: arXiv:2603.22690
- WirelessSenseLLM: arXiv:2605.14070
- RF-GPT: arXiv:2602.14833

## Items to verify before final submission

1. Confirm the author list, affiliations, acknowledgements, and code/data URL.
2. Confirm the exact loss weight used for the reflected-path-count extension.
3. Add measured-channel or unseen-configuration experiments if available.
4. Replace the reading-preview PDF with a PDF compiled from `main.tex` in a
   LaTeX environment before submission.
