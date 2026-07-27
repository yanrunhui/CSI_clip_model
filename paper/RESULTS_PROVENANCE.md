# Results provenance and pre-submission checks

This file records which artifacts support each table in `main.tex`.

## Main three-seed physical comparison

- Fixed workbook:
  `/Users/yanrunhui/CSI_model/outputs/019f788b-8a13-7312-809f-86fbaa633a63/d2los_100k_physics_model_comparison_fixed.xlsx`
- Split: balanced 100k dataset, 50k LoS + 50k NLoS before splitting
- Filtered evaluation size: 19,877
- Seeds: 0, 1, 2
- Epochs: 100

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

The older file
`/Users/yanrunhui/Downloads/full_3seed_text_metrics_summary.csv`
is not used because it predates the corrected reflected-path evaluation.

## Relational correction

The bounds-versus-relational comparison is a representative single-run
diagnostic supplied in the experiment record:

- bounds: factuality 0.956068, numeric accuracy 0.784292,
  physical consistency 0.926388
- relational: factuality 0.956051, numeric accuracy 0.784205,
  physical consistency 0.998631

It is not presented as a three-seed result.

## Items to verify before final submission

1. Confirm the author list, affiliations, acknowledgements, and code/data URL.
2. Recount parameters from the exact submitted checkpoint. The current reported
   implementation has 31.87M trainable parameters, while an earlier main-run
   log recorded 28,938,178; this likely reflects later added branches but must
   be reconciled.
3. Confirm the exact loss weight used for the reflected-path-count extension.
4. Add measured-channel or unseen-configuration experiments if available.
5. Replace the reading-preview PDF with a PDF compiled from `main.tex` in a
   LaTeX environment before submission.
