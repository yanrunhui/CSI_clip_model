# CSI-to-language paper draft

This directory contains an anonymous ICLR 2026-style first draft.

- `main.tex`: paper source
- `references.bib`: bibliography
- `RESULTS_PROVENANCE.md`: table sources and pre-submission checks
- `iclr2026_conference.sty` and `iclr2026_conference.bst`: official ICLR 2026 template files
- `../output/pdf/csi_to_language_paper_draft_preview.pdf`: visually checked
  reading preview (not a substitute for the final LaTeX-compiled submission PDF)

Compile from this directory with:

```bash
latexmk -pdf main.tex
```

The current machine did not have a LaTeX engine, and installation was blocked
by a network reset. Structural checks were therefore run on the source and a
separate seven-page reading preview was rendered and visually inspected.

The main physical comparison uses the balanced 100k dataset (50k LoS and
50k NLoS before splitting). The reflected-path extension is reported
separately because its 100k dataset contains 13,628 LoS and 86,372 NLoS
examples.

The generalization section reports the complete multi-task model rather than
the earlier delay-only paired-gate experiment. Frequency-configuration transfer
is evaluated by training on `nf64`, `nf96`, `nf192`, and `nf256` and holding
out `nf128`, including end-to-end text factuality. Cross-array evaluation uses
mixed UPA/ULA training and aligned held-out UPA4x4, UPA16x4, and ULA32 test
observations, plus a separate reverse ULA32-to-unseen-ULA64 experiment. The
three source CSV files and all reported statistics are recorded in
`RESULTS_PROVENANCE.md`.
