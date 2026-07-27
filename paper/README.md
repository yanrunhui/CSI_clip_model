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
