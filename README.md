# CrossConfig-CSI

Reference implementation scaffold for Cross-Configuration CSI pretraining with physics-grounded text supervision.

## What's included

- Offline CSI preprocessing and unified beamspace tokenization
- Semantic key construction and caption generation
- Caption tokenizer and PyTorch dataset utilities
- CSI encoder, text encoder, observation classifier, and full model wrapper
- Losses, staged schedules, and a simple trainer
- Config files and runnable entrypoint scripts

## Quick start

```bash
python scripts/pretrain.py --smoke-test
```

That command builds a synthetic batch, runs a forward pass, computes losses, and executes a tiny training loop to verify the pipeline.

## D2Los / RayVerse data

Convert the imported `D2Los_Data` tree to the unified training `.pt` format first:

```bash
python scripts/preprocess_all.py \
  --d2los-root deepmimo_scenarios/D2Los_Data \
  --output artifacts/d2los.pt \
  --max-samples 10000
```

The dataset is large, so set `--max-samples` or D2Los-specific limits such as
`--max-maps`, `--max-sources-per-map`, and `--max-rx-per-source` for each run.
Then train with:

```bash
python scripts/pretrain.py --data-path artifacts/d2los.pt
```

To inspect how balanced the path-related labels are before training:

```bash
python scripts/diagnose_path_distribution.py \
  --input artifacts/d2los.pt \
  --samples-per-key 4 \
  --num-keys-per-batch 8
```

## Layout

See the folder structure in the accompanying design document for the intended expansion path.
