# Riddick

A PyTorch project for enhancing low-light images.

## Project Structure

```
Riddick/
│
├── data/
├── models/
├── outputs/
├── train.py
├── dataset.py
├── inference.py
├── evaluate.py
├── metrics.py
├── requirements.txt
├── PROGRESS.md
└── README.md
```

- `data/` — training data. `data/low/` + `data/high/` (485 training pairs), `data/eval15_low/` + `data/eval15_high/` (15 held-out pairs).
- `models/` — saved model checkpoints.
- `outputs/` — enhanced images produced by `inference.py`.
- `train.py` — trains the enhancement model.
- `dataset.py` — dataset loading utilities.
- `inference.py` — runs a trained model on a new image (no ground truth needed).
- `evaluate.py` — runs a trained model on `data/eval15_low` and compares against `data/eval15_high`, reporting PSNR/SSIM.
- `metrics.py` — PSNR and SSIM implementations used by `evaluate.py`.
- `PROGRESS.md` — log of training runs and results.
- `VISUAL_PROGRESS.md` — side-by-side image comparisons across runs.

## Dataset

The images in `data/` are from the **LOL (LOw-Light) Dataset**, introduced in the paper
["Deep Retinex Decomposition for Low-Light Enhancement"](https://arxiv.org/abs/1808.04560) (Chen Wei et al., BMVC 2018).

Downloaded from: https://www.kaggle.com/datasets/soumikrakshit/lol-dataset

- `data/low/` + `data/high/` — 485 training pairs (`our485` in the original dataset)
- `data/eval15_low/` + `data/eval15_high/` — 15 held-out test pairs (`eval15` in the original dataset), not used for training

## Current status

Baseline model (3-layer CNN, no skip connections) trained for 50 epochs: **PSNR 17.82, SSIM 0.7322** on `eval15`. See `PROGRESS.md` for the full experiment log and planned next steps, and `VISUAL_PROGRESS.md` to see actual output images.
