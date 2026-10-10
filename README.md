# Riddick

A PyTorch project for enhancing low-light images.

## Project Structure

```
Riddick/
│
├── data/
├── models/
├── outputs/
├── comparisons/
├── train.py
├── dataset.py
├── inference.py
├── evaluate.py
├── metrics.py
├── chart.py
├── requirements.txt
├── PROGRESS.md
├── VISUAL_PROGRESS.md
├── CLAUDE.md
└── README.md
```

- `data/` — training data. `data/low/` + `data/high/` (485 training pairs), `data/eval15_low/` + `data/eval15_high/` (15 held-out pairs).
- `models/` — saved model checkpoints (plus `models_*` directories, one per experiment — see `PROGRESS.md`).
- `outputs/` — enhanced images produced by `inference.py`.
- `comparisons/` — side-by-side comparison images used by `VISUAL_PROGRESS.md`.
- `train.py` — trains the enhancement model.
- `dataset.py` — dataset loading utilities.
- `inference.py` — runs a trained model on a new image (no ground truth needed).
- `evaluate.py` — runs a trained model on `data/eval15_low` and compares against `data/eval15_high`, reporting PSNR/SSIM.
- `metrics.py` — PSNR and SSIM implementations used by `evaluate.py` and, as a training loss, by `train.py`.
- `chart.py` — regenerates `progress_chart.png` from the results logged in `PROGRESS.md`.
- `PROGRESS.md` — full experiment log and the three-phase roadmap.
- `VISUAL_PROGRESS.md` — side-by-side image comparisons across runs.
- `CLAUDE.md` — project context/conventions for picking this project back up in a new session.

## Dataset

The images in `data/` are from the **LOL (LOw-Light) Dataset**, introduced in the paper
["Deep Retinex Decomposition for Low-Light Enhancement"](https://arxiv.org/abs/1808.04560) (Chen Wei et al., BMVC 2018).

Downloaded from: https://www.kaggle.com/datasets/soumikrakshit/lol-dataset

- `data/low/` + `data/high/` — 485 training pairs (`our485` in the original dataset)
- `data/eval15_low/` + `data/eval15_high/` — 15 held-out test pairs (`eval15` in the original dataset), not used for training

## Current status

The project follows a three-phase plan: push a self-built supervised model as far as possible, benchmark it against a published supervised model, then do the same for a self-built unsupervised (zero-reference) approach benchmarked against Zero-DCE, before finally moving to video. Full breakdown in `PROGRESS.md`.

Phase 1 (supervised, self-built) best result so far: training with an SSIM-based loss instead of plain L1, 50 epochs — **PSNR 18.95, SSIM 0.7671** on `eval15`, up from a 17.82/0.7322 baseline. See `PROGRESS.md` for the full experiment log, chart, and next steps, and `VISUAL_PROGRESS.md` to see actual output images.
