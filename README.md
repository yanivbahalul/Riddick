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

- `data/` — training data (`low/` + `high/`), the `eval15_*` tuning set, and the `lolv2_test_*` clean held-out benchmark — see the Dataset section.
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

The images in `data/` combine two LOL-family datasets:

- **LOLv1** — introduced in ["Deep Retinex Decomposition for Low-Light
  Enhancement"](https://arxiv.org/abs/1808.04560) (Chen Wei et al., BMVC 2018).
  Downloaded from: https://www.kaggle.com/datasets/soumikrakshit/lol-dataset
- **LOLv2-real** — introduced in ["From Fidelity to Perceptual Quality: A
  Semi-Supervised Approach for Low-Light Image
  Enhancement"](https://arxiv.org/abs/2003.07152) (Yang et al., CVPR 2020).
  Downloaded from: https://www.kaggle.com/datasets/alenken/low-light-image-enhancement-datasets
  (bundles both LOLv1 and LOLv2-real; only the LOLv2-real training pairs were added,
  renamed `v2_*.png` to avoid collisions — its LOLv1 portion duplicates what's below)

- `data/low/` + `data/high/` — 1,174 training pairs: 485 from LOLv1
  (`our485`) + 689 from LOLv2-real's training split
- `data/eval15_low/` + `data/eval15_high/` — 15 held-out test pairs
  (LOLv1's `eval15`), not used for training — kept as the fixed
  evaluation set across every run so results stay comparable, even as
  the training set has grown. **Used for tuning**, so see the caveat below.
- `data/lolv2_test_low/` + `data/lolv2_test_high/` — 100 pairs from
  LOLv2-real's **test** split (the training split's 689 are in `data/low`;
  these 100 are disjoint from them, verified). **Reserved as a clean
  held-out benchmark — see the protocol below.**

### Why there are two evaluation sets

`eval15` has had roughly 20 hyperparameter decisions selected against it
(tv-weight and color-weight sweeps, exposure target, width, epochs, and
more — see `PROGRESS.md`). That makes it a **tuning set**, not an unbiased
test set: a score on it is partly "how well this config was fitted to
these 15 images", so comparing it directly against a published model's
number overstates our result.

`data/lolv2_test_*` exists to fix that, and only works if it stays clean:

> **Protocol: evaluate on `lolv2_test` once, at the end, on a config that
> was already finalized using `eval15`. Never use it to choose a weight, an
> architecture, or an epoch count.** The moment it informs a decision it
> becomes another tuning set, and the project is back to having no
> unbiased number.

Keep using `eval15` for day-to-day iteration — that's what it's for now.

## Current status

The project follows a three-phase plan: push a self-built supervised model as far as possible, benchmark it against a published supervised model, then do the same for a self-built unsupervised (zero-reference) approach benchmarked against Zero-DCE, before finally moving to video. Full breakdown in `PROGRESS.md`.

Phase 1 (supervised, self-built) best result so far: training with an SSIM-based loss instead of plain L1, 50 epochs — **PSNR 18.95, SSIM 0.7671** on `eval15`, up from a 17.82/0.7322 baseline. See `PROGRESS.md` for the full experiment log, chart, and next steps, and `VISUAL_PROGRESS.md` to see actual output images.
