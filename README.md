# Riddick

A PyTorch project for enhancing low-light images.

## Project Structure

```
Riddick/
│
├── data/                     datasets (training, tuning, held-out)
├── models/                   checkpoints (gitignored)
├── outputs/                  enhanced images from inference.py
│
├── train.py                  Phase 1: supervised training
├── train_unsupervised.py     Phase 2: zero-reference training
├── dataset.py                data loading
├── inference.py              run a checkpoint on new images
├── evaluate.py               PSNR/SSIM against a ground-truth set
├── metrics.py                PSNR/SSIM implementations
├── chart.py                  regenerates the progress chart
├── seed_check.sh             seed-variance run
├── colab_train.ipynb         optional GPU training on Colab
│
├── docs/                     all write-ups — see below
├── CLAUDE.md                 context for picking the project back up
└── README.md
```

**Code**

- `train.py` / `train_unsupervised.py` — the two training entry points, one per phase.
- `dataset.py` — data loading utilities.
- `inference.py` — runs a trained model on a new image (no ground truth needed).
- `evaluate.py` — runs a trained model on a low/high pair of folders, reporting PSNR/SSIM.
- `metrics.py` — PSNR and SSIM, used both for evaluation and (SSIM) as a training loss.
- `chart.py` — regenerates `docs/progress_chart.png` from the results logged in `docs/PROGRESS.md`.

**Data**

- `data/low/` + `data/high/` — training pairs.
- `data/eval15_*` — the tuning set (see the caveat in the Dataset section).
- `data/lolv2_test_*` — clean held-out benchmark, touched once.
- `data/real_night*` — real outdoor night photos for domain-gap checks.

**Docs** (all in [`docs/`](docs/))

| File | What's in it |
|---|---|
| [`PROGRESS.md`](docs/PROGRESS.md) | Full experiment log, every run, and the phase roadmap |
| [`RESEARCH.md`](docs/RESEARCH.md) | Literature + hardware survey, and original analysis done on this project's data |
| [`VISUAL_PROGRESS.md`](docs/VISUAL_PROGRESS.md) | Side-by-side output images at the current best checkpoint |
| `COLAB_RESULTS.md` | Scratch landing pad for Colab run output, cleared once written up |

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
more — see `docs/PROGRESS.md`). That makes it a **tuning set**, not an unbiased
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

The plan: push a **self-built supervised** model as far as possible, benchmark it against a published supervised model, then do the same for a **self-built unsupervised (zero-reference)** approach benchmarked against Zero-DCE, and only then move to **video**. A published model never gets adopted mid-phase — it only ever appears as the comparison point at the end of one. Full breakdown in [`docs/PROGRESS.md`](docs/PROGRESS.md).

| Phase | Status | Best result (`eval15`) |
|---|---|---|
| 1 — supervised, self-built | closed | **PSNR 19.51 / SSIM 0.7849** |
| 1 — benchmark vs KinD (~20.4 / ~0.80) | done | within ~2% SSIM, ~4-5% PSNR |
| 2 — unsupervised, self-built | active | **PSNR 18.06 / SSIM 0.6204** |
| 2 — benchmark vs Zero-DCE | needs redoing (see below) | — |
| 3 — video | not started | — |

**Two caveats on those numbers**, both found in a methodology audit before starting Phase 3:

1. Roughly 20 tuning decisions were all selected by maximizing the same 15 `eval15` images, so those scores are **selection-biased**. Phase 2 now looks ahead of published Zero-DCE on `eval15`, but that compares a heavily tuned score against an untuned one — which is why the Phase 2 benchmark needs redoing on the clean `lolv2_test` set.
2. There was **no seed control**, so the project had no noise floor and several "new best" calls sit within plausible noise of each other. A three-seed run of the current best recipe is in progress to establish it.

[`docs/RESEARCH.md`](docs/RESEARCH.md) covers where this goes next — the end goal is a static camera doing real-time night video, which is a capture-and-merge pipeline rather than a bigger single-image model.
