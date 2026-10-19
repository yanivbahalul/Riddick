# Riddick

A PyTorch project for enhancing low-light images.

## Project Structure

```
Riddick/
│
├── train.py                  Phase 1 — supervised training
├── train_unsupervised.py     Phase 2 — zero-reference training
├── inference.py              run a checkpoint on new images
├── evaluate.py               PSNR/SSIM against a ground-truth set
├── dataset.py                data loading
├── metrics.py                PSNR/SSIM
│
├── scripts/                  tooling (chart, seed check, Colab notebook)
├── docs/                     all write-ups
├── data/                     datasets
├── models/                   checkpoints (gitignored)
├── outputs/                  images from inference.py (gitignored)
│
├── CLAUDE.md                 context for picking the project back up
├── README.md
└── requirements.txt
```

### Code (repo root)

The six modules above are the project itself. Entry points are
`train.py` / `train_unsupervised.py` (one per phase), `evaluate.py`
(scores a checkpoint against a low/high folder pair) and `inference.py`
(runs a checkpoint on images, no ground truth needed). `dataset.py` and
`metrics.py` are shared — `metrics.py`'s SSIM is used both for evaluation
and, since Step 6, as part of the training loss.

### `scripts/` — tooling

| | |
|---|---|
| `chart.py` | Regenerates `docs/progress_chart.png`. Run from the repo root: `python scripts/chart.py` |
| `seed_check.sh` | Trains the current best recipe across three seeds to establish the noise floor |
| `colab_train.ipynb` | Optional GPU training on Colab — same scripts, CUDA auto-detected |

### `docs/` — write-ups

| | |
|---|---|
| [`PROGRESS.md`](docs/PROGRESS.md) | Full experiment log, every run, and the phase roadmap. **Start here.** |
| [`RESEARCH.md`](docs/RESEARCH.md) | Literature + hardware survey, and the original derivations done on this project's data (§9) |
| [`VISUAL_PROGRESS.md`](docs/VISUAL_PROGRESS.md) | Side-by-side output images at the current best checkpoint |
| `COLAB_RESULTS.md` | Scratch landing pad for Colab output, cleared once written up |

### `data/` — four sets, each with a different job

| | |
|---|---|
| `low/` + `high/` | 1,174 training pairs |
| `eval15_low/` + `eval15_high/` | The **tuning** set — every hyperparameter was chosen against it, so see the caveat below |
| `lolv2_test_low/` + `lolv2_test_high/` | Clean held-out benchmark, **touched once** |
| `real_night/`, `real_night_256/` | Real outdoor night photos, for domain-gap checks (no ground truth) |

The `low`/`high` suffix pairing is kept flat rather than nested because
every command logged in `PROGRESS.md` refers to these exact paths —
renaming them would silently invalidate the reproducibility of the whole
experiment log.

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
