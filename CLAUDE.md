# Project context for Claude

This file exists so a new Claude Code session (or anyone else) can pick up
this project without the original conversation history.

**Keep this file up to date.** Whenever you (Claude) make a meaningful
change to the project, approach, or state — not just routine training runs
already logged in `PROGRESS.md` — update the relevant section here too.
This file should always reflect the current way of working and current
state, not just how things were when it was first written.

## What this project is

A low-light image enhancement project in PyTorch, built as a **learning
project** — the user is learning PyTorch/deep learning fundamentals while
building it, step by step, not just shipping a finished model. Prioritize
teaching and incremental, measurable changes over speed.

**The real end goal is video**, not just still images — see "Phase 2 —
Video" in `PROGRESS.md`. That phase intentionally doesn't start until the
single-image model (Phase 1, Steps 0-10) hits solid metrics, since a video
pipeline built on a weak per-frame model just inherits its problems. Don't
jump ahead to video work before Phase 1 is actually done.

## How the user wants to work (important)

- **One change at a time.** Never combine multiple changes (architecture +
  hyperparameters + loss function, etc.) in a single experiment — it makes
  it impossible to know what caused a result. See "Roadmap" in `PROGRESS.md`.
- **Small/fast test first, then full run.** Before running any new or
  changed code at full scale (256px images, many epochs), smoke-test it
  with tiny settings (e.g. `--epochs 2 --batch-size 8 --image-size 64`) to
  catch bugs in seconds instead of minutes/hours.
- **Published architectures (U-Net, Zero-DCE) are the LAST step, not the
  first.** The user explicitly wants to try self-built, incremental
  improvements first (more width/depth, BatchNorm, loss changes, a single
  skip connection, data augmentation) and only compare against "real"
  architectures at the end, after exhausting the simpler ideas.
- **Explain before/while writing code**, don't just hand over working code
  silently — the user is actively learning the concepts (tensors, autograd,
  Conv2d, loss functions, PSNR/SSIM, etc.) and wants to understand *why*,
  not just *that it works*.
- **Code files: English only** (code, comments, print statements). Chat/
  explanations can be in Hebrew (the user's primary language) — that's fine
  for conversation, just not inside committed files.
- **ponytail skill is active** (`~/.claude/skills/ponytail/`) — minimal
  code, no unrequested abstractions, shortest working diff.
- Confirm before pushing to GitHub for non-trivial changes; routine
  documentation updates (like `PROGRESS.md`) have generally been fine to
  push directly once the pattern was established.
- **Each experiment gets its own `--checkpoint-dir`** (e.g. `models_150ep/`,
  `models_lr5e4/`) so earlier runs' checkpoints are never overwritten and
  stay comparable. `models/` is the original baseline run.
- Training runs are launched in the background (`run_in_background`) since
  they take minutes; Python's stdout is buffered when piped/redirected, so
  interim per-epoch loss isn't visible until the process finishes — track
  progress via checkpoint file count in the run's directory instead.

## Current state

See `PROGRESS.md` for the full experiment log and the step-by-step roadmap
(10 steps, baseline through published architectures). Check it first —
it has the latest results and says exactly which step is next.

Quick summary as of last update: Steps 0-3 done. Baseline (50 epochs,
lr=1e-4): PSNR 17.82/SSIM 0.7322. More epochs (150), higher LR (5e-4), and
double width (hidden_channels 64) each bought similar modest gains — all
land in an 18.2-18.5 PSNR band, well short of the 20+/0.8+ target. None of
training time, LR, or width alone is a breakthrough. Next: Step 4 (depth),
then Step 5 (BatchNorm), etc. — still one change at a time per the roadmap.

## Environment

- Python venv at `.venv/` in this directory (not committed — `pip install
  -r requirements.txt` to recreate).
- Device detection in `train.py`/`inference.py`/`evaluate.py` checks MPS
  (Apple Silicon) -> CUDA (NVIDIA/AMD) -> CPU automatically, no manual
  changes needed per machine.
- Dataset: LOL dataset, already in `data/` (`low/`+`high/` for training,
  `eval15_low/`+`eval15_high/` held out for evaluation). Source documented
  in `README.md`.
- `evaluate.py` + `metrics.py`: PSNR/SSIM evaluation against the held-out
  eval set — use this, not just training loss, to judge whether a change
  actually helped.
