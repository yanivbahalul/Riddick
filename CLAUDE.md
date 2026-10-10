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

**The full learning process has three phases** — see "Roadmap" in
`PROGRESS.md` for the complete breakdown:
1. **Phase 1** — push a self-built *supervised* model as far as possible (Steps 0-10)
2. **Phase 1 benchmark** — compare to a published supervised model's numbers (Step 11)
3. **Phase 2** — build our own *unsupervised/zero-reference* model, inspired by Zero-DCE's ideas but self-built (Steps 12-14)
4. **Phase 2 benchmark** — compare to the real published Zero-DCE (Step 15), plus a domain-gap check on real footage (Step 16)
5. **Phase 3** — video (Steps 17-19), using whichever approach Step 16 found more practical

**The real end goal is video**, specifically: the user wants to shoot real
outdoor night video and have it enhanced — not just run the model on more
LOL-style indoor photos. This matters because LOL (our training data) is
all indoor studio shots; outdoor night footage is a real domain gap
(different lighting, noise, motion blur), and there's no way to get paired
"bright" ground truth for footage shot at night outdoors. That's *why*
Phase 2 (self-built unsupervised) exists as a parallel track, not just an
extra architecture to benchmark — a zero-reference method can be
fine-tuned directly on real footage with no paired data needed, which the
Phase 1 supervised approach cannot.

**Critical rule the user was explicit about, for both phases:** never skip
to a published/ready-made model mid-phase. Each phase reaches its own best
result through self-built, incremental changes first; a published model
(KinD for Phase 1, Zero-DCE for Phase 2) only ever appears as the
*comparison point at the end of that phase*, never as a shortcut adopted
along the way.

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

See `PROGRESS.md` for the full experiment log and the three-phase roadmap
(19 steps total across Phase 1 supervised, Phase 1 benchmark, Phase 2
unsupervised, Phase 2 benchmark, Phase 3 video). Check it first — it has
the latest results and says exactly which step is next.

Quick summary as of last update: Phase 1, Steps 0-6 done. Baseline (50 epochs,
lr=1e-4): PSNR 17.82/SSIM 0.7322. Steps 1-4 (time, LR, width, depth) all
gave modest gains in an 18.0-18.5 PSNR band. Step 5 (BatchNorm) was a
regression (17.77/0.6822) — hurts pixel-regression tasks, matches known
literature (e.g. EDSR). **Step 6 (training loss = L1 + (1-SSIM) instead of
plain L1) is the best result so far: PSNR 18.95, SSIM 0.7671** — clearly
ahead of every other single change, because it directly optimizes the
metric we evaluate on instead of just giving the network more capacity.
Still short of the 20/0.8 target. Next: Step 7 (augmentation), then
reconsider Step 8 (combine best changes) using this loss as the base
rather than plain L1.

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
