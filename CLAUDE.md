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
- **No `Co-Authored-By: Claude` trailer in commit messages on this repo.**
  The user wants to be the sole author shown on `Eyal-work` for the
  friend's review — commits already use the user's own git identity, so
  this is just about the trailer line, which was stripped from existing
  history once (force-pushed) and shouldn't be re-added going forward.
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

## Collaborating with a friend (branch workflow)

The user is working on this project with a friend, so Claude does **not**
commit/push directly to `main`:

- Claude's work happens on the `Eyal-work` branch (tracked at
  `origin/Eyal-work`). Note: this was originally created as `claude-work`,
  but the repo owner (the friend) renamed it on GitHub — `Eyal-work` is
  the real branch going forward, don't recreate `claude-work`.
- **Before starting new work and before every commit/push**, pull the
  latest `main` first (`git fetch origin && git merge origin/main` into
  `Eyal-work`, or rebase) so Claude's branch never drifts from what the
  friend has pushed in the meantime.
- Push updates to `origin/Eyal-work`, not `main`. The friend reviews and
  approves changes (via PR) before anything merges into `main` — Claude
  never merges into `main` unprompted.
- This keeps the friend's direct commits to `main` and Claude's commits
  on `Eyal-work` from colliding.

## Current state

See `PROGRESS.md` for the full experiment log and the three-phase roadmap
(19 steps total across Phase 1 supervised, Phase 1 benchmark, Phase 2
unsupervised, Phase 2 benchmark, Phase 3 video). Check it first — it has
the latest results and says exactly which step is next.

**Phase 1 is complete (Steps 0-11).** Baseline 3-layer CNN: PSNR
17.82/SSIM 0.7322. Through 8 isolated, measured changes — most
impactfully the SSIM-based loss (Step 6) and combining width + loss +
augmentation (Step 8) — the best self-built result reached **PSNR 19.51,
SSIM 0.7849** (`models_step8_combined/`, 150 epochs). Two structural
escalations (a skip connection, then a full U-Net) and a resolution
experiment were all tried and didn't beat it — full details and the
reasoning behind each in `PROGRESS.md`. Step 11 benchmarked that result
against published KinD numbers (~20.4 PSNR/~0.80 SSIM): **within ~2% on
SSIM, ~4-5% on PSNR** — close, not matching, a reasonable outcome given
the project's rule against reaching for published architectures mid-phase.

**`models_step8_combined/` is the final Phase 1 checkpoint going forward.**
**Phase 2** (Steps 12-14) is underway — a self-built *unsupervised /
zero-reference* model from scratch, inspired by Zero-DCE's four loss
ideas (spatial consistency, exposure control, color constancy,
illumination smoothness) but written and tuned by us, same incremental
one-change-at-a-time method as Phase 1. Doesn't use `data/high` at all —
see "What this project is" above for why this track exists (the real
goal is real outdoor night video, which has no paired ground truth).

**Step 12 is complete** (`train_unsupervised.py`, `CurveNet`): a
curve-estimation network trained on just the Exposure Control loss,
150 epochs on Colab GPU (`models_step12_curvenet/`, on Drive, not
committed). Training loss fell smoothly to a plateau — confirms the
curve mechanism works. PSNR 7.06/SSIM 0.2969 against `eval15` ground
truth — expected to be low, since this loss alone has no notion of the
actual target image; not a regression, just the reason Step 13 exists.
Full write-up in `PROGRESS.md` (Run 20). **Next: Step 13** — add the
remaining three losses (spatial consistency, color constancy,
illumination smoothness) one at a time, same isolated-change method.

## Environment

- Python venv at `.venv/` in this directory (not committed — `pip install
  -r requirements.txt` to recreate).
- Device detection in `train.py`/`inference.py`/`evaluate.py` checks MPS
  (Apple Silicon) -> CUDA (NVIDIA/AMD) -> CPU automatically, no manual
  changes needed per machine.
- Dataset: LOL dataset, already in `data/` (`low/`+`high/` for training,
  `eval15_low/`+`eval15_high/` held out for evaluation). Now 1,174
  training pairs — original LOLv1 (485) + LOLv2-real's training split
  (689, filenames prefixed `v2_`) — added to test whether more data
  changes earlier findings. **Confirmed for Step 10's U-Net**: Run 21
  (same U-Net config as Run 18, just the bigger dataset) improved on both
  metrics (18.56/0.7806 -> 18.97/0.7969), beating Run 15 on SSIM but still
  short on PSNR — not adopted as best, but supports that overfitting (not
  the architecture itself) was Run 18's problem. `eval15` (LOLv1's 15
  held-out pairs) stays untouched as the fixed benchmark across every run.
  Source documented in `README.md`.
- `evaluate.py` + `metrics.py`: PSNR/SSIM evaluation against the held-out
  eval set — use this, not just training loss, to judge whether a change
  actually helped.
- **Optional: Colab GPU training** via `colab_train.ipynb` — clones
  `Eyal-work`, trains on a free Colab GPU (same `train.py`, CUDA
  auto-detected, no code changes needed), saves checkpoints to Google
  Drive (not git — same `*.pth`-gitignored convention as local runs), and
  pushes a short results note (`COLAB_RESULTS.md`) back to `Eyal-work`.
  Claude should check for `COLAB_RESULTS.md` after a pull and, if new
  entries are there, write them into `PROGRESS.md` properly (with the
  usual spot-checks/analysis) the same as any locally-run experiment,
  then clear the file out once logged.
