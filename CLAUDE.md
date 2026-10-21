# Project context for Claude

This file exists so a new Claude Code session (or anyone else) can pick up
this project without the original conversation history.

**Keep this file up to date.** Whenever you (Claude) make a meaningful
change to the project, approach, or state — not just routine training runs
already logged in `docs/PROGRESS.md` — update the relevant section here too.
This file should always reflect the current way of working and current
state, not just how things were when it was first written.

## What this project is

A low-light image enhancement project in PyTorch, built as a **learning
project** — the user is learning PyTorch/deep learning fundamentals while
building it, step by step, not just shipping a finished model. Prioritize
teaching and incremental, measurable changes over speed.

**The full learning process has three phases** — see "Roadmap" in
`docs/PROGRESS.md` for the complete breakdown:
1. **Phase 1** — push a self-built *supervised* model as far as possible (Steps 0-10)
2. **Phase 1 benchmark** — compare to a published supervised model's numbers (Step 11)
3. **Phase 2** — build our own *unsupervised/zero-reference* model, inspired by Zero-DCE's ideas but self-built (Steps 12-14)
4. **Phase 2 benchmark** — compare to the real published Zero-DCE (Step 15), plus a domain-gap check on real footage (Step 16)
5. **Phase 2.5** — validation and physics fixes (Steps 17-19): seed noise floor, the linear-light exposure fix, and redoing the Phase 2 benchmark honestly on the clean held-out set
6. **Phase 3** — video (Steps 20-25), reframed as a **capture-and-merge pipeline** rather than per-frame processing: multi-frame merge leads, because a static camera can exploit temporal redundancy far better than a phone can

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
  it impossible to know what caused a result. See "Roadmap" in `docs/PROGRESS.md`.
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
- **No `Co-Authored-By: Claude` trailer in commit messages, and no
  "🤖 Generated with Claude Code" line in PR descriptions, on this
  repo.** The user wants to be the sole author shown on `Eyal-work` for
  the friend's review — commits already use the user's own git
  identity, so this is just about the attribution lines themselves. The
  commit trailer was stripped from existing history once (force-pushed)
  and shouldn't be re-added going forward.
- Confirm before pushing to GitHub for non-trivial changes; routine
  documentation updates (like `docs/PROGRESS.md`) have generally been fine to
  push directly once the pattern was established.
- **Each experiment gets its own `--checkpoint-dir`** (e.g. `models_150ep/`,
  `models_lr5e4/`) so earlier runs' checkpoints are never overwritten and
  stay comparable. `models/` is the original baseline run.
- Training runs are launched in the background (`run_in_background`) since
  they take minutes to hours. **Launch them with `python -u`** (or
  `PYTHONUNBUFFERED=1`): without it Python buffers stdout when redirected
  and the log file stays empty until the process exits, which makes a
  multi-hour run impossible to follow. Checkpoint file count in the run's
  directory is the fallback progress signal and works regardless.

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

## Where the docs live

All write-ups are in `docs/` (root keeps only `README.md` and this file):

| File | What's in it |
|---|---|
| `docs/PROGRESS.md` | Full experiment log + the step-by-step roadmap. **Check this first** — it says exactly which step is next. |
| `docs/RESEARCH.md` | Literature/hardware survey **and** original analysis done on this project's data (§9 — the derivations) |
| `docs/VISUAL_PROGRESS.md` | Output images at the current best checkpoint. Replaced, not appended. |
| `docs/COLAB_RESULTS.md` | Scratch landing pad for Colab output; write it into `PROGRESS.md` properly, then clear it. |

`docs/comparisons/` holds the images `VISUAL_PROGRESS.md` embeds.

`common.py` holds what every checkpoint-loading script needs: `pick_device()`,
`build_model()` and `add_checkpoint_args()`. These were copy-pasted across
evaluate/inference/pipeline/assess/live_enhance and had drifted apart - add
new architecture flags there, once, not per script.

## Read this before trusting any number below

Findings from a methodology audit and the measurement work that followed.
All of them are in `docs/RESEARCH.md` §9 with derivations and numbers.

**Measurement rules, learned the hard way:**

- **Always evaluate with `evaluate.py` at native resolution.** A 256x256
  resize once produced 0.8367 SSIM where the real figure was 0.7847 —
  an apparent jump past the project's target that was pure protocol
  mismatch (§9.14).
- **Judge differences against the noise floor**: ±0.07 PSNR, ±0.0004 SSIM
  (§9.14, Step 17), and only *within one machine* — Colab-vs-local
  carries about ±0.2.
- **`eval15` is the wrong place to test anything noise- or
  exposure-related.** It is clean, and its inputs already sit near the
  trained brightness, so fixes aimed at real footage show up as neutral
  or negative there. This has now misled three separate experiments.

1. **Every score in this log is selection-biased.** ~20 tuning decisions
   were all made by maximizing the same 15 `eval15` images. Phase 2 now
   looks ahead of published Zero-DCE on `eval15` — but that compares a
   heavily tuned score to an untuned one, so **that claim must not be
   made** until re-measured on `data/lolv2_test_*` (100 clean pairs, added
   for this, touch-once protocol in `README.md`).
2. **The exposure loss measures brightness in the wrong space.** sRGB is
   gamma-encoded and `x^(1/2.2)` is concave, so by Jensen the sRGB patch
   mean *understates* true light, with error growing in patch contrast.
   Measured bias: −0.0028 on LOL dark vs **−0.0162 on real night photos**.
   This is a measured mechanism for the Step 16 domain-gap breakdown, not
   a guess. `--linear-exposure` (+ `--exposure-target 0.48`) is the fix.

3. **The real-photo breakdown is input covariate shift, not the loss.**
   LOL training inputs average 0.0496 brightness; real phone night photos
   average **0.2071** (already auto-exposed by the phone's ISP), and the
   ranges barely overlap. The model applies the curve it learned for
   near-black input and overshoots by ~50%, blowing out highlights.
   **Scaling the input to the trained brightness fixes it with no
   retraining** — blown-out pixels 2.88% → 0.00%
   (`inference.py --match-input-brightness 0.0496`). This supersedes the
   Step 16 story, where five rounds of fine-tuning and an LR sweep to
   3e-3 were spent re-calibrating to that distribution.

**Noise floor is now measured** (Step 17, three seeds): PSNR **±0.07**,
SSIM **±0.0004** — tight enough that most of the existing log holds up,
including differences as small as ±0.19. **But that floor is per-machine**:
the same recipe scored 0.19 lower on Colab than on local MPS, so
Colab-vs-local comparisons carry roughly ±0.2 and are weaker than they
look. Prefer comparing runs from the same machine.

## What multi-frame merging changed, and what it did not

Merging N frames of a static scene is the only stage that adds real
signal instead of inferring it, and it is the largest single effect
found in the project (`merge.py`, §9.6-9.10):

- **3.55x noise reduction at 72 real frames**, and it does this *without
  damaging the image*: the edge profile is unchanged (so no blur), and
  effective bit depth goes from 7.8 to **12.0 bits** because averaging
  pushes signal below the 8-bit step.
- **Capped at ~3.55x rather than the theoretical 8.5x** because the
  webcam's own temporal denoising correlates adjacent frames (ρ=0.62).
  Spreading frames out recovers ~27% of the loss.
- **Live is feasible**: 22.1 ms/frame, ~45 fps at 960x540
  (`live_enhance.py`), via reduced-resolution coefficient prediction
  (79.8 dB / SSIM 1.00000 against full-res, so free) and an incremental
  rolling merge (17 ms → 0.004 ms).

**The open problem:** the enhancement stage amplifies noise ~2.6x, which
eats most of what merging buys (2.15x cleanup → 1.69x at the output).
Step 26 tried to fix this with noise-aware training and **failed** — it
made amplification worse (5.08x → 7.08x), most likely because the
synthetic noise could never be calibrated to this camera (§9.8's photon
transfer curve returned physically impossible values). This is still the
main quality limit.

**Phase 3 is on hold, hardware-blocked.** Every capture lands at
0.15-0.21 brightness because the webcam auto-exposes before frames
arrive; no software stage can undo that. Resuming needs manual
exposure/ISO, temporal NR off, and ideally RAW — each requirement traced
to a specific measurement, not a preference. Deep darkness is therefore
**untested, not estimated**; a simulated-darkness experiment was built
and then deliberately deleted, because the user correctly pointed out it
answered a different question.

## Current state

`docs/PROGRESS.md` is the authority - it has every run, the roadmap, and
which step is next. This is only the short version.

**Phase 1 (supervised) and Phase 2 (unsupervised) are both built and
benchmarked.** Current picks, by the rule below:

| | checkpoint | `eval15` | `lolv2_test` (clean) |
|---|---|---|---|
| Phase 1 | `models_expjitter` | 16.95 / 0.7462 | **19.81 / 0.7881** |
| Phase 2 | `models_seedcheck_s1` | 18.21 / 0.6201 | 17.69 / 0.6005 |

**Phase 3 (video) is hardware-blocked, not analysis-blocked.** The merge
pipeline is built, measured and runs live at ~45 fps; the webcam just
cannot produce the conditions it exists for. Three requirements for the
next sensor, each from a measurement: manual exposure/ISO, temporal NR
off, RAW. See the Phase 3 note in `docs/PROGRESS.md`.

**Picking a "best" checkpoint takes three numbers, not one** - run
`scripts/assess.py <dir>...`:
1. real footage (blown-out share, raw input) - closest to the goal
2. robustness (output drift across an 8x exposure range)
3. `eval15` - a regression guard only, since it was tuned against

A large `eval15` drop still needs explaining, but a small one bought with
real robustness is an acceptable trade. Step 30 was rejected on visual
inspection despite winning on criteria 1 and 2 - metrics do not get the
last word.

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
  changes earlier findings. **Resolved**: more data only helps the
  architecture that was overfitting. Run 21 (U-Net, same config as Run 18,
  bigger dataset) improved on both metrics (18.56/0.7806 -> 18.97/0.7969),
  beating Run 15 on SSIM but still short on PSNR. Run 22 (Step 8 combo,
  same config as Run 15, bigger dataset) got slightly *worse* on both
  metrics (19.51/0.7849 -> 19.18/0.7811) — it wasn't capacity-starved, so
  more (more varied) data just diluted its fit to `eval15`'s LOLv1
  distribution. `models_step8_combined/` (Run 15) remains the project's
  best checkpoint, confirmed on firmer footing. `eval15` (LOLv1's 15
  held-out pairs) stays untouched as the fixed benchmark across every run.
  Source documented in `README.md`.
- `evaluate.py` + `metrics.py`: PSNR/SSIM evaluation against the held-out
  eval set — use this, not just training loss, to judge whether a change
  actually helped.
- **Optional: Colab GPU training** via `scripts/colab_train.ipynb` — clones
  `Eyal-work`, trains on a free Colab GPU (same `train.py`/
  `train_unsupervised.py`, CUDA auto-detected, no code changes needed),
  saves checkpoints to Google Drive (not git — same `*.pth`-gitignored
  convention as local runs), and pushes a results note (`docs/COLAB_RESULTS.md`)
  back to `Eyal-work`. The Train/Evaluate/Push cells capture the real
  command and eval output into variables automatically (`TRAIN_CMD`,
  `EVAL_CMD`, `eval_output`) — no manual copy-paste into placeholders
  needed anymore. The Evaluate cell also saves `eval_result.txt` straight
  into the checkpoint's own Drive folder, as a backup that survives even
  if the push-to-git cell never runs.
- **Whenever Claude sets up a new Colab run**: give an upfront time
  estimate first (basis: Step 13's CurveNet stages, hidden_channels=32,
  150 epochs, ~58 min/stage on a T4 — scale from there based on what's
  changing, e.g. wider network or bigger dataset; say clearly it's an
  estimate). Then self-check for completion (`git fetch` + check
  `docs/COLAB_RESULTS.md` for a new entry, e.g. via a self-paced loop) instead
  of passively waiting to be told. Once there's a new entry, write it
  into `docs/PROGRESS.md` properly (with the usual spot-checks/analysis) the
  same as any locally-run experiment, then clear `docs/COLAB_RESULTS.md` out.
