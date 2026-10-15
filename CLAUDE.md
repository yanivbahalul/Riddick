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
- **No `Co-Authored-By: Claude` trailer in commit messages, and no
  "🤖 Generated with Claude Code" line in PR descriptions, on this
  repo.** The user wants to be the sole author shown on `Eyal-work` for
  the friend's review — commits already use the user's own git
  identity, so this is just about the attribution lines themselves. The
  commit trailer was stripped from existing history once (force-pushed)
  and shouldn't be re-added going forward.
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
Full write-up in `PROGRESS.md` (Run 20).

**Step 13 is complete** (Runs 23-25): the remaining three losses
(`spatial_consistency_loss`, `color_constancy_loss`,
`illumination_smoothness_loss` in `train_unsupervised.py`, each gated
behind its own `--spatial-weight`/`--color-weight`/`--tv-weight` flag)
added cumulatively on an overnight Colab GPU run. Color constancy drove
almost all of the improvement (+4.67 PSNR over spatial-only); the full
recipe reached **PSNR 11.55, SSIM 0.4141** (`models_step13c_smoothness/`,
on Drive, not committed) — the best Phase 2 result so far, still well
short of Phase 1 as expected (no ground truth to train against at all).
One caveat: stage 13a's comparison to Step 12 isn't fully clean — it
trained on the expanded 1,174-image set (default `--data-dir data/low`),
while Step 12 used the original 485. Full write-up and reasoning in
`PROGRESS.md` (Runs 23-25).

**Step 14, first attempt (Run 26)**: doubled `hidden_channels` 32 -> 64
on top of Step 13c's recipe — a wash (11.55/0.4141 -> 11.51/0.4155),
unlike Phase 1 where width gave a real gain.

**Step 14, second attempt (Run 27)**: pivoted from architecture to loss
*weight* tuning (all four currently equal at 1.0, untested). Raised
`--spatial-weight` 1.0 -> 5.0 on top of 13c - also didn't help: PSNR flat
(11.58 vs 11.55), SSIM measurably worse (0.4046 vs 0.4141). Spatial
consistency's raw loss landed in the same ~0.006 range regardless of the
5x weight change, suggesting it was already near a floor with little
room for more weight to extract.

**Step 14, third attempt (Run 28) - success, new best**: raised
`--color-weight` 1.0 -> 5.0 instead (same 13c base) - this one worked.
**PSNR 11.99, SSIM 0.4228**, beating 13c (11.55/0.4141) on both metrics.
Color's raw loss dropped ~5.7x with the higher weight (0.0102 -> 0.0018),
unlike spatial's which plateaued regardless - direct evidence color had
real headroom left and spatial didn't. **`models_step14_colorweight5/`
is now the project's best Phase 2 checkpoint.**

**Step 15 is complete**: benchmarked Run 28 (11.99/0.4228) against
published Zero-DCE (~14.86 PSNR/~0.56 SSIM on LOL, verified against
multiple papers citing its results table). Gap was ~19% PSNR, ~25% SSIM
— meaningfully wider than Phase 1's ~2-5% gap to KinD, which makes sense
given Zero-DCE's deeper architecture and fully-tuned implementation
versus this project's minimal, just-barely-tuned `CurveNet`. Full
reasoning in `PROGRESS.md`.

**Squeezing Step 14 further**: pushed color-weight again, 5.0 -> 10.0
(Run 29). Further improvement, but with diminishing returns: **PSNR
12.18, SSIM 0.4259** (+0.19/+0.0031 over Run 28, smaller than the
+0.44/+0.0087 jump from 1.0->5.0). `models_step14_colorweight10/` was
the project's best Phase 2 checkpoint at this point (later superseded -
see the exposure-target follow-up near the end of this section).

**Step 16 is complete — and the result is a real surprise.** Ran Phase 1
and Phase 2's best checkpoints on 11 real outdoor night photos (`data/
real_night/`, phone-camera shots - garden, pathway, street - not LOL).
**Phase 1 generalized reasonably** (color-cast/exposure artifacts, but
stayed a recognizable usable photo). **Phase 2 broke down almost
completely** — output looked like an edge-detection filter, not an
enhanced image. Ruled out a resolution-pipeline bug by re-testing at
`CurveNet`'s exact training resolution (`data/real_night_256/`) - same
broken result, confirmed via output dimensions. Likely cause: the
exposure loss pushes every patch toward the same flat brightness target
regardless of its starting point, which real photos' extreme local
dynamic range (spotlit areas next to pitch black) handles very badly
compared to LOL's uniformly-dark training images - and `CurveNet`'s 8x
iterative curve application compounds that instability, unlike Phase 1's
single forward pass. Full writeup in `PROGRESS.md` (Step 16).

**Important nuance**: this only tested zero-shot transfer (no
fine-tuning). Phase 2's actual justification for existing (see "What
this project is" above) was always about *fine-tuning* on real footage
with no paired data needed - that's not what's been tested yet.

**Tested it, twice.** Added `--init-checkpoint` to `train_unsupervised.py`
(load an existing checkpoint instead of random init - new capability).

- Round 1: `--epochs 50 --lr 1e-5`. `eval15` 12.38/0.4322 (slightly
  better than pre-fine-tune, no forgetting), real-photo output unchanged.
- Round 2: `--epochs 500` (10x more steps), same `lr=1e-5`. `eval15`
  improved *again* to 12.55/0.4385, real-photo output **still visually
  identical** to round 1 and to the original un-fine-tuned run.

**The clarifying pattern**: the model keeps getting measurably better at
`eval15` the longer it trains - even though it's training *on the real
photos*, not on LOL - while showing zero visible change on the real
photos themselves, across a 10x range of training duration. That's hard
to explain as "just needed more steps," since more steps should show up
on the data actually being trained on if that were the issue. Points
toward the original architectural hypothesis (flat per-patch exposure
target + 8x iterative curve compounding) rather than an undertrained
model. Epoch count is reasonably ruled out now.

**Round 3 (run locally, not Colab - only 11 images): much higher LR
(1e-5 -> 1e-3, 100x), same 50 epochs.** `eval15` improved to
PSNR=13.05/SSIM=0.4464, real-photo output visibly more saturated but
still not clean. Learning rate, not epoch count, was the real lever.

**Round 4 — the breakthrough.** 4a (same `lr=1e-3`, epochs 50 -> 200)
plateaued, confirming epochs alone top out quickly at a given LR. 4b
(back to 50 epochs, `lr` pushed 1e-3 -> 3e-3) **fixed it**:
**PSNR=14.08, SSIM=0.5782** - SSIM now *exceeds* Zero-DCE's published
benchmark (~0.56), and the real-photo output is a genuinely
recognizable, well-exposed photo for the first time in this whole
investigation - garden, pathway, shed all clearly visible, not an
edge-map. `models_step16_finetuned_lr3e3/` (local checkpoint,
gitignored) is the result.

**This answers Step 16's open question decisively in Phase 2's favor**:
a zero-reference model genuinely can adapt to real footage with no
paired data, exactly as its original justification claimed - it just
needed a properly-tuned fine-tuning learning rate, not the project's
standard from-scratch training rate (1e-4). The full pattern across all
four rounds: epoch count made essentially no difference at any LR
tried; LR was the entire story (1e-5 -> nothing, 1e-3 -> partial,
3e-3 -> full fix).

**Round 5 - found the ceiling, Step 16 is now closed.** Pushed `--lr`
further past 3e-3 (3.5e-3, 4e-3, 5e-3, same 50-epoch recipe) to see where
it stops helping. All three collapsed to the *same* degenerate result
(`eval15` ~6.2/0.39, exposure loss stuck high, color loss zeroed out,
real-photo output blown out to near-white) - not a gradual decline but a
sharp cliff landing on one specific bad fixed point. The cliff edge sits
just above 3e-3 (between 3e-3 and 3.5e-3), meaning `lr=3e-3` is close to
the top of a narrow stable window, not a conservative middle value with
room to spare. **`models_step16_finetuned_lr3e3` (PSNR=14.08,
SSIM=0.5782) is confirmed as the final answer to Step 16** - pushing
further isn't a viable path to more improvement. Full round-5 write-up
in `PROGRESS.md`. **Next**: Phase 3 (video), carrying forward
`--lr 3e-3`, 50 epochs as the fine-tuning recipe for adapting to new
real footage.

**Before Phase 3: one more Phase 1 follow-up, tried and didn't help.**
Added `--lr-schedule` to `train.py` (cosine-anneals `lr` to 0 over the
run) and tested it on top of Step 8's winning recipe, run on Colab GPU
(`models_step8_lrschedule/`). Result: **PSNR=19.37, SSIM=0.7840** - a
wash, marginally worse than the 19.51/0.7849 baseline. The schedule
decayed too fast relative to how much the model still had left to learn
at 150 epochs (LR already down to 5e-5 by epoch 75, barely moving after
that), unlike the flat `lr=1e-4` baseline which keeps taking full-size
steps the whole run. `models_step8_combined/` remains the best Phase 1
checkpoint. Full write-up in `PROGRESS.md`.

**Phase 2 follow-up, same session — a real improvement.** Checked an
assumption instead of a weight: `exposure_loss`'s target brightness
(0.6) was inherited from Zero-DCE's paper and never verified against
this project's own data. `data/high` actually averages **0.448**
brightness (full 1,174-image set) - meaningfully below 0.6. Tested
`--exposure-target 0.45` on top of Run 29's recipe (color-weight=10.0,
spatial=1.0, tv=1.0), run on Colab GPU -> `models_step14_exptarget045/`.
**PSNR=12.75, SSIM=0.4373** - the biggest single-change gain since the
original color-weight 1.0->5.0 jump (+0.57/+0.0114 over Run 29's
12.18/0.4259). Unlike every other weight-tuning attempt, which searched
blind, this one had a checkable right answer sitting in the dataset the
whole time. **`models_step14_exptarget045/` is now the project's best
Phase 2 checkpoint**, replacing `models_step14_colorweight10/`. Worth
carrying into Phase 3: any other Zero-DCE-inherited constant is worth
checking against this project's own data before assuming it's correct.
Full write-up in `PROGRESS.md`.

**Phase 2 follow-up, same session — tv-weight had real headroom too.**
`--tv-weight` (illumination smoothness) had sat at its Step 13c default
of 1.0, never isolated the way spatial-weight and color-weight both
were. Pushed 1.0 -> 5.0 on top of `models_step14_exptarget045`, run on
Colab GPU -> `models_step14_tvweight5/`. **PSNR=13.21, SSIM=0.4647**
(+0.46/+0.0274 over 12.75/0.4373) - the largest single SSIM gain of any
Step 14 change yet. Unlike spatial-weight's test (flat raw loss
regardless of weight - a floor), smoothness's raw loss actually dropped
with the higher weight, the same "real headroom" signature color-weight
showed - so this likely isn't fully squeezed yet either.
**`models_step14_tvweight5/` is now the project's best Phase 2
checkpoint**, replacing `models_step14_exptarget045/`. Full write-up in
`PROGRESS.md`.

**Pushed again, same session - the biggest gain yet.** Followed
color-weight's own precedent: tv-weight 5.0 -> 10.0, same recipe
otherwise, run on Colab GPU (part of a larger heavy-compute batch across
both phases) -> `models_step14_tvweight10/`. **PSNR=14.13, SSIM=0.4978**
(+0.92/+0.0331 over 13.21/0.4647) - the largest single-change gain of
the entire Step 14 series, bigger than every weight-tuning or
exposure-target result before it. **`models_step14_tvweight10/` is now
the project's best Phase 2 checkpoint**, replacing `models_step14_
tvweight5/`. tv-weight hasn't plateaued across two consecutive pushes -
worth another round if Phase 2 keeps getting squeezed. Full write-up in
`PROGRESS.md`.

**Batch Stage 2 - width retest, still a wash.** Rechecked
`hidden_channels` 32 -> 64 on `CurveNet` now that the loss recipe is far
better tuned than at Run 26's original test -> `models_step14_
hc64_retest/`. **PSNR=13.05, SSIM=0.4624** - essentially the same
marginal regression Run 26 found originally (vs. its actual baseline,
`models_step14_tvweight5`'s 13.21/0.4647: -0.16/-0.0023). Training loss
was comparable or slightly better across every component, but eval
didn't follow - same "no real capacity need" signature as before, now
confirmed at two very different points in the recipe's maturity. Width
is reasonably closed as a Phase 2 lever. Full write-up in `PROGRESS.md`.

**Batch Stage 3 - epochs 150->600, Phase 2's first overfitting
signature.** Same recipe as `models_step14_tvweight5`, only `--epochs`
150 -> 600 -> `models_step14_tvweight5_600ep/`. Training loss fell
smoothly to a genuine plateau by epoch 600 (0.1034, exposure=0.0292,
smoothness=0.0127), no instability anywhere. **PSNR=12.97, SSIM=0.4634**
- a real, if small, *regression* against the 150-epoch baseline
(13.21/0.4647: -0.24/-0.0013). Training loss kept improving while
held-out eval went slightly backward - the same shape Run 18's U-Net
showed, unlike Phase 1's Run 13 epoch-extension (supervised, so still
anchored to real target pixels - no regression there). With no ground
truth at all, more epochs just overfits the training images' own
statistical quirks. Epoch count beyond 150 is now closed as a Phase 2
lever. `models_step14_tvweight10/` remains the best Phase 2 checkpoint.
Full write-up in `PROGRESS.md`.

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
- **Optional: Colab GPU training** via `colab_train.ipynb` — clones
  `Eyal-work`, trains on a free Colab GPU (same `train.py`/
  `train_unsupervised.py`, CUDA auto-detected, no code changes needed),
  saves checkpoints to Google Drive (not git — same `*.pth`-gitignored
  convention as local runs), and pushes a results note (`COLAB_RESULTS.md`)
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
  `COLAB_RESULTS.md` for a new entry, e.g. via a self-paced loop) instead
  of passively waiting to be told. Once there's a new entry, write it
  into `PROGRESS.md` properly (with the usual spot-checks/analysis) the
  same as any locally-run experiment, then clear `COLAB_RESULTS.md` out.
