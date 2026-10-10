# Progress Log

Tracking each training run: config, results, and what we learned. Goal is
incremental improvement, not jumping straight to a complex architecture.

## Baseline model

`LowLightEnhanceNet` in `train.py` — 3 Conv2d layers (32 hidden channels),
ReLU between, Sigmoid output. No skip connections, no normalization. This is
intentionally the simplest possible baseline.

## Runs

### Run 1 — pipeline smoke test
- Config: `--epochs 3 --batch-size 8 --image-size 64`
- Loss: 0.1757 -> 0.1703 -> 0.1631
- Purpose: verify train.py runs end to end on real data. No real learning expected.

### Run 2 — MPS smoke test
- Config: `--epochs 2 --batch-size 8 --image-size 64`
- Confirmed `torch.backends.mps.is_available() == True` on M4, training ran
  successfully through MPS instead of CPU.
- Loss: 0.1745 -> 0.1691

### Run 3 — first full run (complete)
- Config: `--epochs 50 --batch-size 8 --image-size 256 --lr 1e-4` (all defaults)
- Loss: 0.1290 (epoch 37) -> 0.1263 (epoch 50) — flattened out in the last
  13 epochs, confirms the plateau observation below.
- Evaluation on `eval15` (`model_epoch50.pth`): PSNR=17.82, SSIM=0.7322
  (avg over 15 images). Big jump from the epoch-1 smoke-test checkpoint
  (PSNR~12.75, SSIM~0.45), so the model is genuinely learning, not just
  memorizing — but numbers still modest for this task (strong results in
  papers are usually PSNR 20+, SSIM 0.8+).
- Observation: loss decreased slowly relative to epoch count, and flattened
  near the end. Likely ceiling of the simple 3-layer architecture, not a bug.

### Run 4 — Step 1: same architecture, 150 epochs (complete)
- Config: `--epochs 150 --batch-size 8 --image-size 256 --lr 1e-4`, saved to
  `models_150ep/` to keep Run 3's checkpoints intact.
- Loss: 0.1811 (epoch 1) -> 0.1263 (epoch 50) -> 0.1211 (epoch 150). Loss
  kept inching down but very slowly after ~epoch 60 (0.126x range for most
  of epochs 60-150) — diminishing returns, not a clean plateau but close to one.
- Evaluation on `eval15` (`model_epoch150.pth`): PSNR=18.38, SSIM=0.7573.
- **Conclusion: tripling the training time (50 -> 150 epochs) bought only
  +0.56 PSNR / +0.025 SSIM.** That's a small return for 3x the compute.
  This points to the ceiling being mostly the **architecture**, not training
  time — more epochs on this same 3-layer CNN is not the efficient path to
  the 20+ PSNR / 0.8+ SSIM range seen in papers. Moving to Step 2 (LR tuning)
  for one more check before structural changes, but expectations are now
  lower that LR alone closes the gap either.

### Run 5 & 6 — Step 2: learning rate tuning (complete)
- Same architecture, 50 epochs (comparable to Run 3), only `--lr` changed.
- Run 5: `--lr 0.0005` (5x higher than default) -> `models_lr5e4/`
  - Evaluation: PSNR=18.54, SSIM=0.7442 — modest improvement over baseline.
- Run 6: `--lr 0.00005` (half the default) -> `models_lr5e5/`
  - Evaluation: PSNR=17.16, SSIM=0.6969 — worse than baseline.
- **Conclusion: higher LR helps a little (comparable to what Run 4's extra
  100 epochs bought), lower LR hurts.** Best result so far is still Run 5
  (18.54/0.7442), but it's in the same range as Run 4 (18.38/0.7573) — not
  a breakthrough. Neither more training time nor LR tuning gets close to
  the 20+/0.8+ range seen in papers. Both Step 1 and Step 2 point the same
  direction: **the 3-layer CNN architecture itself is the bottleneck.**
  Moving to structural changes (Step 3+) next.

### Run 7 — Step 3: width increase (complete)
- Added `--hidden-channels` CLI arg to `train.py`/`inference.py`/`evaluate.py`
  (model constructor already supported it, just wasn't exposed) so future
  width experiments don't need code edits each time.
- Config: `--epochs 50 --hidden-channels 64` (lr=1e-4 default, same as Run 3,
  to isolate width as the only changed variable) -> `models_hc64/`
- Evaluation: PSNR=18.26, SSIM=0.7474.
- **Conclusion: doubling width (32->64) bought +0.44 PSNR / +0.015 SSIM —
  same modest range as Step 1 (more epochs) and Step 2 (higher LR).** Four
  different changes now (more time, higher LR, more width) all land in the
  same narrow 18.2-18.5 PSNR band. Width alone is not a breakthrough either.

### Run 8 — Step 4: depth increase (in progress)
- Added `--extra-layer` CLI flag (adds a 4th Conv2d+ReLU, 32->32) to
  `train.py`/`inference.py`/`evaluate.py`.
- Config: `--epochs 50 --extra-layer` (lr=1e-4, hidden_channels=32 default,
  same as Run 3, to isolate depth as the only changed variable) -> `models_extralayer/`
- Running in background. Will evaluate against `eval15` and compare to
  Run 3's baseline (17.82/0.7322) and the 18.2-18.5 band from Steps 1-3.

## Roadmap — one change at a time, measure after each

Goal: understand *why* each change helps (or doesn't) by changing exactly
one thing per step, training, running `evaluate.py`, and comparing against
the previous best before moving to the next step. Not skipping ahead to a
published architecture without having tried the simpler, self-built version
first.

**Phase 1 success target:** PSNR >= 20, SSIM >= 0.8 on `eval15` (the range
papers typically report for this task). Phase 1 ends when we hit that
target, or after Step 10 is tried either way — whichever comes first. Not
an open-ended search for perfection.

- [x] Step 0 — Baseline: 3-layer CNN, 50 epochs. **PSNR 17.82, SSIM 0.7322**
- [x] Step 1 — Same architecture, 150 epochs. **PSNR 18.38, SSIM 0.7573** —
      small gain for 3x training time, points to architecture being the ceiling.
- [x] Step 2 — Tune learning rate. Higher (`5e-4`): **PSNR 18.54, SSIM 0.7442**
      (best so far). Lower (`5e-5`): PSNR 17.16, SSIM 0.6969 (worse). Confirms
      architecture, not hyperparameters, is the bottleneck.
- [x] Step 3 — Increase width: `hidden_channels` 32 -> 64. **PSNR 18.26,
      SSIM 0.7474** — modest gain, same range as Steps 1-2.
- [ ] Step 4 — Increase depth: add a 4th Conv2d+ReLU layer *(running now, Run 8)*
- [ ] Step 5 — Add `BatchNorm2d` after each Conv2d
- [ ] Step 6 — Swap loss function: try `MSELoss`, or add the SSIM from
      `metrics.py` as part of the training loss (not just evaluation)
- [ ] Step 7 — Data augmentation: random flip/crop in the training transform
- [ ] Step 7.5 — Combine whichever of Steps 3-7 individually helped (not all
      of them automatically — only the ones that showed a real gain) into
      one run, before moving to skip connections. Skip this step if none of
      3-7 helped meaningfully on their own.
- [ ] Step 8 — Add one skip connection (first step toward U-Net) — requires
      switching `forward` from `nn.Sequential` to manual layer calls + `torch.cat`
- [ ] Step 9 — Full U-Net-style architecture (multiple downsample/upsample
      stages with skip connections), compare against all of the above
- [ ] Step 10 — Zero-DCE approach (zero-reference, no `data/high` needed):
      DCE-Net + the 4 non-reference losses, compare against the supervised
      approach used in all steps above

Steps 3-10 are not strictly sequential — once Steps 1-2 isolate whether
training time/LR explain the gap, pick whichever structural change seems
most promising based on results so far, not necessarily in this exact order.

## Phase 2 — Video (after Steps 0-10 above are done)

The end goal is video, not just single images. Not starting this until the
image model (Steps 0-10) hits good, stable metrics — a video pipeline built
on a weak per-frame model just inherits all its problems, frame by frame.

- [ ] Step 11 — Naive baseline: run the best image checkpoint on every frame
      of a short test video independently (loop over frames, same as
      `inference.py`), reassemble into a video. No new code logic, just a
      frame-extraction/reassembly wrapper.
- [ ] Step 12 — Watch the result and check specifically for **flickering**
      (brightness/color changing frame-to-frame in ways that weren't in the
      original video) — this is the expected failure mode of per-frame
      processing with no memory between frames.
- [ ] Step 13 — If flickering shows up: investigate temporal consistency
      fixes, roughly in order of complexity — simple post-process smoothing
      between consecutive output frames, then (if needed) a temporal
      consistency loss during training that penalizes large differences
      between consecutive processed frames, then (if still needed) feeding
      the previous frame's output as extra input to the model.

Step 11 tells us whether this is even a real problem for our case before
investing in anything more complex — same "measure before you build"
approach as Phase 1.

## Evaluation results

| Checkpoint | PSNR | SSIM | Notes |
|---|---|---|---|
| Run 1, epoch 3 | N/A | N/A | smoke test — 64x64, pipeline check only, not evaluated |
| Run 2, epoch 2 | N/A | N/A | smoke test — 64x64, MPS check only, not evaluated |
| Run 3, epoch 50 | 17.82 | 0.7322 | baseline 3-layer CNN, 50 epochs, 256x256 |
| Run 4, epoch 150 | 18.38 | 0.7573 | same architecture, 150 epochs — diminishing returns |
| Run 5, epoch 50 (lr=5e-4) | 18.54 | 0.7442 | best so far, still modest |
| Run 6, epoch 50 (lr=5e-5) | 17.16 | 0.6969 | lower LR hurts |
| Run 7, epoch 50 (hc=64) | 18.26 | 0.7474 | width increase, same modest range as Steps 1-2 |
