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

## Roadmap — one change at a time, measure after each

Goal: understand *why* each change helps (or doesn't) by changing exactly
one thing per step, training, running `evaluate.py`, and comparing against
the previous best before moving to the next step. Not skipping ahead to a
published architecture without having tried the simpler, self-built version
first.

- [x] Step 0 — Baseline: 3-layer CNN, 50 epochs. **PSNR 17.82, SSIM 0.7322**
- [ ] Step 1 — Same architecture, longer run (150 epochs) — isolate whether
      the ceiling is training time or architecture. *(running now)*
- [ ] Step 2 — Tune learning rate on the baseline architecture (try higher/lower than 1e-4)
- [ ] Step 3 — Increase width: `hidden_channels` 32 -> 64
- [ ] Step 4 — Increase depth: add a 4th Conv2d+ReLU layer
- [ ] Step 5 — Add `BatchNorm2d` after each Conv2d
- [ ] Step 6 — Swap loss function: try `MSELoss`, or add the SSIM from
      `metrics.py` as part of the training loss (not just evaluation)
- [ ] Step 7 — Data augmentation: random flip/crop in the training transform
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

## Evaluation results

| Checkpoint | PSNR | SSIM | Notes |
|---|---|---|---|
| Run 3, epoch 50 | 17.82 | 0.7322 | baseline 3-layer CNN, 50 epochs, 256x256 |
