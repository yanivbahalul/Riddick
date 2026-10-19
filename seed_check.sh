#!/bin/bash
# Seed variance check: train the current best Phase 2 recipe
# (models_step14_tv40_color1) three times, changing only the random seed,
# to measure how much of a PSNR/SSIM difference is just noise.
# Every "new best" call in PROGRESS.md was a single run with no seed control,
# so differences smaller than this spread were never actually distinguishable.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

for SEED in 1 2 3; do
  DIR="models_seedcheck_s${SEED}"
  echo "=== seed ${SEED}: training ==="
  python train_unsupervised.py \
    --epochs 150 \
    --seed "${SEED}" \
    --spatial-weight 1.0 \
    --color-weight 1.0 \
    --tv-weight 40.0 \
    --exposure-target 0.45 \
    --checkpoint-dir "${DIR}"

  echo "=== seed ${SEED}: evaluating ==="
  python evaluate.py --curvenet --checkpoint "${DIR}/model_epoch150.pth" | tail -2
done
echo "=== all seeds done ==="
