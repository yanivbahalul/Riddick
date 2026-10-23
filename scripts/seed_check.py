"""Seed variance check: train the current best Phase 2 recipe
(models_step14_tv40_color1) three times, changing only the random seed,
to measure how much of a PSNR/SSIM difference is just noise.

Every "new best" call in docs/PROGRESS.md was a single run with no seed
control, so differences smaller than this spread were never distinguishable.

    python scripts/seed_check.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# sys.executable is the Python running this file, so the same venv is used
# for every step on any OS.
for seed in (1, 2, 3):
    out_dir = f"models_seedcheck_s{seed}"
    print(f"=== seed {seed}: training ===", flush=True)
    subprocess.run([sys.executable, "-u", "scripts/train_unsupervised.py",
                    "--epochs", "150", "--seed", str(seed),
                    "--spatial-weight", "1.0", "--color-weight", "1.0",
                    "--tv-weight", "40.0", "--exposure-target", "0.45",
                    "--checkpoint-dir", out_dir], cwd=ROOT, check=True)

    print(f"=== seed {seed}: evaluating ===")
    result = subprocess.run([sys.executable, "scripts/evaluate.py", "--curvenet",
                             "--checkpoint", f"{out_dir}/model_epoch150.pth"],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    print(result.stdout.strip().splitlines()[-1])

print("=== all seeds done ===")
