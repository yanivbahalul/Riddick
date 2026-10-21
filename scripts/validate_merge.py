"""Does merge.py actually deliver the sqrt(N) noise reduction it should?

Simulates N captures of the same dark scene with a physically-shaped
sensor noise model (Poisson shot noise + Gaussian read noise), merges
them, and compares the measured noise against the sqrt(N) prediction.

Synthetic on purpose: a real burst has no ground truth, so there would be
nothing to measure error against. Here the clean source image is known
exactly, so the merge can be checked rather than eyeballed.

    python scripts/validate_merge.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os

import numpy as np
from PIL import Image

from riddick.merge import merge_frames  # noqa: E402

DARKEN = 0.06        # scale a well-exposed photo down to night-ish levels
FULL_WELL = 2000.0   # electrons at full scale; sets how strong shot noise is
READ_NOISE = 2.0     # electrons, typical modern BSI CMOS
RNG = np.random.default_rng(0)


def simulate_capture(clean_linear):
    """One noisy sensor read of a scene, in linear light."""
    electrons = clean_linear * FULL_WELL
    # shot noise is Poisson in the photon count; read noise is additive
    noisy = RNG.poisson(electrons) + RNG.normal(0, READ_NOISE, electrons.shape)
    return np.clip(noisy / FULL_WELL, 0, 1)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    src = sorted(os.listdir("data/high"))[:6]
    counts = [1, 2, 4, 8, 16, 32, 64]
    results = {n: [] for n in counts}

    for name in src:
        clean = np.asarray(Image.open(f"data/high/{name}").convert("RGB"), dtype=np.float64) / 255.0
        clean_dark = (clean ** 2.2) * DARKEN          # to linear light, then darken

        frames = [simulate_capture(clean_dark) for _ in range(max(counts))]
        for n in counts:
            # alignment off: these are a locked-off camera by construction,
            # so this isolates the merge itself from any alignment error
            merged, _ = merge_frames(frames[:n], align=False)
            results[n].append(rmse(merged, clean_dark))

    base = np.mean(results[1])
    print(f"Noise vs clean source, averaged over {len(src)} scenes")
    print(f"(simulated sensor: full-well {FULL_WELL:.0f}e-, read noise {READ_NOISE}e-)\n")
    print(f"{'N':>4} {'RMSE':>10} {'measured':>10} {'sqrt(N)':>10} {'error':>8}")
    print("-" * 46)
    for n in counts:
        got = np.mean(results[n])
        measured, predicted = base / got, np.sqrt(n)
        print(f"{n:>4} {got:10.5f} {measured:9.2f}x {predicted:9.2f}x {abs(measured / predicted - 1) * 100:7.1f}%")

    final = base / np.mean(results[counts[-1]])
    print(f"\n{counts[-1]} frames cut noise {final:.1f}x "
          f"(~{np.log2(final ** 2):.1f} stops of exposure, for free)")


if __name__ == "__main__":
    main()
