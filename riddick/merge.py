"""Multi-frame merge: the step that adds real signal instead of guessing it.

Averaging N frames of the same scene cuts noise by ~sqrt(N), because shot
and read noise are independent per frame while the scene is not. This is
the core of what Night Sight / HDR+ do, and it is the one thing a
single-image model fundamentally cannot do.

A static camera has an advantage a phone does not: no hand shake, so it
can integrate far more frames. See docs/RESEARCH.md sections 2 and 9.3.

Usage:
    python -m riddick.merge --frames path/to/frame_dir --output merged.png
    python -m riddick.merge --video clip.mp4 --num-frames 16 --output merged.png
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from riddick.common import list_images


def estimate_shift(ref_gray, img_gray):
    """Whole-pixel (dy, dx) that best aligns img onto ref, via phase correlation.

    Phase correlation works in the frequency domain: a spatial shift is a
    phase ramp in the FFT, so normalizing away magnitude and inverting
    leaves a single sharp peak at the shift. It is robust to the brightness
    differences between frames, which plain cross-correlation is not.
    """
    f_ref = np.fft.fft2(ref_gray)
    f_img = np.fft.fft2(img_gray)
    cross = f_ref * np.conj(f_img)
    cross /= np.abs(cross) + 1e-8          # keep phase, discard magnitude
    corr = np.fft.ifft2(cross).real

    peak = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = ref_gray.shape
    # the FFT wraps, so a peak past the midpoint means a negative shift
    dy = peak[0] - h if peak[0] > h // 2 else peak[0]
    dx = peak[1] - w if peak[1] > w // 2 else peak[1]
    return dy, dx


def merge_frames(frames, align=True, reference=0):
    """Average a list of HxWx3 float arrays, optionally aligning first.

    Returns (merged, shifts). Averaging is the simplest merge and is
    optimal for independent zero-mean noise; it does ghost anything that
    moves, which is what motion masking would address later.
    """
    ref = frames[reference]
    ref_gray = ref.mean(axis=2)

    aligned, shifts = [], []
    for i, f in enumerate(frames):
        if not align or i == reference:
            aligned.append(f)
            shifts.append((0, 0))
            continue
        dy, dx = estimate_shift(ref_gray, f.mean(axis=2))
        aligned.append(np.roll(f, (dy, dx), axis=(0, 1)))
        shifts.append((dy, dx))

    return np.mean(aligned, axis=0), shifts


def frames_from_video(path, num_frames):
    """Pull the first num_frames frames out of a video with ffmpeg."""
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path),
         "-vframes", str(num_frames), str(tmp / "f%04d.png")],
        check=True,
    )
    return sorted(tmp.glob("*.png"))


def load(paths):
    """Load frames as float RGB in [0,1]. Handles 8-bit images and RAW/DNG.

    RAW matters here beyond convenience: the webcam's 8-bit output was a
    hard ceiling on what merging could recover, and its already-processed
    pixels made the sensor impossible to characterise (docs/RESEARCH.md
    9.8, 9.10). DNG from the Pi is 12-bit and unprocessed, so it is read
    without white balance or auto-brightness applied - we want the sensor's
    numbers, not a rendering decision made on our behalf.
    """
    out = []
    for p in paths:
        if str(p).lower().endswith((".dng", ".arw", ".nef", ".cr2")):
            import rawpy
            with rawpy.imread(str(p)) as raw:
                rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True,
                                      use_camera_wb=False, output_bps=16)
            out.append(rgb.astype(np.float64) / 65535.0)
        else:
            out.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple frames of one scene to cut noise")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", type=str, help="Directory of already-extracted frames")
    src.add_argument("--video", type=str, help="Video file; frames are extracted with ffmpeg")
    parser.add_argument("--num-frames", type=int, default=16, help="How many frames to merge")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--no-align", action="store_true", help="Skip alignment (valid only for a locked-off camera)")
    args = parser.parse_args()

    if args.video:
        paths = frames_from_video(args.video, args.num_frames)
    else:
        paths = list_images(args.frames)[:args.num_frames]

    frames = load(paths)
    merged, shifts = merge_frames(frames, align=not args.no_align)

    moved = [s for s in shifts if s != (0, 0)]
    print(f"Merged {len(frames)} frames; {len(moved)} needed alignment"
          + (f", max shift {max(max(abs(a), abs(b)) for a, b in moved)}px" if moved else ""))
    Image.fromarray((merged.clip(0, 1) * 255).astype(np.uint8)).save(args.output)
    print(f"Saved {args.output}")
