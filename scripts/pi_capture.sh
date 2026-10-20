#!/bin/bash
# Capture a burst on a Raspberry Pi with manual exposure and RAW output.
#
# Runs ON THE PI, not the Mac. Everything this does is a direct answer to a
# ceiling measured on the webcam (docs/RESEARCH.md 9.7-9.11):
#
#   --shutter / --gain    fixed, so auto-exposure cannot lift the scene
#                         before we see it. Every webcam capture landed at
#                         0.15-0.21 brightness for exactly this reason.
#   --denoise off         the webcam's temporal NR correlated adjacent
#                         frames (rho=0.62) and capped merging at 3.55x
#                         instead of the theoretical 8.5x.
#   --raw                 12-bit DNG. The webcam's 8-bit output was a hard
#                         ceiling, and made the photon transfer curve return
#                         physically impossible values.
#
#   scripts/pi_capture.sh <outdir> [frames] [shutter_us] [gain]
#
# shutter is in MICROseconds: 100000 = 1/10s, 33000 = 1/30s, 10000 = 1/100s.
# Start long (100000) in the dark and shorten until it stops clipping.
set -e
OUT="${1:?usage: pi_capture.sh <outdir> [frames] [shutter_us] [gain]}"
N="${2:-32}"
SHUTTER="${3:-100000}"
GAIN="${4:-1.0}"

command -v rpicam-still >/dev/null && CAM=rpicam-still || CAM=libcamera-still
mkdir -p "$OUT"

echo "capturing $N frames: shutter ${SHUTTER}us, gain ${GAIN}, denoise off, raw on"
for i in $(seq -w 1 "$N"); do
  $CAM --nopreview --immediate \
       --shutter "$SHUTTER" --gain "$GAIN" --awbgains 1,1 \
       --denoise off --raw \
       -o "$OUT/f$i.jpg" 2>/dev/null
done

echo "$OUT: $(ls "$OUT"/*.dng 2>/dev/null | wc -l | tr -d ' ') raw, $(ls "$OUT"/*.jpg | wc -l | tr -d ' ') jpg"
echo
echo "Sanity-check before trusting a burst (both bit us on the webcam):"
echo "  1. frames must be UNIQUE   - md5sum $OUT/*.jpg | awk '{print \$1}' | sort -u | wc -l"
echo "  2. scene must be DARK      - if the jpgs look well-exposed, shorten --shutter"
echo
echo "Copy to the Mac, then:  python merge.py --frames <dir> --output merged.png"
