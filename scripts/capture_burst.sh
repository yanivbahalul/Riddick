#!/bin/bash
# Capture a burst of real frames from a connected camera, for merge testing.
#
#   scripts/capture_burst.sh <outdir> [seconds] [device] [size] [fps]
#
# Records to a raw .mov first and extracts afterwards. Writing PNGs directly
# from avfoundation makes ffmpeg duplicate a single frame to fill the count
# (look for "dup=" in its output) - the burst comes out as N identical
# copies, which merges to exactly nothing. Learned that the hard way.
set -e
OUT="${1:?usage: capture_burst.sh <outdir> [seconds] [device] [size] [fps]}"
SECS="${2:-5}"
DEV="${3:-0}"
SIZE="${4:-1920x1080}"
FPS="${5:-30}"

rm -rf "$OUT" && mkdir -p "$OUT"
TMP="$(mktemp -d)/burst.mov"

ffmpeg -loglevel error -f avfoundation -pixel_format uyvy422 \
       -framerate "$FPS" -video_size "$SIZE" -i "$DEV" \
       -t "$SECS" -c:v copy "$TMP"
ffmpeg -loglevel error -i "$TMP" "$OUT/f%04d.png"
rm -f "$TMP"

N=$(ls "$OUT" | wc -l | tr -d ' ')
UNIQ=$(md5 -q "$OUT"/*.png | sort -u | wc -l | tr -d ' ')
echo "$OUT: $N frames, $UNIQ unique ($SIZE @ ${FPS}fps, ${SECS}s)"
[ "$UNIQ" -lt "$N" ] && echo "WARNING: duplicate frames - merging them buys nothing"
exit 0
