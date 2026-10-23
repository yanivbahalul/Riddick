#!/bin/bash
# Arrange the LOL datasets into the layout the code expects.
#
#   scripts/get_data.sh ~/Downloads/archive.zip
#
# Kaggle needs a login, so the download itself is manual:
#   https://www.kaggle.com/datasets/alenken/low-light-image-enhancement-datasets
# That one archive carries both LOLv1 and LOLv2-real.
#
# Not committed to the repo because it is ~900MB of data that is already
# published elsewhere. data/real_night* IS committed - those are original
# to this project and not downloadable.
set -e
ARCHIVE="${1:?usage: get_data.sh <path to kaggle archive.zip>}"
cd "$(dirname "$0")/.."

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "Extracting (this takes a minute)..."
unzip -q "$ARCHIVE" -d "$TMP"

mkdir -p data/{low,high,eval15_low,eval15_high,lolv2_test_low,lolv2_test_high}

# LOLv1: 485 training pairs + the 15-pair eval15 benchmark
cp "$TMP"/LOLv1/train/low/*.png   data/low/
cp "$TMP"/LOLv1/train/high/*.png  data/high/
cp "$TMP"/LOLv1/test/low/*.png    data/eval15_low/
cp "$TMP"/LOLv1/test/high/*.png   data/eval15_high/

# LOLv2-real: 689 more training pairs, prefixed to avoid colliding with LOLv1's
# numbering, plus its 100-pair test split kept separate as the clean benchmark.
# The archive names these low00690.png / normal00690.png - both sides are
# renamed to the same v2_<number>.png because the loaders pair by filename.
for f in "$TMP"/LOLv2-real/train/low/low*.png; do
  n=$(basename "$f" .png); n=${n#low}
  cp "$f" "data/low/v2_${n}.png"
  cp "$TMP/LOLv2-real/train/high/normal${n}.png" "data/high/v2_${n}.png"
done
for f in "$TMP"/LOLv2-real/test/low/low*.png; do
  n=$(basename "$f" .png); n=${n#low}
  cp "$f" "data/lolv2_test_low/v2_${n}.png"
  cp "$TMP/LOLv2-real/test/high/normal${n}.png" "data/lolv2_test_high/v2_${n}.png"
done

printf '%-22s %s\n' \
  "data/low, data/high:" "$(ls data/low | wc -l | tr -d ' ') training pairs (expect 1174)" \
  "data/eval15_*:"       "$(ls data/eval15_low | wc -l | tr -d ' ') pairs (expect 15)" \
  "data/lolv2_test_*:"   "$(ls data/lolv2_test_low | wc -l | tr -d ' ') pairs (expect 100)"
