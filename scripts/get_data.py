"""Arrange the LOL datasets into the layout the code expects.

    python scripts/get_data.py ~/Downloads/archive.zip

Kaggle needs a login, so the download itself is manual:
  https://www.kaggle.com/datasets/alenken/low-light-image-enhancement-datasets
That one archive carries both LOLv1 and LOLv2-real.

Not committed to the repo because it is ~900MB of data that is already
published elsewhere. data/real_night IS committed - those photos are
original to this project and not downloadable.
"""
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def copy_all(src_dir, dest_dir):
    for f in sorted(src_dir.glob("*.png")):
        shutil.copy(f, dest_dir / f.name)


def copy_v2(src, split, low_dest, high_dest):
    """LOLv2-real names a pair low00690.png / normal00690.png. Both sides
    become v2_00690.png: the loaders pair by filename, and the prefix keeps
    them from colliding with LOLv1's numbering."""
    for f in sorted((src / "LOLv2-real" / split / "low").glob("low*.png")):
        n = f.stem[len("low"):]
        shutil.copy(f, low_dest / f"v2_{n}.png")
        shutil.copy(src / "LOLv2-real" / split / "high" / f"normal{n}.png", high_dest / f"v2_{n}.png")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/get_data.py <path to kaggle archive.zip>")

    dirs = {name: DATA / name for name in
            ("low", "high", "eval15_low", "eval15_high", "lolv2_test_low", "lolv2_test_high")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        print("Extracting (this takes a minute)...")
        zipfile.ZipFile(sys.argv[1]).extractall(tmp)
        src = Path(tmp)

        # LOLv1: 485 training pairs + the 15-pair eval15 benchmark
        copy_all(src / "LOLv1/train/low", dirs["low"])
        copy_all(src / "LOLv1/train/high", dirs["high"])
        copy_all(src / "LOLv1/test/low", dirs["eval15_low"])
        copy_all(src / "LOLv1/test/high", dirs["eval15_high"])

        # LOLv2-real: 689 more training pairs, plus its 100-pair test split
        # kept separate as the clean benchmark
        copy_v2(src, "train", dirs["low"], dirs["high"])
        copy_v2(src, "test", dirs["lolv2_test_low"], dirs["lolv2_test_high"])

    count = lambda name: len(list(dirs[name].glob("*.png")))
    print(f"data/low, data/high:   {count('low')} training pairs (expect 1174)")
    print(f"data/eval15_*:         {count('eval15_low')} pairs (expect 15)")
    print(f"data/lolv2_test_*:     {count('lolv2_test_low')} pairs (expect 100)")


if __name__ == "__main__":
    main()
