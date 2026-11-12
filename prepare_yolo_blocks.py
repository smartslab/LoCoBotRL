
#!/usr/bin/env python3
"""Prepare labeled LoCoBot block images for YOLO training."""

import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source", default="block_dataset"
    )
    parser.add_argument(
        "--destination", default="blocks_yolo"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )

    args = parser.parse_args()

    if not 0 < args.val_fraction < 1:
        parser.error(
            "Validation fraction must be between 0 and 1"
        )

    src = Path(args.source)
    dest = Path(args.destination).resolve()

    files = sorted((src / "images").glob("*.jpg"))

    labeled = [
        f for f in files
        if (src / "labels" / (f.stem + ".txt")).exists()
    ]

    if len(labeled) < 2:
        parser.error(
            "Need at least two manually labeled images"
        )

    random.Random(args.seed).shuffle(labeled)

    n_val = max(
        1, round(len(labeled) * args.val_fraction)
    )

    for i, f in enumerate(labeled):
        split = "val" if i < n_val else "train"

        for kind in ("images", "labels"):
            (dest / kind / split).mkdir(
                parents=True, exist_ok=True
            )

        shutil.copy2(
            f,
            dest / "images" / split / f.name
        )

        shutil.copy2(
            src / "labels" / (f.stem + ".txt"),
            dest / "labels" / split / (f.stem + ".txt")
        )

    yaml_content = (
        "path: " + str(dest) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: block\n"
    )

    (dest / "data.yaml").write_text(yaml_content)

    print(
        "Exported %d labeled frames" % len(labeled)
    )
    print(
        "Training: %d" % (len(labeled) - n_val)
    )
    print(
        "Validation: %d" % n_val
    )
    print(
        "Unlabeled frames skipped: %d" %
        (len(files) - len(labeled))
    )
    print(
        "Dataset:", dest / "data.yaml"
    )


if __name__ == "__main__":
    main()
