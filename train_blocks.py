#!/usr/bin/env python3

import os

# Limit CPU threading
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


# =====================================================
# CONFIGURATION
# =====================================================

DATASET = Path("blocks_yolo/data.yaml").resolve()
MODEL = "yolov8n.pt"

# Start with 3 epochs to verify training works.
# Change to 100 after confirming losses are finite.
EPOCHS = 3

IMAGE_SIZE = 1280
BATCH_SIZE = 2

DEVICE = "cpu"
LEARNING_RATE = 0.0001

PROJECT = "runs/block_training"
RUN_NAME = "cpu_fixed"

torch.set_num_threads(4)


# =====================================================
# STEP 1: VALIDATE DATASET
# =====================================================

def validate_dataset():

    print("\n===================================")
    print("CHECKING DATASET")
    print("===================================\n")

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset configuration missing: {DATASET}"
        )

    import yaml

    with open(DATASET) as f:
        config = yaml.safe_load(f)

    dataset_root = Path(
        config.get("path", DATASET.parent)
    ).expanduser()

    if not dataset_root.is_absolute():
        dataset_root = (
            DATASET.parent / dataset_root
        ).resolve()

    total_images = 0
    total_boxes = 0
    problems = []

    for split in ["train", "val"]:

        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Missing image directory: {image_dir}"
            )

        image_files = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower()
            in [".jpg", ".jpeg", ".png"]
        )

        print(f"{split}: {len(image_files)} images")

        for image_path in image_files:

            total_images += 1

            # Check whether image can be opened
            try:
                with Image.open(image_path) as img:
                    img.verify()

                with Image.open(image_path) as img:
                    width, height = img.size

                    if width < 32 or height < 32:
                        problems.append(
                            f"{image_path}: Image too small"
                        )

            except Exception as e:
                problems.append(
                    f"{image_path}: Invalid image ({e})"
                )
                continue

            # Find corresponding annotation
            label_path = (
                label_dir / (image_path.stem + ".txt")
            )

            if not label_path.exists():
                problems.append(
                    f"{image_path.name}: Missing label file"
                )
                continue

            # Check YOLO labels
            with open(label_path) as f:
                lines = f.readlines()

            for line_number, line in enumerate(lines, 1):

                line = line.strip()

                if not line:
                    continue

                try:

                    values = [float(x) for x in line.split()]

                    if len(values) != 5:
                        raise ValueError(
                            "Expected 5 values"
                        )

                    cls, x, y, w, h = values

                    # Check finite values
                    if not all(
                        math.isfinite(v) for v in values
                    ):
                        raise ValueError(
                            "NaN or infinity in label"
                        )

                    # Single-class detection
                    if cls != 0:
                        raise ValueError(
                            f"Invalid class ID: {cls}"
                        )

                    # Normalized coordinates
                    if not (
                        0 <= x <= 1
                        and 0 <= y <= 1
                        and 0 < w <= 1
                        and 0 < h <= 1
                    ):
                        raise ValueError(
                            "Invalid normalized coordinates"
                        )

                    # Check bounding box boundaries
                    tolerance = 1e-5

                    if (
                        x - w / 2 < -tolerance
                        or x + w / 2 > 1 + tolerance
                        or y - h / 2 < -tolerance
                        or y + h / 2 > 1 + tolerance
                    ):
                        raise ValueError(
                            "Bounding box exceeds image"
                        )

                    total_boxes += 1

                except Exception as e:

                    problems.append(
                        f"{label_path.name}, "
                        f"line {line_number}: {e}"
                    )

    print("\nTotal images:", total_images)
    print("Total bounding boxes:", total_boxes)
    print("Problems detected:", len(problems))

    if problems:

        print("\nDATASET ERRORS:\n")

        for problem in problems[:30]:
            print(" -", problem)

        raise RuntimeError(
            "Dataset validation failed. "
            "Fix these issues before training."
        )

    if total_images == 0 or total_boxes == 0:
        raise RuntimeError(
            "No usable training images or boxes."
        )

    print("\nDataset checks passed!")
    print("Note: Visual annotation accuracy must")
    print("still be verified manually.\n")


# =====================================================
# STEP 2: STOP ON NaN / INF LOSSES
# =====================================================

def check_training_loss(trainer):

    loss_items = getattr(trainer, "loss_items", None)

    if loss_items is None:
        loss_items = getattr(trainer, "loss", None)

    if loss_items is None:
        return

    if isinstance(loss_items, torch.Tensor):
        values = loss_items.detach().cpu().numpy()
    else:
        values = np.asarray(loss_items)

    if not np.all(np.isfinite(values)):

        print("\n===================================")
        print("ERROR: NaN / INF LOSS DETECTED")
        print("===================================")

        print("Loss values:", values)
        print("Stopping training immediately.")

        raise RuntimeError(
            "Training stopped: non-finite loss. "
            "Check dataset and numerical stability."
        )


# =====================================================
# STEP 3: TRAIN
# =====================================================

def train():

    print("\n===================================")
    print("CPU-ONLY YOLO TRAINING")
    print("===================================\n")

    print("Dataset:", DATASET)
    print("Model:", MODEL)
    print("Epochs:", EPOCHS)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Device:", DEVICE)

    # Load clean pretrained weights
    model = YOLO(MODEL)

    # Check losses after each batch
    model.add_callback(
        "on_train_batch_end",
        check_training_loss
    )

    print("\nStarting training...\n")

    results = model.train(

        data=str(DATASET),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        device="cpu",

        workers=0,

        # Disable mixed precision
        amp=False,

        # Explicit optimizer
        optimizer="AdamW",

        # Conservative learning rate
        lr0=LEARNING_RATE,

        # Reduce learning-rate schedule complexity
        lrf=0.1,

        # Prevent extreme gradient updates
        # (uses Ultralytics' default clipping behavior)

        # Disable strong augmentations temporarily
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        # Disable most geometric augmentation
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,

        # No flips during initial debugging
        fliplr=0.0,
        flipud=0.0,

        # Training output
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,

        # Early stopping
        patience=20,

        # Validation
        val=True,

        # Debugging plots
        plots=True,

        verbose=True,

        seed=42,
        deterministic=True,
    )

    print("\n===================================")
    print("TRAINING COMPLETED")
    print("===================================\n")

    best_path = (
        Path(PROJECT)
        / RUN_NAME
        / "weights"
        / "best.pt"
    )

    print("Best model:", best_path)

    return results


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    validate_dataset()

    train()
