
#!/usr/bin/env python3

from ultralytics import YOLO
from pathlib import Path
import torch

# --------------------------------
# Configuration
# --------------------------------
DATASET = "blocks_yolo/data.yaml"
MODEL = "yolov8n.pt"

EPOCHS = 100
IMAGE_SIZE = 1280
BATCH_SIZE = 2
DEVICE = "cpu"

# --------------------------------
# Check dataset
# --------------------------------
if not Path(DATASET).exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET}\n"
        "Run prepare_yolo_blocks.py first."
    )

# --------------------------------
# Configure CPU
# --------------------------------
torch.set_num_threads(4)

print("Training on CPU")
print("Dataset:", DATASET)
print("Image size:", IMAGE_SIZE)

# --------------------------------
# Load pretrained YOLO
# --------------------------------
model = YOLO(MODEL)

# --------------------------------
# Train
# --------------------------------
model.train(
    data=DATASET,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=0,
    patience=20,
    project="runs/block_training",
    name="yolov8n_cpu",
    exist_ok=True,
    plots=True,
)

# --------------------------------
# Validate best model
# --------------------------------
best_path = "runs/block_training/yolov8n_cpu/weights/best.pt"

print("\nTraining complete!")
print("Best weights:", best_path)

best_model = YOLO(best_path)
best_model.val(
    data=DATASET,
    imgsz=IMAGE_SIZE,
    device="cpu",
    batch=1,
    workers=0,
)

print("\nYour custom block detector is ready!")
