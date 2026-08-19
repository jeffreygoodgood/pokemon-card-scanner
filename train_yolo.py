from ultralytics import YOLO

# === Config ===
DATASET_YAML = "data/detection_dataset/dataset.yaml"
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16  # adjust if GPU memory is tight

# Load YOLO11 Nano OBB — pre-trained on COCO, single class fine-tune
model = YOLO("yolo11n-obb.pt")

# Train
results = model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name="card_detector",
    patience=10,       # early stopping if no improvement for 10 epochs
    save=True,
    device=0,          # GPU
)

print("\nTraining complete!")
print(f"Best model saved to: runs/obb/card_detector/weights/best.pt")