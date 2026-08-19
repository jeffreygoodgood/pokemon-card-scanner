import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def order_corners(pts):
    """Order four corners as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect

# === Config ===
MODEL_PATH = "checkpoints/best.pt"
TEST_DIR = "testcard"

# === Load YOLO model ===
model = YOLO(MODEL_PATH)

# === Test on all images ===
test_files = sorted([
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
])

for filename in test_files:
    path = os.path.join(TEST_DIR, filename)
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    results = model(path, verbose=False)
    result = results[0]

    if result.obb is None or len(result.obb) == 0:
        print(f"{filename}: No card detected")
        continue

    # Take highest confidence detection
    best = result.obb[result.obb.conf.argmax()]
    conf = best.conf.item()
    points = best.xyxyxyxy.cpu().numpy().reshape(4, 2).astype(np.float32)
    ordered = order_corners(points)

    # Crop using ordered corners
    width = int(np.linalg.norm(ordered[1] - ordered[0]))
    height = int(np.linalg.norm(ordered[3] - ordered[0]))

    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    cropped = cv2.warpPerspective(image, matrix, (width, height))

    # Show result
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    annotated = image.copy()
    cv2.polylines(annotated, [points.astype(np.int32)], True, (0, 255, 0), 3)
    axes[0].imshow(annotated)
    axes[0].set_title(f"{filename} (conf={conf:.2f})")
    axes[0].axis("off")
    axes[1].imshow(cropped)
    axes[1].set_title(f"Cropped ({cropped.shape[1]}x{cropped.shape[0]})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()