import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from model import CardEmbeddingModel
from augmentations import paste_card_on_background, val_transform


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


def detect_and_crop(yolo_model, image_path):
    """Use YOLO to detect and crop the card from a photo."""
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    results = yolo_model(image_path, verbose=False)
    result = results[0]

    if result.obb is None or len(result.obb) == 0:
        return None, image

    best = result.obb[result.obb.conf.argmax()]
    points = best.xyxyxyxy.cpu().numpy().reshape(4, 2).astype(np.float32)
    ordered = order_corners(points)

    width = int(np.linalg.norm(ordered[1] - ordered[0]))
    height = int(np.linalg.norm(ordered[3] - ordered[0]))

    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    cropped = cv2.warpPerspective(image, matrix, (width, height))

    return cropped, image


def build_reference_db(embed_model, image_dir, device):
    """Build reference embeddings on constant background across all sets."""
    embed_model.eval()
    ref_embeddings = []
    ref_paths = []

    with torch.no_grad():
        for set_id in sorted(os.listdir(image_dir)):
            set_dir = os.path.join(image_dir, set_id)
            if not os.path.isdir(set_dir):
                continue

            for f in sorted(os.listdir(set_dir)):
                if not f.endswith(".png"):
                    continue
                path = os.path.join(set_dir, f)
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                bg_image = paste_card_on_background(img)
                bg_image = cv2.resize(bg_image, (224, 224))
                tensor = val_transform(image=bg_image)["image"].unsqueeze(0).to(device)
                emb = embed_model(tensor).cpu()
                ref_embeddings.append(emb)
                ref_paths.append(path)

                del tensor
                torch.cuda.empty_cache()

            print(f"  Loaded {set_id}")

    return torch.cat(ref_embeddings), ref_paths


# === Config ===
YOLO_PATH = "checkpoints/detection.pt"
EMBED_PATH = "checkpoints/best_model.pth"
IMAGE_DIR = "data/images"
TEST_DIR = "testcard/sample5"
TOP_K = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load models ===
yolo_model = YOLO(YOLO_PATH)
embed_model = CardEmbeddingModel(embedding_dim=128).to(DEVICE)
embed_model.load_state_dict(torch.load(EMBED_PATH, map_location=DEVICE))
embed_model.eval()

# === Build reference database ===
print("Building reference database...")
ref_embeddings, ref_paths = build_reference_db(embed_model, IMAGE_DIR, DEVICE)
print(f"Database: {len(ref_paths)} cards loaded")

# === Process all test images ===
test_files = sorted([
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
])

print(f"Found {len(test_files)} test images\n")

for filename in test_files:
    query_path = os.path.join(TEST_DIR, filename)

    # Step 1: YOLO detects and crops the card
    cropped, original = detect_and_crop(yolo_model, query_path)

    if cropped is None:
        print(f"{filename}: No card detected, skipping")
        continue

    # Step 2: EfficientNet matches the cropped card
    with torch.no_grad():
        query_tensor = val_transform(image=cropped)["image"].unsqueeze(0).to(DEVICE)
        query_embedding = embed_model(query_tensor).cpu()
        del query_tensor
        torch.cuda.empty_cache()

    distances = torch.cdist(query_embedding, ref_embeddings, p=2).squeeze(0)
    sorted_indices = distances.argsort()

    # Get top 5 matches
    top_paths = [ref_paths[sorted_indices[i].item()] for i in range(TOP_K)]
    top_dists = [distances[sorted_indices[i].item()].item() for i in range(TOP_K)]

    # Step 3: Show original → top 5 matches
    fig, axes = plt.subplots(1, TOP_K + 1, figsize=(4 * (TOP_K + 1), 5))

    axes[0].imshow(original)
    axes[0].set_title(f"Photo: {filename}", fontsize=8)
    axes[0].axis("off")

    for i in range(TOP_K):
        match_img = cv2.cvtColor(cv2.imread(top_paths[i]), cv2.COLOR_BGR2RGB)
        axes[i + 1].imshow(match_img)
        rank_label = f"#{i+1}: {os.path.basename(top_paths[i])}\nd={top_dists[i]:.3f}"
        color = "green" if i == 0 else "gray"
        axes[i + 1].set_title(rank_label, fontsize=8, color=color)
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()