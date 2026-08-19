"""
FastAPI server for Pokémon card identification.
Deployed on Hugging Face Spaces.

Endpoint:
  POST /identify — accepts an image, returns the matched card + metadata from TCGDex
  GET  /health   — server status check
"""

import os
import pickle
import asyncio
import cv2
import torch
import numpy as np
import httpx
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from model import CardEmbeddingModel
from augmentations import val_transform

# === Paths (relative to /app inside the container) ===
YOLO_PATH = "checkpoints/detection.pt"
EMBED_PATH = "checkpoints/final_model.pth"
DB_PATH = "data/reference_db.pkl"
IMAGE_DIR = "data/images"

# === TCGDex API ===
TCGDEX_BASE_URL = "https://api.tcgdex.net/v2/en/cards"


# ============================================================
# Detection helpers (from Inference.py)
# ============================================================

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


def detect_and_crop(yolo_model, image_rgb):
    """Use YOLO to detect and crop the card from an RGB image array.
    Returns the cropped card image (RGB), or None if no card found."""
    results = yolo_model(image_rgb, verbose=False)
    result = results[0]

    if result.obb is None or len(result.obb) == 0:
        return None

    best = result.obb[result.obb.conf.argmax()]
    points = best.xyxyxyxy.cpu().numpy().reshape(4, 2).astype(np.float32)
    ordered = order_corners(points)

    width = int(np.linalg.norm(ordered[1] - ordered[0]))
    height = int(np.linalg.norm(ordered[3] - ordered[0]))

    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    cropped = cv2.warpPerspective(image_rgb, matrix, (width, height))

    return cropped


def identify_card(embed_model, cropped_image, ref_embeddings, device):
    """Get embedding for cropped card and find nearest match in reference DB."""
    with torch.no_grad():
        tensor = val_transform(image=cropped_image)["image"].unsqueeze(0).to(device)
        query_embedding = embed_model(tensor).cpu()

    distances = torch.cdist(query_embedding, ref_embeddings, p=2).squeeze(0)
    top5_distances, top5_indices = distances.topk(5, largest=False)

    return top5_indices.tolist(), top5_distances.tolist()


# Persistent HTTP client
http_client: httpx.AsyncClient | None = None


async def fetch_card_metadata(card_id: str) -> dict:
    """Fetch card metadata from TCGDex API."""
    url = f"{TCGDEX_BASE_URL}/{card_id}"
    try:
        response = await http_client.get(url, timeout=5.0)
        if response.status_code == 200:
            return response.json()
        else:
            return {"id": card_id, "error": f"TCGDex returned {response.status_code}"}
    except Exception as e:
        return {"id": card_id, "error": f"TCGDex request failed: {str(e)}"}


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(title="Pokémon Card Identifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model/data references
yolo_model = None
embed_model = None
ref_embeddings = None
ref_card_ids = None
DEVICE = None


@app.on_event("startup")
def load_models():
    """Load all models and reference data once when the server starts."""
    global yolo_model, embed_model, ref_embeddings, ref_card_ids, DEVICE, http_client

    # Create persistent HTTP client
    http_client = httpx.AsyncClient()

    # CPU only on Hugging Face free tier
    DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")

    # Load YOLO detector
    yolo_model = YOLO(YOLO_PATH)
    print(f"YOLO loaded from {YOLO_PATH}")

    # Load EfficientNet embedding model
    embed_model = CardEmbeddingModel(embedding_dim=128).to(DEVICE)
    embed_model.load_state_dict(torch.load(EMBED_PATH, map_location=DEVICE))
    embed_model.eval()
    print(f"EfficientNet loaded from {EMBED_PATH}")

    # Load reference database
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)
    ref_embeddings = db["embeddings"]
    ref_card_ids = db["card_ids"]
    print(f"Reference DB loaded: {len(ref_card_ids)} cards")

    print("\nServer ready!")


@app.post("/identify")
async def identify(image: UploadFile = File(...)):
    """Accept an uploaded photo, identify the card, fetch metadata from TCGDex."""

    contents = await image.read()
    np_array = np.frombuffer(contents, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {"success": False, "error": "Could not decode image"}

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Step 1: YOLO detection and crop
    cropped = detect_and_crop(yolo_model, img_rgb)

    if cropped is None:
        return {"success": False, "error": "No card detected in image"}

    # Step 2: Identify card via embedding matching
    top5_indices, top5_distances = identify_card(
        embed_model, cropped, ref_embeddings, DEVICE
    )

    # Step 3: Fetch metadata from TCGDex for ALL top-5 concurrently
    top5_card_ids = [ref_card_ids[idx] for idx in top5_indices]
    all_metadata = await asyncio.gather(
        *[fetch_card_metadata(card_id) for card_id in top5_card_ids]
    )

    # Best match
    best_card_id = top5_card_ids[0]
    best_distance = top5_distances[0]
    best_metadata = all_metadata[0]

    # Build top-5 list
    top5_results = []
    for card_id, dist, meta in zip(top5_card_ids, top5_distances, all_metadata):
        top5_results.append({
            "card_id": card_id,
            "name": meta.get("name", "Unknown"),
            "image": meta.get("image", None),
            "distance": round(dist, 4),
        })

    return {
        "success": True,
        "match": {
            "card_id": best_card_id,
            "distance": round(best_distance, 4),
            **best_metadata,
        },
        "top5": top5_results,
    }


@app.get("/card/{card_id:path}")
async def get_card(card_id: str):
    """Fetch full metadata for a specific card from TCGDex."""
    metadata = await fetch_card_metadata(card_id)
    return metadata


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "cards_loaded": len(ref_card_ids) if ref_card_ids else 0,
        "device": str(DEVICE),
    }