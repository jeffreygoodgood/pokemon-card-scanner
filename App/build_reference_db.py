"""
Build reference database for the card identification server.
Run this ONCE after training is complete (or whenever you add new card sets).

Outputs: /home/jeff/Desktop/PokemonExplorer/data/reference_db.pkl
  - embeddings: Tensor of shape [N, 128] — one embedding per card
  - card_ids:   List of card ID strings (e.g., "sv03.5-001")

Metadata is fetched from TCGDex API at runtime, so it's not stored here.

Usage:
    python build_reference_db.py
"""

import sys
import os
import pickle
import cv2
import torch

# Add project root so we can import model.py and augmentations.py
PROJECT_ROOT = "/home/jeff/Desktop/PokemonExplorer"
sys.path.insert(0, PROJECT_ROOT)

from model import CardEmbeddingModel
from augmentations import paste_card_on_background, val_transform

# === Paths ===
EMBED_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "final_model.pth")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "images")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "reference_db.pkl")


def build_reference_db(embed_model, device):
    """Build reference embeddings for all cards across all sets."""
    embed_model.eval()
    ref_embeddings = []
    ref_card_ids = []

    with torch.no_grad():
        for set_id in sorted(os.listdir(IMAGE_DIR)):
            set_dir = os.path.join(IMAGE_DIR, set_id)
            if not os.path.isdir(set_dir):
                continue

            count = 0
            for f in sorted(os.listdir(set_dir)):
                if not f.endswith(".png"):
                    continue

                # Card ID: "sv03.5-001"
                local_id = f.replace(".png", "")
                card_id = f"{set_id}-{local_id}"

                # Embedding (same logic as Inference.py)
                path = os.path.join(set_dir, f)
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                bg_image = paste_card_on_background(img)
                bg_image = cv2.resize(bg_image, (224, 224))
                tensor = val_transform(image=bg_image)["image"].unsqueeze(0).to(device)
                emb = embed_model(tensor).cpu()
                ref_embeddings.append(emb)
                ref_card_ids.append(card_id)

                del tensor
                torch.cuda.empty_cache()
                count += 1

            print(f"  {set_id}: {count} cards processed")

    embeddings_tensor = torch.cat(ref_embeddings)
    return embeddings_tensor, ref_card_ids


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load trained EfficientNet
    model = CardEmbeddingModel(embedding_dim=128).to(DEVICE)
    model.load_state_dict(torch.load(EMBED_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from {EMBED_PATH}")

    # Build the database
    print("Building reference database...")
    embeddings, card_ids = build_reference_db(model, DEVICE)

    # Save to pickle
    db = {
        "embeddings": embeddings,
        "card_ids": card_ids,
    }
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(db, f)

    print(f"\nSaved reference database to {OUTPUT_PATH}")
    print(f"  Cards: {len(card_ids)}")
    print(f"  Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
