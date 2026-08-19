import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

from model import CardEmbeddingModel
from triplet_dataset import TripletCardDataset, PKSampler


# ============================================================
# Hard triplet mining
# ============================================================

def mine_hard_triplets(embeddings, labels, margin=0.3):
    """For each anchor, find the hardest positive and hardest negative.
    Returns the triplet loss averaged over all valid triplets."""
    # Distance matrix: pairwise euclidean distances between all embeddings
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    labels = labels.unsqueeze(0)
    same_card = (labels == labels.T)     # True where two images are same card
    diff_card = ~same_card               # True where two images are different cards

    loss = torch.tensor(0.0, device=embeddings.device)
    count = 0

    for i in range(len(embeddings)):
        # Hardest positive: same card, maximum distance
        pos_mask = same_card[i].clone()
        pos_mask[i] = False  # exclude self
        if not pos_mask.any():
            continue
        hardest_pos_dist = dist_matrix[i][pos_mask].max()

        # Hardest negative: different card, minimum distance
        neg_mask = diff_card[i]
        if not neg_mask.any():
            continue
        hardest_neg_dist = dist_matrix[i][neg_mask].min()

        # Triplet loss: push positive closer, negative further
        triplet_loss = torch.clamp(hardest_pos_dist - hardest_neg_dist + margin, min=0.0)
        loss += triplet_loss
        count += 1

    if count > 0:
        loss = loss / count

    return loss


# ============================================================
# Validation: compute Top-1 and Top-5 accuracy
# ============================================================

@torch.no_grad()
def build_reference_db(model, image_dir, device):
    """Build reference embeddings from all clean card images across all sets."""
    import os
    import cv2
    from augmentations import paste_card_on_background, val_transform

    model.eval()
    ref_embeddings = []
    ref_card_ids = []

    for set_id in sorted(os.listdir(image_dir)):
        set_dir = os.path.join(image_dir, set_id)
        if not os.path.isdir(set_dir):
            continue

        for filename in sorted(os.listdir(set_dir)):
            if not filename.endswith(".png"):
                continue

            card_id = f"{set_id}-{filename.replace('.png', '')}"
            image_path = os.path.join(set_dir, filename)

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bg_image = paste_card_on_background(image)
            bg_image = cv2.resize(bg_image, (224, 224))
            tensor = val_transform(image=bg_image)["image"]
            tensor = tensor.unsqueeze(0).to(device)

            embedding = model(tensor)
            ref_embeddings.append(embedding.cpu())
            ref_card_ids.append(card_id)

    ref_embeddings = torch.cat(ref_embeddings)
    print(f"Reference database: {len(ref_card_ids)} cards")
    return ref_embeddings, ref_card_ids


@torch.no_grad()
def validate(model, csv_path, image_dir, device):
    """Match each val image against the full reference database of all 99 cards.
    This simulates real usage: user photo → find closest known card."""
    import csv
    import cv2
    from augmentations import val_transform

    model.eval()

    # Build reference database from ALL clean card images
    ref_embeddings, ref_card_ids = build_reference_db(model, image_dir, device)

    # Load val card IDs from CSV
    val_card_ids = []
    val_image_paths = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "val":
                val_card_ids.append(row["card_id"])
                val_image_paths.append(row["image_path"])

    # Embed each val image
    val_embeddings = []
    for image_path in val_image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = val_transform(image=image)["image"]
        image = image.unsqueeze(0).to(device)
        embedding = model(image)
        val_embeddings.append(embedding.cpu())

    val_embeddings = torch.cat(val_embeddings)

    # Distance from each val image to each reference image
    dist_matrix = torch.cdist(val_embeddings, ref_embeddings, p=2)

    # For each val image, rank reference images by distance
    sorted_indices = dist_matrix.argsort(dim=1)

    top1_correct = 0
    top5_correct = 0

    for i in range(len(val_card_ids)):
        query_id = val_card_ids[i]
        nearest_ids = [ref_card_ids[j] for j in sorted_indices[i][:5]]

        if nearest_ids[0] == query_id:
            top1_correct += 1
        if query_id in nearest_ids:
            top5_correct += 1

    top1_acc = top1_correct / len(val_card_ids)
    top5_acc = top5_correct / len(val_card_ids)

    return top1_acc, top5_acc


# ============================================================
# Training
# ============================================================

def train():
    # === Config ===
    CSV_PATH = "data/splits/splits.csv"
    IMAGE_DIR = "data/images"
    SAVE_DIR = "checkpoints"
    EPOCHS = 150
    P, K = 8, 4          # 8 cards × 4 augmentations = batch size 32
    MARGIN = 0.3          # triplet loss margin
    LR_BACKBONE = 1e-5    # slow learning rate for pre-trained layers
    LR_HEAD = 1e-3        # faster learning rate for new embedding layer
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESUME_FROM = None  # Set to "checkpoints/best_model.pth" to continue training

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    # === Model ===
    model = CardEmbeddingModel(embedding_dim=128).to(DEVICE)

    # Load previous checkpoint if resuming training
    if RESUME_FROM is not None:
        model.load_state_dict(torch.load(RESUME_FROM, map_location=DEVICE))
        print(f"Resumed from {RESUME_FROM}")

    # Freeze early backbone layers (first 5 of 9 blocks)
    # Only fine-tune the last 4 blocks + embedding head
    for i, block in enumerate(model.backbone):
        if i < 5:
            for param in block.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # === Optimizer with different learning rates ===
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.embedding.parameters())

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ])

    # Reduce learning rate when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # === Dataset and loader ===
    train_dataset = TripletCardDataset(CSV_PATH, split="train")
    sampler = PKSampler(train_dataset, p=P, k=K)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=16)

    # === Training loop ===
    best_top1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward: get embeddings
            embeddings = model(images)

            # Compute triplet loss with hard mining
            loss = mine_hard_triplets(embeddings, labels, margin=MARGIN)

            # Backward: update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - start_time

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            top1, top5 = validate(model, CSV_PATH, IMAGE_DIR, DEVICE)
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Top-1: {top1:.2%} | Top-5: {top5:.2%} | Time: {elapsed:.1f}s")

            # Save best model
            if top1 > best_top1:
                best_top1 = top1
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
                print(f"  → Saved new best model (Top-1: {top1:.2%})")
        else:
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        # Step scheduler based on loss
        scheduler.step(avg_loss)

    # Save final model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
    print(f"\nTraining complete. Best Top-1: {best_top1:.2%}")


if __name__ == "__main__":
    train()