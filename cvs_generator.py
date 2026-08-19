import os
import random
import csv

# === Config ===
DATA_DIR = "data/images"  # parent directory containing all set folders
SPLITS_DIR = "data/splits"
SEED = 42

# === Collect all images from all set subfolders ===
all_images = []
for set_id in sorted(os.listdir(DATA_DIR)):
    set_dir = os.path.join(DATA_DIR, set_id)
    if not os.path.isdir(set_dir):
        continue

    for filename in sorted(os.listdir(set_dir)):
        if not filename.endswith(".png"):
            continue
        card_id = f"{set_id}-{filename.replace('.png', '')}"
        image_path = os.path.join(set_dir, filename)
        all_images.append({
            "card_id": card_id,
            "image_path": image_path,
            "set_id": set_id,
        })

print(f"Found {len(all_images)} total cards across all sets")

# Count per set
sets = {}
for img in all_images:
    sets[img["set_id"]] = sets.get(img["set_id"], 0) + 1
for set_id, count in sorted(sets.items()):
    print(f"  {set_id}: {count} cards")

# === All cards go to training, sample 10% for validation monitoring ===
random.seed(SEED)
random.shuffle(all_images)

train_images = all_images
val_sample_count = max(1, int(len(all_images) * 0.10))
val_images = random.sample(all_images, val_sample_count)

print(f"\nTrain: {len(train_images)} cards (all cards)")
print(f"Val:   {len(val_images)} cards (sampled from training for monitoring)")

# === Write CSV ===
os.makedirs(SPLITS_DIR, exist_ok=True)
csv_path = os.path.join(SPLITS_DIR, "splits.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["card_id", "image_path", "set_id", "split"])

    # All cards go into training
    for img in train_images:
        writer.writerow([img["card_id"], img["image_path"], img["set_id"], "train"])

    # A sample also goes into val for monitoring (same cards, duplicated)
    for img in val_images:
        writer.writerow([img["card_id"], img["image_path"], img["set_id"], "val"])

print(f"\nCSV saved to {csv_path}")