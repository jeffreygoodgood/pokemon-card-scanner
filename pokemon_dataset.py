import csv
import cv2
from torch.utils.data import Dataset, DataLoader
from augmentations import (
    build_train_transform,
    paste_card_on_background,
    final_transform,
    val_transform,
)


class PokemonCardDataset(Dataset):
    def __init__(self, csv_path, split="train"):
        self.samples = []
        self.label_to_idx = {}
        self.split = split

        # Read CSV and filter by split
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.samples.append({
                        "card_id": row["card_id"],
                        "image_path": row["image_path"],
                        "set_id": row["set_id"],
                    })

        # Create a numeric label for each unique card_id
        unique_ids = sorted(set(s["card_id"] for s in self.samples))
        self.label_to_idx = {cid: idx for idx, cid in enumerate(unique_ids)}

        print(f"[{split.upper()}] Loaded {len(self.samples)} images, "
              f"{len(self.label_to_idx)} unique cards")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image with OpenCV and convert BGR -> RGB
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.split == "train":
            # Step 1: Paste card on noisy background
            transform = build_train_transform()
            image = paste_card_on_background(image)

            # Step 2: Augmentation (geometric, color temp, exposure, shadow, noise)
            image = transform(image=image)["image"]

            # Step 3: Normalize + convert to tensor
            image = final_transform(image=image)["image"]
        else:
            # Validation: just resize, normalize, and convert to tensor
            image = val_transform(image=image)["image"]

        label = self.label_to_idx[sample["card_id"]]
        return image, label


# === Quick test ===
if __name__ == "__main__":
    csv_path = "data/splits/splits.csv"

    train_dataset = PokemonCardDataset(csv_path, split="train")
    val_dataset = PokemonCardDataset(csv_path, split="val")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    images, labels = next(iter(train_loader))

    print(f"\nBatch shape: {images.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Labels:      {labels.tolist()}")
    print(f"Pixel range: [{images.min():.2f}, {images.max():.2f}]")