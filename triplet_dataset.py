import csv
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from augmentations import build_train_transform, paste_card_on_background, final_transform, val_transform


class TripletCardDataset(Dataset):
    """Dataset that returns multiple augmented versions of the same card.
    Each __getitem__ call generates a fresh augmentation of the requested card."""

    def __init__(self, csv_path, split="train"):
        self.split = split
        self.samples = []       # list of {card_id, image_path, set_id}
        self.label_to_idx = {}  # card_id string → numeric label
        self.label_to_samples = {}  # numeric label → list of sample indices

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

        # Create numeric labels
        unique_ids = sorted(set(s["card_id"] for s in self.samples))
        self.label_to_idx = {cid: idx for idx, cid in enumerate(unique_ids)}

        # Map each label to its sample indices (for the sampler)
        for i, sample in enumerate(self.samples):
            label = self.label_to_idx[sample["card_id"]]
            if label not in self.label_to_samples:
                self.label_to_samples[label] = []
            self.label_to_samples[label].append(i)

        self.labels = [self.label_to_idx[s["card_id"]] for s in self.samples]

        print(f"[{split.upper()}] Loaded {len(self.samples)} images, "
              f"{len(self.label_to_idx)} unique cards")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.split == "train":
            # Step 1: Paste card on noisy background
            image = paste_card_on_background(image)
            # Step 2: Augmentation (geometric, color temp, exposure, shadow, noise)
            transform = build_train_transform()
            image = transform(image=image)["image"]
            # Step 3: Normalize + tensor
            image = final_transform(image=image)["image"]
        else:
            image = val_transform(image=image)["image"]

        label = self.label_to_idx[sample["card_id"]]
        return image, label


class PKSampler(Sampler):
    """Samples P random cards, then samples K images of each card per batch.
    Since each card has only 1 source image, K copies of the same index are
    returned — the dataset generates different augmentations each time."""

    def __init__(self, dataset, p=8, k=4):
        self.dataset = dataset
        self.p = p  # cards per batch
        self.k = k  # images per card per batch
        self.label_to_samples = dataset.label_to_samples
        self.all_labels = list(self.label_to_samples.keys())
        self.batch_size = p * k

    def __iter__(self):
        # Shuffle labels each epoch
        labels = self.all_labels.copy()
        np.random.shuffle(labels)

        # Walk through labels P at a time
        for i in range(0, len(labels) - self.p + 1, self.p):
            batch_labels = labels[i:i + self.p]
            batch_indices = []

            for label in batch_labels:
                indices = self.label_to_samples[label]
                # Since we likely have 1 image per card, repeat it K times
                # Each call to __getitem__ produces a different augmentation
                chosen = np.random.choice(indices, size=self.k, replace=True)
                batch_indices.extend(chosen.tolist())

            yield batch_indices

    def __len__(self):
        return len(self.all_labels) // self.p


# === Quick test ===
if __name__ == "__main__":
    csv_path = "data/splits/splits.csv"

    dataset = TripletCardDataset(csv_path, split="train")
    sampler = PKSampler(dataset, p=8, k=4)

    loader = DataLoader(dataset, batch_sampler=sampler)
    images, labels = next(iter(loader))

    print(f"\nBatch shape:  {images.shape}")    # expect [32, 3, 224, 224]
    print(f"Labels shape: {labels.shape}")       # expect [32]
    print(f"Labels:       {labels.tolist()}")
    print(f"Unique cards in batch: {len(set(labels.tolist()))}")  # expect 8
    print(f"Images per card: {labels.tolist().count(labels[0].item())}")  # expect 4