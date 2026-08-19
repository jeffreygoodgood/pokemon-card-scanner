import os
import cv2
import numpy as np
from augmentations import build_train_transform, paste_card_on_background

# === Config ===
image_path = "data/images/sv06.5/041.png"
output_dir = "testcard"
num_augmented = 3

os.makedirs(output_dir, exist_ok=True)

# === Load original ===
original = cv2.imread(image_path)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# === Generate and save augmented images ===
for i in range(num_augmented):
    # Paste card on noisy background, then augment
    image = paste_card_on_background(original)
    transform = build_train_transform()
    aug = transform(image=image)["image"]

    # Save as BGR for OpenCV
    save_path = os.path.join(output_dir, f"aug_{i+1}.png")
    cv2.imwrite(save_path, cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
    print(f"Saved {save_path}")

print(f"\nDone. Update QUERY_PATH in inference.py to test each one.")