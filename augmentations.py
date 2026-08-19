import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# Background generation
# ============================================================

BG_COLOR = (128, 128, 128)


def paste_card_on_background(card_image):
    """Place the card in the center of a constant gray background."""
    card_h, card_w = card_image.shape[:2]
    bg = np.full((1020, 1020, 3), BG_COLOR, dtype=np.uint8)

    # Center the card on the background
    y_offset = (1020 - card_h) // 2
    x_offset = (1020 - card_w) // 2
    bg[y_offset:y_offset + card_h, x_offset:x_offset + card_w] = card_image

    return bg


# ============================================================
# Augmentation pipelines
# ============================================================

def build_train_transform():
    """Build a training pipeline with noisy background.
    Call this for EACH image to get a new random background.
    Does NOT include Normalize + ToTensorV2 — apply those separately."""

    # Random fill color for geometric transform borders
    # Use constant background color so borders blend with padding
    fill = BG_COLOR

    return A.Compose([
        # Step 1: Geometric — simulate hand-held angles
        A.Rotate(limit=15, border_mode=0, fill=fill, p=0.5),
        A.Perspective(scale=(0.02, 0.06), border_mode=0, fill=fill, p=0.5),
        A.Affine(shear=(-8, 8), border_mode=0, fill=fill, p=0.3),

        # Step 2: Color temperature — simulate warm/cool lighting
        # 3000K = warm incandescent, 5500K = daylight, 7000K = cool overcast
        A.PlanckianJitter(temperature_limit=(3000, 7000), p=0.5),

        # Step 3: Exposure — simulate bright/dim environments
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.5),

        # Step 4: Shadow — simulate hand/object casting shadow
        A.RandomShadow(
            num_shadows_limit=(1, 2),
            shadow_dimension=8,
            shadow_intensity_range=(0.2, 0.4),
            shadow_roi=(0, 0, 1, 1),
            p=0.3,
        ),

        # Step 5: Noise — simulate low-light phone camera sensor noise
        A.GaussNoise(std_range=(0.03, 0.15), p=0.4),

        # Step 6: Minor occlusion — simulate finger or sleeve edge
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(10, 30),
                        hole_width_range=(10, 30), fill=200, p=0.15),

        # Step 7: Resize to model input
        A.Resize(224, 224),
    ])


# Final step: applied AFTER augmentation to prepare for model
final_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Validation: no augmentation, just resize and normalize
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Reference: matches training layout (padded + centered) but no augmentation
ref_transform = A.Compose([
    A.PadIfNeeded(min_height=1020, min_width=1020, border_mode=0, fill=(128, 128, 128), p=1.0),
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])