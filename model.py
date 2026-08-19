import torch
import torch.nn as nn
from torchvision import models


class CardEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        # Load EfficientNet-B0 with pre-trained ImageNet weights
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Remove the classification head, keep everything else
        # backbone.features = all the convolutional layers
        # backbone.avgpool = global average pooling (1280 features out)
        self.backbone = backbone.features
        self.pool = backbone.avgpool

        # Our embedding head: 1280 backbone features → 128 fingerprint
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, embedding_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.embedding(x)
        # Normalize to unit length — makes cosine distance == euclidean distance
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


# === Quick test ===
if __name__ == "__main__":
    model = CardEmbeddingModel(embedding_dim=128)

    # Dummy input: batch of 4 images, 3 channels, 224x224
    dummy = torch.randn(4, 3, 224, 224)
    output = model(dummy)

    print(f"Input shape:  {dummy.shape}")       # [4, 3, 224, 224]
    print(f"Output shape: {output.shape}")       # [4, 128]
    print(f"Embedding norm: {output.norm(dim=1)}")  # should be ~1.0 for each
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable:        {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")