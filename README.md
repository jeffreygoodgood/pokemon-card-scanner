# Pokémon TCG Card Scanner

Two-stage pipeline for identifying Pokémon TCG cards from smartphone photos. YOLO11 Nano OBB detects and crops the card, EfficientNet-B0 with hard triplet loss identifies it via nearest-neighbor embedding search. **~95.5% Top-1 accuracy** across 558 cards from three Scarlet & Violet sets.

Built for EECE 7370 — Advanced Computer Vision at Northeastern University.

## Pipeline

```
Photo → YOLO11 OBB Detection → Perspective Correction → Gray Background Composite
     → EfficientNet-B0 Embedding (128-d) → torch.cdist vs Reference DB → Card ID
```

**Stage 1 — Detection:** YOLO11 Nano OBB trained on 4,500 synthetic images (card composited on random DTD textures). Outputs oriented bounding box, corrected via perspective warp.

**Stage 2 — Recognition:** EfficientNet-B0 backbone (partially frozen), 128-d L2-normalized embedding head, trained with PK-sampled hard triplet mining (P=8, K=4, margin=0.3). Identity resolved by Euclidean nearest-neighbor search against a precomputed reference database.

## Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Top-1 Accuracy | ≥ 85% | ~95.5% (21/22) |
| Top-5 Accuracy | ≥ 95% | ~100% |
| Card Coverage | ≥ 3 sets | 3 sets (558 cards) |
| Inference Time | < 500ms | ~200ms |

## Dataset

Three Scarlet & Violet sets downloaded via [TCGDex API](https://tcgdex.dev):

- **sv03.5** — Pokémon 151 (207 cards)
- **sv06.5** — Shrouded Fable (99 cards)
- **sv08** — Surging Sparks (252 cards)

Card images are not included in this repo. Run `ShroudedFableDownload.py` to re-download.

## Project Structure

```
├── model.py                  # EfficientNet-B0 embedding model
├── train.py                  # Training loop with hard triplet mining
├── train_yolo.py             # YOLO11 OBB detector training
├── augmentations.py          # Albumentations v2 augmentation pipeline
├── triplet_dataset.py        # PK-sampled dataset + batch sampler
├── pokemon_dataset.py        # Basic dataset class
├── Inference.py              # Full pipeline: detect → embed → match
├── cvs_generator.py          # Train/val split CSV generator
├── ShroudedFableDownload.py  # TCGDex bulk image downloader
├── test_yolo_detection.py    # YOLO detection visualization
├── visual_augmentation.py    # Augmentation preview utility
├── App/
│   ├── server.py             # FastAPI inference server
│   └── build_reference_db.py # Reference embedding database builder
└── hf_space/                 # Hugging Face Spaces deployment config
```

## Quick Start

**Requirements:** Python 3.10, NVIDIA GPU with driver ≥ 525

```bash
# Install PyTorch first (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install ultralytics albumentations>=2.0 opencv-python numpy<2 \
    tcgdex-sdk fastapi uvicorn python-multipart httpx pandas tqdm Pillow huggingface_hub

# Download card images
python ShroudedFableDownload.py

# Generate train/val splits
python cvs_generator.py

# Train recognition model (150 epochs)
python train.py

# Run inference
python Inference.py
```

## Key Lessons

- **Detection is a prerequisite, not an enhancement.** Recognition without detection was non-functional — background dominated the embeddings.
- **Background strategy has outsized impact.** Constant gray background forced the model to learn card content; random noise backgrounds caused unstable training.
- **Preprocessing consistency is non-negotiable.** A padding mismatch between training and reference DB construction dropped real-world accuracy to ~50% despite strong validation metrics.

## Known Limitations

- Holographic/foil cards produce specular glare that the augmentation pipeline cannot simulate, causing the single misidentification in the test set.
- Evaluation was conducted on 22 test photographs — representative but modest in scale.

## License

Academic project — no license specified.
