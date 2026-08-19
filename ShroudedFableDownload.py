"""
Bulk Download Script - Shrouded Fable (sv06.5)
Downloads all card images and metadata for the set.
Easily extensible to other sets by adding to SET_IDS.
"""

from tcgdexsdk import TCGdex
from tcgdexsdk.enums import Quality, Extension
import json
import os
import time

sdk = TCGdex("en")

# Add more set IDs here when you're ready to scale
SET_IDS = ["sv03.5", "sv08"]

# Base directory for all data
BASE_DIR = "data"

def download_set(set_id):
    """Download all card images and metadata for a given set."""

    print(f"\n{'=' * 60}")
    print(f"Downloading set: {set_id}")
    print(f"{'=' * 60}")

    # Fetch set details
    set_data = sdk.set.getSync(set_id)
    print(f"Set Name: {set_data.name}")
    print(f"Total Cards: {set_data.cardCount.total if set_data.cardCount else '?'}")

    # Create directories
    img_dir = os.path.join(BASE_DIR, "images", set_id)
    meta_dir = os.path.join(BASE_DIR, "metadata")
    meta_card_dir = os.path.join(BASE_DIR, "metadata_by_card")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(meta_card_dir, exist_ok=True)

    if not set_data.cards:
        print("[ERROR] No cards found in set.")
        return

    # Track results
    success = 0
    failed = []
    all_metadata = []

    total = len(set_data.cards)

    for i, card_brief in enumerate(set_data.cards):
        card_id = f"{set_id}-{card_brief.localId}"
        print(f"  [{i+1}/{total}] {card_id}: {card_brief.name}...", end=" ")

        try:
            # Fetch full card details
            card = sdk.card.getSync(card_id)

            # --- Save image ---
            img_path = os.path.join(img_dir, f"{card_brief.localId}.png")
            if not os.path.exists(img_path):
                img_response = card.get_image(Quality.HIGH, Extension.PNG)
                img_data = img_response.read()
                with open(img_path, "wb") as f:
                    f.write(img_data)

            # --- Build metadata ---
            card_meta = {
                "id": card.id,
                "localId": card.localId,
                "name": card.name,
                "hp": card.hp,
                "category": card.category,
                "rarity": card.rarity,
                "types": card.types if card.types else [],
                "set": {
                    "id": set_id,
                    "name": set_data.name
                },
                "image_path": os.path.join("images", set_id, f"{card_brief.localId}.png"),
                "image_url_high": card.get_image_url(Quality.HIGH, Extension.PNG),
                "image_url_low": card.get_image_url(Quality.LOW, Extension.PNG),
            }

            # Optional fields - safely extract
            if hasattr(card, 'attacks') and card.attacks:
                card_meta["attacks"] = [
                    {
                        "name": atk.name,
                        "cost": atk.cost if atk.cost else [],
                        "damage": atk.damage,
                        "effect": atk.effect if hasattr(atk, 'effect') else None
                    }
                    for atk in card.attacks
                ]

            if hasattr(card, 'abilities') and card.abilities:
                card_meta["abilities"] = [
                    {
                        "name": ab.name,
                        "type": ab.type if hasattr(ab, 'type') else None,
                        "effect": ab.effect if hasattr(ab, 'effect') else None
                    }
                    for ab in card.abilities
                ]

            if hasattr(card, 'weaknesses') and card.weaknesses:
                card_meta["weaknesses"] = [
                    {"type": w.type, "value": w.value}
                    for w in card.weaknesses
                ]

            if hasattr(card, 'retreats') and card.retreats:
                card_meta["retreat_cost"] = card.retreats

            if hasattr(card, 'retreat') and card.retreat:
                card_meta["retreat_cost"] = card.retreat

            # Save individual card metadata
            card_meta_path = os.path.join(meta_card_dir, f"{card_id}.json")
            with open(card_meta_path, "w") as f:
                json.dump(card_meta, f, indent=2)

            all_metadata.append(card_meta)
            success += 1
            print("OK")

        except Exception as e:
            failed.append({"id": card_id, "error": str(e)})
            print(f"FAILED ({e})")

        # Small delay to be respectful to the API
        time.sleep(0.3)

    # Save complete set metadata
    set_meta_path = os.path.join(meta_dir, f"{set_id}.json")
    with open(set_meta_path, "w") as f:
        json.dump({
            "set_id": set_id,
            "set_name": set_data.name,
            "total_cards": set_data.cardCount.total if set_data.cardCount else len(all_metadata),
            "downloaded": success,
            "failed": len(failed),
            "cards": all_metadata
        }, f, indent=2)

    # Print summary
    print(f"\n{'─' * 40}")
    print(f"DONE: {set_data.name}")
    print(f"  Downloaded: {success}/{total}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Images:     {img_dir}/")
    print(f"  Metadata:   {set_meta_path}")

    if failed:
        print(f"\n  Failed cards:")
        for f_card in failed:
            print(f"    - {f_card['id']}: {f_card['error']}")

# ============================================================
# Run downloads
# ============================================================
if __name__ == "__main__":
    for set_id in SET_IDS:
        download_set(set_id)

    print(f"\n{'=' * 60}")
    print("All downloads complete!")
    print(f"Data saved to: {os.path.abspath(BASE_DIR)}/")
    print(f"{'=' * 60}")