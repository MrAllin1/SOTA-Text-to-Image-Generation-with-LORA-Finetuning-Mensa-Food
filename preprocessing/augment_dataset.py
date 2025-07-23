#!/usr/bin/env python3
"""
augment_mensafood.py
--------------------
Light data‑augmentation for Stable‑Diffusion fine‑tuning.

Examples
--------
# 4 variants per photo (default)
python augment_mensafood.py \
    --input_csv  meals_unique_mensafood.csv \
    --images_root /work/.../mensa_t2i/images \
    --output_dir  /work/.../mensa_t2i/aug \
    --output_csv  meals_augmented.csv

# 6 variants, using 8 CPU workers
python augment_mensafood.py -i meals_unique_mensafood.csv -o aug --n_augs 6 -j 8
"""
from __future__ import annotations

import argparse, logging, multiprocessing as mp, random
from pathlib import Path

import cv2
import albumentations as A
import pandas as pd
from tqdm import tqdm

# ─────────────────────────── logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────── augmentation pipeline ──────────────
def build_transforms() -> A.Compose:
    """Albumentations ≥ 2.0 pipeline tuned for food shots."""
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(512, 512),
                scale=(0.90, 1.00),
                ratio=(0.90, 1.10),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT101, p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(0.15, 0.15, p=0.7),
                    A.HueSaturationValue(10, 10, 10, p=0.3),
                ],
                p=0.5,
            ),
            # std_range expects values in 0‒1 (image intensities are normalised)
            A.GaussNoise(std_range=(5 / 255, 20 / 255), p=0.1),
            A.ToGray(p=0.05),
        ],
        p=1.0,
    )


TFM = build_transforms()

# ───────────────────────────── helpers ───────────────────────────
def _resolve_image_path(images_root: Path, rel_path: Path) -> Path:
    """
    Join *images_root* with *rel_path* while avoiding 'images/images/…' duplication.
    Returns the first existing path candidate; otherwise the joined path.
    """
    cand1 = images_root / rel_path
    if cand1.exists():
        return cand1

    if rel_path.parts and rel_path.parts[0] == images_root.name:
        cand2 = images_root.parent / rel_path
        if cand2.exists():
            return cand2

    return cand1  # may not exist; caller handles this


def _augment_row(
    row: pd.Series,
    images_root: Path,
    out_dir: Path,
    n_augs: int,
    idx_offset: int,
) -> list[dict]:
    """
    Augment one image *n_augs* times.
    Return new CSV rows as dicts. Missing files are skipped with a warning.
    """
    orig_path = _resolve_image_path(images_root, Path(row["image_path"]))
    orig_img = cv2.imread(str(orig_path))

    if orig_img is None:
        logging.warning(f"Could not read image: {orig_path} – skipping.")
        return []

    new_rows: list[dict] = []
    stem = Path(row["image_path"]).stem
    ext = Path(row["image_path"]).suffix or ".jpg"

    for i in range(n_augs):
        augmented = TFM(image=orig_img)["image"]
        out_name = f"{stem}_aug{idx_offset + i:04d}{ext}"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), augmented, [cv2.IMWRITE_JPEG_QUALITY, 95])
        new_rows.append(
            {
                "image_path": out_path.relative_to(images_root).as_posix(),
                "description": row["description"],
            }
        )
    return new_rows


def _worker(args):
    return _augment_row(*args)


# ─────────────────────────────────── main ─────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Data‑augmentation for Mensa food LoRA")
    p.add_argument("-i", "--input_csv", required=True,
                   help="CSV with columns: image_path, description")
    p.add_argument("--images_root", default=".",
                   help="Root directory of the image_path entries")
    p.add_argument("-o", "--output_dir", required=True,
                   help="Where augmented images will be written")
    p.add_argument("--output_csv", default="augmented.csv",
                   help="CSV (original + augmented rows)")
    p.add_argument("--n_augs", type=int, default=4,
                   help="Augmented variants per source image")
    p.add_argument("-j", "--workers", type=int,
                   default=max(1, mp.cpu_count() // 2),
                   help="CPU workers")
    p.add_argument("--seed", type=int, default=42,
                   help="Global RNG seed")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    images_root = Path(args.images_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    assert {"image_path", "description"}.issubset(df.columns), \
        "CSV must have columns: image_path, description"

    jobs = [
        (
            row,
            images_root,
            out_dir,
            args.n_augs,
            idx * args.n_augs,
        )
        for idx, row in df.iterrows()
    ]

    new_rows: list[dict] = []
    with mp.Pool(args.workers) as pool:
        for aug_rows in tqdm(
            pool.imap_unordered(_worker, jobs),
            total=len(jobs),
            desc="Augmenting",
        ):
            new_rows.extend(aug_rows)

    df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df_aug.to_csv(args.output_csv, index=False)

    logging.info(f"✓ Saved {len(new_rows)} augmented images → {out_dir}")
    logging.info(f"✓ New CSV with {len(df_aug)} rows → {args.output_csv}")


if __name__ == "__main__":
    main()
