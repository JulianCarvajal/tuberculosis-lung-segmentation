"""Prepara Montgomery + Shenzhen para entrenar las redes de segmentación.

- Redimensiona imágenes y máscaras a 512x512, igual que TBX11K (sin conservar proporción).
- Divide en train/val/test estratificando por dataset de origen y diagnóstico.

Uso: python -m tbseg.segmentation.prepare
  -> data/processed/segtrain/{images,masks}/<image_id>.png
  -> data/metadata/segtrain_split.csv
"""

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tbseg.config import (
    IMAGE_SIZE,
    SEED,
    SEGTRAIN_CSV,
    SEGTRAIN_DIR,
    SEGTRAIN_RAW_DIR,
    SEGTRAIN_SPLIT,
)

SOURCES = {"CHNCXR": "shenzhen", "MCUCXR": "montgomery"}


def mask_path(image_id: str):
    # Shenzhen: <id>_mask.png | Montgomery: <id>.png
    suffix = "_mask" if image_id.startswith("CHNCXR") else ""
    return SEGTRAIN_RAW_DIR / "masks" / f"{image_id}{suffix}.png"


def build_index() -> pd.DataFrame:
    """Lista las imágenes con máscara. El nombre termina en _0 (normal) o _1 (TB)."""
    rows = [
        {
            "image_id": p.stem,
            "source": SOURCES[p.stem.split("_")[0]],
            "tb": int(p.stem.endswith("_1")),
        }
        for p in sorted((SEGTRAIN_RAW_DIR / "CXR_png").glob("*.png"))
        if mask_path(p.stem).exists()
    ]
    return pd.DataFrame(rows)


def assign_split(index: pd.DataFrame) -> pd.DataFrame:
    """Divide en train/val/test manteniendo la proporción de origen y diagnóstico."""
    strata = index["source"] + "_" + index["tb"].astype(str)
    n_val_test = SEGTRAIN_SPLIT["val"] + SEGTRAIN_SPLIT["test"]
    train_idx, rest_idx = train_test_split(
        index.index, test_size=n_val_test, stratify=strata, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=SEGTRAIN_SPLIT["test"] / n_val_test,
        stratify=strata[rest_idx],
        random_state=SEED,
    )
    index = index.copy()
    index.loc[train_idx, "split"] = "train"
    index.loc[val_idx, "split"] = "val"
    index.loc[test_idx, "split"] = "test"
    return index


def preprocess(image_id: str):
    """Imagen en grises y máscara binaria (0/255), ambas de IMAGE_SIZE x IMAGE_SIZE."""
    size = (IMAGE_SIZE, IMAGE_SIZE)
    img = cv2.imread(str(SEGTRAIN_RAW_DIR / "CXR_png" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path(image_id)), cv2.IMREAD_GRAYSCALE)
    if img.shape != mask.shape:
        raise ValueError(f"{image_id}: imagen {img.shape} y máscara {mask.shape} no coinciden")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    # INTER_AREA suaviza el borde; el umbral devuelve una máscara binaria
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    mask = ((mask >= 128) * 255).astype("uint8")
    return img, mask


def main():
    index = assign_split(build_index())

    for sub in ("images", "masks"):
        (SEGTRAIN_DIR / sub).mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(index["image_id"], desc="Redimensionando"):
        img, mask = preprocess(image_id)
        cv2.imwrite(str(SEGTRAIN_DIR / "images" / f"{image_id}.png"), img)
        cv2.imwrite(str(SEGTRAIN_DIR / "masks" / f"{image_id}.png"), mask)

    SEGTRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(SEGTRAIN_CSV, index=False)
    print(f"{len(index)} pares imagen/máscara en {SEGTRAIN_DIR}")
    print(pd.crosstab([index["source"], index["tb"]], index["split"], margins=True, margins_name="total"))


if __name__ == "__main__":
    main()
