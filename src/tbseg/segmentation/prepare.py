"""Prepara Montgomery + Shenzhen para entrenar las redes de segmentación.

- Redimensiona imágenes y máscaras a 512x512, igual que TBX11K (sin conservar proporción).
- Divide en train/val/test por paciente, estratificando por dataset de origen y diagnóstico.

Uso: python -m tbseg.segmentation.prepare
  -> data/processed/segtrain/{images,masks}/<image_id>.png
  -> data/metadata/segtrain_split.csv
"""

import re

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


def patient_ids(image_ids) -> dict:
    """Asigna a cada imagen el identificador de su paciente.

    Algunas lecturas de Montgomery dicen "same pt as MCUCXR_…": son controles del mismo
    paciente. Todas sus imágenes reciben como identificador la primera (orden alfabético);
    el resto de imágenes son su propio paciente.
    """
    pid = {}
    for image_id in image_ids:
        reading = (SEGTRAIN_RAW_DIR / "ClinicalReadings" / f"{image_id}.txt").read_text()
        same = re.findall(r"MCUCXR_\d{4}_\d", reading, re.I) if "same pt" in reading.lower() else []
        pid[image_id] = min([image_id, *(s.upper() for s in same)])
    # Segunda pasada para cadenas (A apunta a B y B apunta a C)
    return {image_id: pid.get(p, p) for image_id, p in pid.items()}


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
    index = pd.DataFrame(rows)
    index.insert(1, "patient_id", index["image_id"].map(patient_ids(index["image_id"])))
    return index


def assign_split(index: pd.DataFrame) -> pd.DataFrame:
    patients = index.groupby("patient_id")[["source", "tb"]].first()
    strata = patients["source"] + "_" + patients["tb"].astype(str)
    n_val_test = SEGTRAIN_SPLIT["val"] + SEGTRAIN_SPLIT["test"]

    train_ids, rest_ids = train_test_split(
        patients.index.tolist(),  # ← tolist() evita el error de pyarrow
        test_size=n_val_test, stratify=strata.tolist(), random_state=SEED
    )
    val_ids, test_ids = train_test_split(
        rest_ids,
        test_size=SEGTRAIN_SPLIT["test"] / n_val_test,
        stratify=strata[rest_ids].tolist(),  # ← igual aquí
        random_state=SEED,
    )
    split = pd.Series("train", index=patients.index)
    split[val_ids]  = "val"
    split[test_ids] = "test"
    return index.assign(split=index["patient_id"].map(split))


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
    print(f"{len(index)} pares imagen/máscara de {index['patient_id'].nunique()} pacientes en {SEGTRAIN_DIR}")
    print(pd.crosstab([index["source"], index["tb"]], index["split"], margins=True, margins_name="total"))


if __name__ == "__main__":
    main()
