"""Índice de TBX11K, subconjunto balanceado con folds y carga de imágenes.

Uso: python -m tbseg.data   -> genera data/metadata/subset.csv
"""

import hashlib

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from tbseg.config import (
    N_FOLDS,
    POSITIVE_CLASS,
    SEED,
    SUBSET_CSV,
    SUBSET_SIZES,
    TBX11K_IMG_DIR,
)


def load_gray(rel_path: str) -> np.ndarray:
    """Carga una imagen de TBX11K en escala de grises (uint8, 512x512).

    Las imágenes vienen en RGB con los tres canales idénticos, así que leerlas
    en gris no pierde información. Todos los métodos deben usar esta función.
    """
    img = cv2.imread(str(TBX11K_IMG_DIR / rel_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(TBX11K_IMG_DIR / rel_path)
    return img


def build_index() -> pd.DataFrame:
    """Lista las imágenes etiquetadas de TBX11K con su clase y un hash del contenido."""
    rows = [
        {"image_id": p.stem, "rel_path": f"{cls}/{p.name}", "source_class": cls}
        for cls in SUBSET_SIZES
        for p in sorted((TBX11K_IMG_DIR / cls).glob("*.png"))
    ]
    index = pd.DataFrame(rows)
    # El hash se calcula sobre los píxeles, no sobre el archivo, para detectar
    # imágenes idénticas aunque el PNG esté codificado distinto.
    index["md5"] = [hashlib.md5(load_gray(p).tobytes()).hexdigest() for p in index["rel_path"]]
    index["label"] = (index["source_class"] == POSITIVE_CLASS).astype(int)
    return index


def build_subset(index: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados, muestrea por clase y asigna folds estratificados."""
    # Duplicados: se conserva la primera aparición (orden alfabético de image_id)
    unique = index.sort_values("image_id").drop_duplicates("md5", keep="first")

    subset = pd.concat(
        [
            unique[unique["source_class"] == cls].sample(n=n, random_state=SEED)
            for cls, n in SUBSET_SIZES.items()
        ]
    ).sort_values("image_id", ignore_index=True)

    # Estratificar por source_class (tb/health/sick) también balancea label (TB vs no-TB)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    subset["fold"] = -1
    for fold, (_, test_idx) in enumerate(skf.split(subset, subset["source_class"])):
        subset.loc[test_idx, "fold"] = fold

    return subset[["image_id", "rel_path", "source_class", "label", "fold"]]


def load_subset() -> pd.DataFrame:
    return pd.read_csv(SUBSET_CSV)


def main():
    index = build_index()
    n_dups = index.duplicated("md5").sum()
    print(f"Imágenes etiquetadas: {len(index)} | copias duplicadas descartadas: {n_dups}")

    subset = build_subset(index)
    SUBSET_CSV.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(SUBSET_CSV, index=False)

    print(f"Subconjunto guardado en {SUBSET_CSV.relative_to(SUBSET_CSV.parents[2])} ({len(subset)} imágenes)")
    print(pd.crosstab(subset["fold"], subset["source_class"], margins=True, margins_name="total"))


if __name__ == "__main__":
    main()
