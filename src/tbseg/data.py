"""Índice de TBX11K, subconjunto balanceado con partición y carga de imágenes.

Uso: python -m tbseg.data
  -> data/metadata/subset.csv                       subconjunto con su partición
  -> data/processed/tbx11k/images/<image_id>.png    imágenes del subconjunto en gris (512x512)
"""

import hashlib

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tbseg.config import (
    POSITIVE_CLASS,
    SEED,
    SUBSET_CSV,
    SUBSET_SIZES,
    SUBSET_TEST_SIZE,
    TBX11K_IMG_DIR,
    TBX11K_PROCESSED_DIR,
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


def build_index(classes=tuple(SUBSET_SIZES)) -> pd.DataFrame:
    """Lista las imágenes de las clases indicadas de TBX11K con su clase y un hash del contenido."""
    rows = [
        {"image_id": p.stem, "rel_path": f"{cls}/{p.name}", "source_class": cls}
        for cls in classes
        for p in sorted((TBX11K_IMG_DIR / cls).glob("*.png"))
    ]
    index = pd.DataFrame(rows)
    # El hash se calcula sobre los píxeles, no sobre el archivo, para detectar
    # imágenes idénticas aunque el PNG esté codificado distinto.
    index["md5"] = [hashlib.md5(load_gray(p).tobytes()).hexdigest() for p in index["rel_path"]]
    index["label"] = (index["source_class"] == POSITIVE_CLASS).astype(int)
    return index


def build_subset(index: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados, muestrea por clase y asigna la partición desarrollo/prueba."""
    # Duplicados: se conserva la primera aparición (orden alfabético de image_id)
    unique = index.sort_values("image_id").drop_duplicates("md5", keep="first")

    subset = pd.concat(
        [
            unique[unique["source_class"] == cls].sample(n=n, random_state=SEED)
            for cls, n in SUBSET_SIZES.items()
        ]
    ).sort_values("image_id", ignore_index=True)

    # Estratificar por source_class equivale a estratificar por label (TB vs sana).
    # TBX11K no publica identificadores de paciente, así que la partición es por imagen.
    _, test_idx = train_test_split(
        subset.index, test_size=SUBSET_TEST_SIZE, stratify=subset["source_class"], random_state=SEED
    )
    subset["split"] = "train"
    subset.loc[test_idx, "split"] = "test"

    return subset[["image_id", "rel_path", "source_class", "label", "split"]]


def load_subset() -> pd.DataFrame:
    return pd.read_csv(SUBSET_CSV)


def load_labels() -> pd.DataFrame:
    """Subconjunto con las columnas comunes a todos los datasets: image_id, patient_id, source, tb, split.

    TBX11K no publica identificadores de paciente: cada imagen cuenta como su propio paciente.
    """
    subset = load_subset()
    return pd.DataFrame({
        "image_id": subset["image_id"],
        "patient_id": subset["image_id"],
        "source": "tbx11k",
        "tb": subset["label"],
        "split": subset["split"],
    })


def export_images(subset: pd.DataFrame):
    """Guarda cada imagen del subconjunto en gris con nombre <image_id>.png, como las de Montgomery-Shenzhen."""
    out = TBX11K_PROCESSED_DIR / "images"
    out.mkdir(parents=True, exist_ok=True)
    for image_id, rel_path in zip(subset["image_id"], subset["rel_path"]):
        cv2.imwrite(str(out / f"{image_id}.png"), load_gray(rel_path))


def main():
    index = build_index()
    n_dups = index.duplicated("md5").sum()
    print(f"Imágenes etiquetadas: {len(index)} | copias duplicadas descartadas: {n_dups}")

    subset = build_subset(index)
    SUBSET_CSV.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(SUBSET_CSV, index=False)

    print(f"Subconjunto guardado en {SUBSET_CSV.relative_to(SUBSET_CSV.parents[2])} ({len(subset)} imágenes)")
    print(pd.crosstab(subset["split"], subset["source_class"], margins=True, margins_name="total"))

    export_images(subset)
    print(f"Imágenes en gris en {TBX11K_PROCESSED_DIR.relative_to(SUBSET_CSV.parents[2])}/images")


if __name__ == "__main__":
    main()
