"""Extrae las características radiómicas de primer orden dentro de la máscara pulmonar de un método.

La radiografía es la misma para todos los métodos y los parámetros salen del mismo archivo
(firstorder.yaml): lo único que cambia entre métodos es la máscara.

Uso:
    python -m tbseg.radiomics.extract --method unet
    python -m tbseg.radiomics.extract --method attention_unet
    python -m tbseg.radiomics.extract --method medsam
    python -m tbseg.radiomics.extract --method deeplabv3plus

Salida:
    results/radiomics/<dataset>_firstorder_<método>.csv
    una fila por imagen (ordenadas por image_id, iguales en todos los métodos):
    image_id, patient_id, source, tb, split y las 18 características de primer orden
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm

from tbseg.config import MASKS_DIR, RESULTS_DIR
from tbseg.segmentation.predict import DATASETS

PARAMS = Path(__file__).with_name("firstorder.yaml")
PREFIX = "original_firstorder_"
RADIOMICS_DIR = RESULTS_DIR / "radiomics"

logging.getLogger("radiomics").setLevel(logging.ERROR)


def build_extractor(params: Path | dict = PARAMS) -> featureextractor.RadiomicsFeatureExtractor:
    """Extractor configurado con el YAML del proyecto (o con un diccionario, para pruebas)."""
    # El validador de parámetros de pyradiomics (pykwalify) solo acepta la ruta como str
    return featureextractor.RadiomicsFeatureExtractor(params if isinstance(params, dict) else str(params))


def extract_features(extractor, image: np.ndarray, mask: np.ndarray) -> dict:
    """Características de primer orden de los píxeles de la imagen dentro de la máscara.

    La imagen se pasa a float32: con uint8, pyradiomics desborda al discretizar
    (maximum + 2*binWidth > 255) y devuelve Entropy≈0 y Uniformity=1 sin dar error.
    La máscara se pasa a 0/1 porque se guarda como 0/255 y pyradiomics busca la etiqueta 1.
    """
    sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
    sitk_mask = sitk.GetImageFromArray((mask > 0).astype(np.uint8))
    result = extractor.execute(sitk_image, sitk_mask)
    return {k.removeprefix(PREFIX): float(v) for k, v in result.items() if k.startswith(PREFIX)}


def main():
    parser = argparse.ArgumentParser(description="Extrae características de primer orden por método")
    parser.add_argument("--method", required=True, help="carpeta en masks/<dataset>/, p. ej. unet")
    parser.add_argument("--dataset", choices=list(DATASETS), default="montgomery_shenzhen")
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    mask_dir = MASKS_DIR / args.dataset / args.method
    labels = pd.read_csv(dataset["split"])[["image_id", "patient_id", "source", "tb", "split"]]
    labels = labels.sort_values("image_id", ignore_index=True)
    missing = [i for i in labels["image_id"] if not (mask_dir / f"{i}.png").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} imágenes sin máscara en {mask_dir}, p. ej. {missing[:3]}")

    extractor = build_extractor()
    rows = []
    for image_id in tqdm(labels["image_id"], desc=args.method):
        image = cv2.imread(str(dataset["images"] / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_dir / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        rows.append(extract_features(extractor, image, mask))

    features = pd.concat([labels, pd.DataFrame(rows)], axis=1)
    RADIOMICS_DIR.mkdir(parents=True, exist_ok=True)
    out = RADIOMICS_DIR / f"{args.dataset}_firstorder_{args.method}.csv"
    features.to_csv(out, index=False)
    print(f"{len(features)} imágenes x {features.shape[1] - labels.shape[1]} características en {out}")


if __name__ == "__main__":
    main()
