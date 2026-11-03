"""Genera las máscaras pulmonares de una red entrenada sobre todas las imágenes de un dataset.

Las máscaras se guardan tal como salen de la red (umbral 0.5, sin posprocesar), porque
lo que se estudia es el efecto de cada técnica de segmentación tal cual.

Uso:
    python -m tbseg.segmentation.predict --arch unet
    python -m tbseg.segmentation.predict --arch attention_unet --seed 43

Salidas:
    masks/<dataset>/<método>/<image_id>.png   binaria (0/255), 512x512
    results/segmentation/mask_qc_<dataset>.csv  control de calidad por imagen y método

El método es el nombre de la arquitectura; con una semilla distinta a la por defecto
se agrega el sufijo _seed<n> (p. ej. unet_seed43).
"""

import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tbseg.config import MASKS_DIR, MODELS_DIR, SEED, SEGTRAIN_CSV, SEGTRAIN_DIR
from tbseg.segmentation.train import ARCHITECTURES, BATCH_SIZE, THRESHOLD, build_model, update_csv

# Por ahora solo Montgomery-Shenzhen; TBX11K se agrega aquí cuando se procese.
# images: carpeta de radiografías 512x512 | manual: máscaras de referencia (None si no hay)
DATASETS = {
    "montgomery_shenzhen": {
        "images": SEGTRAIN_DIR / "images",
        "manual": SEGTRAIN_DIR / "masks",
        "split": SEGTRAIN_CSV,
    },
}


def method_name(arch: str, seed: int) -> str:
    return arch if seed == SEED else f"{arch}_seed{seed}"


def load_model(arch: str, seed: int, device: str) -> torch.nn.Module:
    checkpoint = torch.load(MODELS_DIR / f"{arch}_seed{seed}.pt", map_location=device)
    model = build_model(arch)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def predict_masks(model, image_paths, device: str):
    """Devuelve una máscara binaria uint8 (0/255) por imagen, en el mismo orden."""
    for start in range(0, len(image_paths), BATCH_SIZE):
        batch = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in image_paths[start:start + BATCH_SIZE]]
        x = torch.from_numpy(np.stack(batch).astype(np.float32) / 255.0)[:, None].to(device)
        probs = torch.sigmoid(model(x))[:, 0].cpu().numpy()
        yield from ((probs > THRESHOLD) * 255).astype(np.uint8)


def mask_qc(mask: np.ndarray, manual: np.ndarray | None = None) -> dict:
    """Indicadores para detectar máscaras rotas.

    - area_frac: fracción de la imagen cubierta por la máscara.
    - n_components: regiones conectadas (lo esperado es 2: un pulmón a cada lado).
    - extra_area_frac: fracción de la máscara fuera de las dos regiones más grandes (manchas sueltas).
    - two_sides: las dos regiones más grandes quedan una a cada lado de la línea media.
    - dice: superposición con la máscara manual, si existe.
    """
    binary = (mask > 0).astype(np.uint8)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas, cx = stats[1:, cv2.CC_STAT_AREA], centroids[1:, 0]
    total = int(areas.sum())
    largest = np.argsort(areas)[::-1][:2]
    middle = mask.shape[1] / 2
    qc = {
        "area_frac": total / binary.size,
        "n_components": n - 1,
        "extra_area_frac": (total - areas[largest].sum()) / total if total else 0.0,
        "two_sides": bool(len(largest) == 2 and (cx[largest[0]] < middle) != (cx[largest[1]] < middle)),
    }
    if manual is not None:
        ref = manual > 0
        denom = binary.sum() + ref.sum()
        qc["dice"] = 2 * np.logical_and(binary, ref).sum() / denom if denom else 1.0
    return qc


def main():
    parser = argparse.ArgumentParser(description="Genera las máscaras pulmonares de una red entrenada")
    parser.add_argument("--arch", choices=list(ARCHITECTURES), required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dataset", choices=list(DATASETS), default="montgomery_shenzhen")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = DATASETS[args.dataset]
    method = method_name(args.arch, args.seed)
    out_dir = MASKS_DIR / args.dataset / method
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(dataset["images"].glob("*.png"))
    model = load_model(args.arch, args.seed, device)
    rows = []
    masks = predict_masks(model, image_paths, device)
    for path, mask in tqdm(zip(image_paths, masks), total=len(image_paths), desc=method):
        cv2.imwrite(str(out_dir / path.name), mask)
        manual = None
        if dataset["manual"] is not None:
            manual = cv2.imread(str(dataset["manual"] / path.name), cv2.IMREAD_GRAYSCALE)
        rows.append({"image_id": path.stem, **mask_qc(mask, manual)})

    qc = pd.DataFrame(rows)
    if dataset["split"] is not None:
        qc = qc.merge(pd.read_csv(dataset["split"])[["image_id", "split"]], on="image_id", how="left")
    qc.insert(0, "arch", args.arch)
    qc.insert(1, "seed", args.seed)
    update_csv(f"mask_qc_{args.dataset}.csv", qc, args.arch, args.seed)
    print(f"{len(qc)} máscaras en {out_dir}")
    if "dice" in qc:
        print(qc.groupby("split")["dice"].describe().round(4))


if __name__ == "__main__":
    main()
