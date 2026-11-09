"""Segmentación pulmonar con MedSAM y cajas automáticas.

MedSAM ya viene entrenado y segmenta lo que indica una caja. Lo único que se ajusta aquí es
cómo se obtienen las cajas, sin usar ninguna máscara de la imagen que se segmenta:

- Caja a priori: una por pulmón, igual para todas las imágenes. Es la mediana, coordenada
  por coordenada, de las cajas de las máscaras manuales de train, agrandada un margen.
- Caja refinada: MedSAM segmenta con la caja a priori y la caja del resultado (más un margen)
  se usa en una segunda pasada.

Los márgenes y la estrategia se eligen con el Dice medio de val. El conjunto de prueba se evalúa
una sola vez, solo con la opción elegida.

Uso:
    python -m tbseg.segmentation.medsam           # elige las cajas y evalúa en prueba
    python -m tbseg.segmentation.medsam --smoke   # prueba rápida, sin guardar nada
    python -m tbseg.segmentation.predict --arch medsam

Salidas (results/segmentation/):
    medsam_prompt.json     caja a priori, márgenes y estrategia elegidos
    medsam_selection.csv   Dice de val de cada opción probada
    test_per_image.csv     Dice e IoU de cada imagen de prueba
    test_metrics.csv       una fila por corrida
"""

import argparse
import json
import time
import urllib.request

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from tbseg.config import IMAGE_SIZE, MODELS_DIR, SEED, SEGTRAIN_CSV, SEGTRAIN_DIR
from tbseg.segmentation.train import SEG_RESULTS_DIR, THRESHOLD, update_csv

MEDSAM = "medsam"
MEDSAM_SIZE = 1024  # resolución de entrada fija de MedSAM
MEDSAM_CKPT = MODELS_DIR / "medsam" / "medsam_vit_b.pth"
MEDSAM_URL = "https://drive.usercontent.google.com/download?id=1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_&export=download&confirm=t"
MEDSAM_BYTES = 375_049_145

PROMPT_JSON = SEG_RESULTS_DIR / "medsam_prompt.json"
SELECTION_CSV = "medsam_selection.csv"

# Márgenes por lado, en píxeles de la imagen de 512
PRIOR_MARGINS = (0, 10, 20, 40, 60)
REFINE_MARGINS = (0, 5, 10, 20)


# ------------------------------------------------------------------------- cajas


def box_of(mask: np.ndarray) -> list[int]:
    """Caja [x0, y0, x1, y1] de los píxeles marcados."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("La máscara no tiene píxeles para calcular una caja")
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def lung_boxes(mask: np.ndarray) -> np.ndarray:
    """Una caja por pulmón (2, 4), ordenadas de izquierda a derecha en la imagen.

    Usa las dos regiones más grandes si quedan una a cada lado de la línea media.
    Si no, divide la máscara por la columna central.
    """
    binary = (mask > 0).astype(np.uint8)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    largest = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:2] + 1  # la etiqueta 0 es el fondo
    middle = binary.shape[1] / 2
    if len(largest) == 2 and (centroids[largest[0], 0] < middle) != (centroids[largest[1], 0] < middle):
        parts = [labels == i for i in largest]
    else:
        column = binary.shape[1] // 2
        parts = [binary.copy(), binary.copy()]
        parts[0][:, column:] = 0
        parts[1][:, :column] = 0
    return np.array(sorted(box_of(part) for part in parts))


def expand(boxes, margin: int, size: int = IMAGE_SIZE) -> np.ndarray:
    """Agranda cada caja `margin` píxeles por lado, sin salir de la imagen."""
    return np.clip(np.asarray(boxes) + np.array([-margin, -margin, margin, margin]), 0, size - 1)


def prior_boxes(masks) -> np.ndarray:
    """Caja a priori (2, 4): mediana, coordenada por coordenada, de las cajas de cada máscara."""
    return np.median(np.stack([lung_boxes(mask) for mask in masks]), axis=0).round().astype(int)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Región conectada más grande de una máscara (vacía si no hay ninguna)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n == 1:
        return np.zeros_like(mask, dtype=bool)
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


# --------------------------------------------------------------------- inferencia


def ensure_checkpoint():
    """Descarga los pesos oficiales de MedSAM si no existen y comprueba el tamaño."""
    if not MEDSAM_CKPT.exists():
        MEDSAM_CKPT.parent.mkdir(parents=True, exist_ok=True)
        partial = MEDSAM_CKPT.with_suffix(".part")
        print(f"Descargando los pesos de MedSAM en {MEDSAM_CKPT}")
        urllib.request.urlretrieve(MEDSAM_URL, partial)
        partial.rename(MEDSAM_CKPT)
    if MEDSAM_CKPT.stat().st_size != MEDSAM_BYTES:
        raise RuntimeError(f"{MEDSAM_CKPT} pesa {MEDSAM_CKPT.stat().st_size} bytes y se esperaban {MEDSAM_BYTES}")


def load_medsam(device: str) -> torch.nn.Module:
    """Arquitectura SAM ViT-B con los pesos de MedSAM, en modo inferencia."""
    # Se importa aquí para que U-Net y la radiómica no necesiten segment_anything ni torchvision
    from segment_anything import sam_model_registry

    ensure_checkpoint()
    model = sam_model_registry["vit_b"]()
    model.load_state_dict(torch.load(MEDSAM_CKPT, map_location="cpu"))
    return model.to(device).eval()


def preprocess(image: np.ndarray) -> torch.Tensor:
    """Gris (H, W) -> (1, 3, 1024, 1024) en [0, 1].

    Se recorta a los percentiles 0,5 y 99,5, como indica el artículo para rayos X. No se usa la
    normalización de SAM (media y desviación de ImageNet): MedSAM se entrenó con valores en [0, 1].

    Diferencia con utils/pre_grey_rgb.py de MedSAM: allí los percentiles se calculan solo con los
    píxeles mayores que 0 y aquí con todos. Medido en val: elige la misma opción y el Dice cambia 0,0008.
    """
    low, high = np.percentile(image, (0.5, 99.5))
    scaled = np.clip((image.astype(np.float32) - low) / max(high - low, 1e-8), 0, 1)
    resized = cv2.resize(scaled, (MEDSAM_SIZE, MEDSAM_SIZE), interpolation=cv2.INTER_CUBIC)
    # La interpolación cúbica puede salir de [0, 1]
    resized = (resized - resized.min()) / max(resized.max() - resized.min(), 1e-8)
    return torch.from_numpy(resized.astype(np.float32))[None, None].repeat(1, 3, 1, 1)


@torch.no_grad()
def embed(model, image: np.ndarray, device: str) -> torch.Tensor:
    """Salida del codificador de imagen (1, 256, 64, 64), en CPU. Es la parte costosa."""
    return model.image_encoder(preprocess(image).to(device)).cpu()


@torch.no_grad()
def segment(model, embedding: torch.Tensor, boxes, device: str, size: int = IMAGE_SIZE) -> np.ndarray:
    """Probabilidad de pulmón (K, size, size), una por caja. Reproduce medsam_inference del repositorio."""
    scaled = torch.as_tensor(np.asarray(boxes), dtype=torch.float, device=device) * (MEDSAM_SIZE / size)
    sparse, dense = model.prompt_encoder(points=None, boxes=scaled[:, None, :], masks=None)
    logits, _ = model.mask_decoder(
        image_embeddings=embedding.to(device),
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    probs = F.interpolate(torch.sigmoid(logits), size=(size, size), mode="bilinear", align_corners=False)
    return probs[:, 0].cpu().numpy()


def refine_boxes(model, embedding, boxes, margin: int, device: str) -> np.ndarray:
    """Caja de la región más grande de cada máscara, más un margen. Sin región, conserva la caja recibida."""
    refined = []
    for box, probs in zip(boxes, segment(model, embedding, boxes, device)):
        part = largest_component(probs > THRESHOLD)
        refined.append(expand(box_of(part), margin) if part.any() else box)
    return np.array(refined)


def prompt_boxes(model, embedding, prompt: dict, device: str) -> np.ndarray:
    """Cajas finales de una imagen según la estrategia del prompt."""
    boxes = expand(prompt["prior_boxes"], prompt["margin"])
    if prompt["strategy"] == "refined":
        boxes = refine_boxes(model, embedding, boxes, prompt["refine_margin"], device)
    return boxes


def segment_with_prompt(model, embedding, prompt: dict, device: str) -> np.ndarray:
    """Máscara booleana: unión de las máscaras de las dos cajas, con umbral y sin posprocesar."""
    boxes = prompt_boxes(model, embedding, prompt, device)
    return (segment(model, embedding, boxes, device) > THRESHOLD).any(axis=0)


def predict_masks(image_paths, device: str, prompt: dict | None = None):
    """Devuelve una máscara binaria uint8 (0/255) por imagen, en el mismo orden."""
    prompt = prompt or load_prompt()
    model = load_medsam(device)
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        yield segment_with_prompt(model, embed(model, image, device), prompt, device).astype(np.uint8) * 255


# ---------------------------------------------------------------------- selección


def overlap(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Dice e IoU de dos máscaras booleanas, con la misma definición de train.overlap_metrics."""
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    total = pred.sum() + target.sum()
    return float(2 * intersection / max(total, 1)), float(intersection / max(union, 1))


def select_prompt(model, prior: np.ndarray, cases: list[dict], device: str) -> tuple[dict, pd.DataFrame]:
    """Elige márgenes y estrategia con el Dice medio de `cases` (val).

    Primero el margen de la caja a priori; con ese margen, el de la segunda pasada. El refinamiento
    solo se elige si supera a la caja a priori. En un empate gana el margen menor.
    """

    def score(strategy: str, margin: int, refine_margin: int | None) -> dict:
        prompt = {"prior_boxes": prior.tolist(), "margin": margin, "strategy": strategy, "refine_margin": refine_margin}
        dices = [overlap(segment_with_prompt(model, c["embedding"], prompt, device), c["mask"])[0] for c in cases]
        return {**prompt, "val_dice": float(np.mean(dices))}

    prior_rows = [score("prior", margin, None) for margin in PRIOR_MARGINS]
    best_prior = max(prior_rows, key=lambda row: row["val_dice"])
    refine_rows = [score("refined", best_prior["margin"], margin) for margin in REFINE_MARGINS]
    best = max([best_prior, *refine_rows], key=lambda row: row["val_dice"])

    selection = pd.DataFrame(prior_rows + refine_rows).drop(columns="prior_boxes")
    selection["selected"] = [row is best for row in prior_rows + refine_rows]
    return best, selection


def save_prompt(prompt: dict):
    PROMPT_JSON.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_JSON.write_text(json.dumps(prompt, indent=2) + "\n")


def load_prompt() -> dict:
    if not PROMPT_JSON.exists():
        raise FileNotFoundError(f"No existe {PROMPT_JSON}. Falta: python -m tbseg.segmentation.medsam")
    return json.loads(PROMPT_JSON.read_text())


# ------------------------------------------------------------------------- corrida


def load_case(image_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Imagen en gris (uint8) y máscara manual (booleana) de 512x512."""
    image = cv2.imread(str(SEGTRAIN_DIR / "images" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(SEGTRAIN_DIR / "masks" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    return image, mask >= 128


def run(limit: int | None = None, save: bool = True) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = pd.read_csv(SEGTRAIN_CSV)
    ids = {name: split.loc[split["split"] == name, "image_id"].tolist()[:limit] for name in ("train", "val", "test")}
    start = time.time()

    prior = prior_boxes(load_case(image_id)[1] for image_id in tqdm(ids["train"], desc="Cajas de train"))
    print("Caja a priori (x0, y0, x1, y1):", prior.tolist())

    model = load_medsam(device)
    cases = []
    for image_id in tqdm(ids["val"], desc="Validación"):
        image, mask = load_case(image_id)
        cases.append({"mask": mask, "embedding": embed(model, image, device)})
    prompt, selection = select_prompt(model, prior, cases, device)
    print(selection.round(4).to_string(index=False))

    # El conjunto de prueba se evalúa una sola vez, con la opción elegida en validación
    rows = []
    for image_id in tqdm(ids["test"], desc="Prueba"):
        image, mask = load_case(image_id)
        dice, iou = overlap(segment_with_prompt(model, embed(model, image, device), prompt, device), mask)
        rows.append({"arch": MEDSAM, "seed": SEED, "image_id": image_id, "dice": dice, "iou": iou})
    test_scores = pd.DataFrame(rows)
    minutes = (time.time() - start) / 60
    print(f"Estrategia {prompt['strategy']} | margen {prompt['margin']} | margen de refinamiento "
          f"{prompt['refine_margin']} | Dice val {prompt['val_dice']:.3f} | test: Dice "
          f"{test_scores['dice'].mean():.3f} IoU {test_scores['iou'].mean():.3f} | {minutes:.1f} min")

    if save:
        save_run(prompt, selection, test_scores, minutes)
    return prompt, selection, test_scores


def save_run(prompt: dict, selection: pd.DataFrame, test_scores: pd.DataFrame, minutes: float):
    """Guarda el prompt y acumula las métricas en los mismos archivos que las redes."""
    SEG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_prompt(prompt)
    selection.to_csv(SEG_RESULTS_DIR / SELECTION_CSV, index=False)
    # MedSAM no se entrena: 0 épocas, y sin filas en history.csv
    summary = pd.DataFrame([{
        "arch": MEDSAM, "seed": SEED, "best_epoch": 0, "epochs_run": 0,
        "val_dice": round(prompt["val_dice"], 4),
        "test_dice": round(float(test_scores["dice"].mean()), 4),
        "test_dice_std": round(float(test_scores["dice"].std()), 4),
        "test_dice_min": round(float(test_scores["dice"].min()), 4),
        "test_iou": round(float(test_scores["iou"].mean()), 4),
        "minutes": round(minutes, 1),
    }])
    update_csv("test_per_image.csv", test_scores, MEDSAM, SEED)
    update_csv("test_metrics.csv", summary, MEDSAM, SEED)


def main():
    parser = argparse.ArgumentParser(description="Elige las cajas de MedSAM en val y evalúa en prueba")
    parser.add_argument("--smoke", action="store_true", help="prueba rápida: 4 imágenes por partición, sin guardar nada")
    args = parser.parse_args()
    if args.smoke:
        run(limit=4, save=False)
    else:
        run()


if __name__ == "__main__":
    main()
