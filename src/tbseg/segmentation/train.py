"""Entrenamiento de las redes de segmentación pulmonar.

Una sola rutina para todas las arquitecturas: los datos, los aumentos, la función
de pérdida, el optimizador y el número de épocas son idénticos. Lo único que
cambia entre corridas es la arquitectura (--arch) y la semilla (--seed), de modo
que las diferencias observadas se atribuyan a la arquitectura y no al protocolo.

Uso:
    python -m tbseg.segmentation.train --arch unet --seed 42
    python -m tbseg.segmentation.train --arch unet --smoke   # prueba rápida

Salidas (results/segmentation/ acumula todas las corridas en tres archivos):
    models/<arch>_seed<seed>.pt   pesos de la mejor época
    history.csv                   métricas por época
    test_per_image.csv            Dice e IoU de cada imagen de prueba
    test_metrics.csv              una fila por corrida
"""

import argparse
import copy
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet, UNet
from torch.utils.data import DataLoader, Dataset

from tbseg.config import (
    IMAGE_SIZE,
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    SEGTRAIN_CSV,
    SEGTRAIN_DIR,
)

# Configuración común a todas las arquitecturas
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
MAX_EPOCHS = 60
PATIENCE = 10  # épocas sin mejorar el Dice de validación antes de detenerse
CHANNELS = (16, 32, 64, 128, 256)
STRIDES = (2, 2, 2, 2)
THRESHOLD = 0.5

SEG_RESULTS_DIR = RESULTS_DIR / "segmentation"
ARCHITECTURES = {"unet": UNet, "attention_unet": AttentionUnet}


def build_model(arch: str) -> nn.Module:
    """Crea la red. Ambas arquitecturas reciben exactamente los mismos argumentos."""
    return ARCHITECTURES[arch](
        spatial_dims=2,      # imágenes 2D
        in_channels=1,       # escala de grises
        out_channels=1,      # una clase: pulmón sí o no
        channels=CHANNELS,   # filtros por nivel
        strides=STRIDES,     # cada nivel reduce la resolución a la mitad
    )


# --------------------------------------------------------------------------- datos


def load_pair(image_id: str):
    """Imagen (float32 en [0, 1]) y máscara (float32 en {0, 1}) de 512x512."""
    img = cv2.imread(str(SEGTRAIN_DIR / "images" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(SEGTRAIN_DIR / "masks" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    return (img / 255.0).astype(np.float32), (mask >= 128).astype(np.float32)


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Rotación, escala y traslación (imagen y máscara) + brillo y contraste (solo imagen).

    No se usa volteo horizontal: produciría radiografías con el corazón a la derecha,
    que no existen en los datos reales.
    """
    size = img.shape[0]
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(-10, 10), rng.uniform(0.9, 1.1))
    matrix[:, 2] += rng.uniform(-0.05, 0.05, size=2) * size
    img = cv2.warpAffine(img, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(mask, matrix, (size, size), flags=cv2.INTER_NEAREST, borderValue=0)
    img = np.clip(img * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1), 0, 1)
    return img, mask


class LungDataset(Dataset):
    """Pares imagen/máscara en memoria. Los aumentos se aplican solo en entrenamiento."""

    def __init__(self, image_ids, train: bool = False, seed: int = SEED):
        pairs = [load_pair(i) for i in image_ids]
        self.images = [p[0] for p in pairs]
        self.masks = [p[1] for p in pairs]
        self.image_ids = list(image_ids)
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img, mask = self.images[i], self.masks[i]
        if self.train:
            img, mask = augment(img, mask, self.rng)
        return torch.from_numpy(img)[None], torch.from_numpy(mask)[None]


def build_loaders(seed: int, limit: int | None = None) -> dict[str, DataLoader]:
    """Un DataLoader por partición. Solo el de entrenamiento baraja y aumenta."""
    split = pd.read_csv(SEGTRAIN_CSV)
    loaders = {}
    for name in ("train", "val", "test"):
        ids = split.loc[split["split"] == name, "image_id"]
        if limit:
            ids = ids.head(limit)
        dataset = LungDataset(ids, train=(name == "train"), seed=seed)
        loaders[name] = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(name == "train"))
    return loaders


# ------------------------------------------------------------- pérdida y métricas


def dice_loss(logits, target, eps: float = 1.0):
    """1 - Dice suave. Complementa a BCE: optimiza la superposición, no el acierto por píxel."""
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    total = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * intersection + eps) / (total + eps)).mean()


bce_loss = nn.BCEWithLogitsLoss()


def loss_fn(logits, target):
    return bce_loss(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def overlap_metrics(logits, target, threshold: float = THRESHOLD):
    """Dice e IoU por imagen, tras aplicar el umbral."""
    pred = (torch.sigmoid(logits) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3))
    total = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (2 * intersection / total.clamp(min=1)).cpu(), (intersection / union.clamp(min=1)).cpu()


# ----------------------------------------------------------------- entrenamiento


def run_epoch(model, loader, device, optimizer=None):
    """Recorre una partición completa; entrena si recibe optimizador.

    Devuelve la pérdida media y el Dice e IoU de cada imagen.
    """
    training = optimizer is not None
    model.train(training)
    losses, dices, ious = [], [], []
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = loss_fn(logits, masks)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            dice, iou = overlap_metrics(logits, masks)
            losses.append(loss.item() * len(images))
            dices.append(dice)
            ious.append(iou)
    return sum(losses) / len(loader.dataset), torch.cat(dices), torch.cat(ious)


def train(arch: str, seed: int, epochs: int = MAX_EPOCHS, limit: int | None = None, save: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    loaders = build_loaders(seed, limit)
    model = build_model(arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"{arch} (semilla {seed}) en {device} | "
          f"{sum(p.numel() for p in model.parameters()):,} parámetros | "
          f"{len(loaders['train'].dataset)} imágenes de entrenamiento")

    history, best = [], {"dice": 0.0, "epoch": 0, "state": None}
    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_dice, _ = run_epoch(model, loaders["train"], device, optimizer)
        val_loss, val_dice, _ = run_epoch(model, loaders["val"], device)
        train_dice, val_dice = train_dice.mean().item(), val_dice.mean().item()
        history.append({"arch": arch, "seed": seed, "epoch": epoch,
                        "train_loss": train_loss, "val_loss": val_loss,
                        "train_dice": train_dice, "val_dice": val_dice,
                        "minutes": (time.time() - start) / 60})
        if val_dice > best["dice"]:
            best = {"dice": val_dice, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        print(f"  época {epoch:2d} | pérdida train {train_loss:.3f} val {val_loss:.3f} | "
              f"Dice train {train_dice:.3f} val {val_dice:.3f}", flush=True)
        # Detención temprana: si el Dice de validación no mejora, no vale la pena seguir
        if epoch - best["epoch"] >= PATIENCE:
            print(f"  detención temprana: {PATIENCE} épocas sin mejorar")
            break

    # El conjunto de prueba se evalúa una sola vez, con los pesos de la mejor época
    model.load_state_dict(best["state"])
    _, dice, iou = run_epoch(model, loaders["test"], device)
    test_scores = pd.DataFrame({"arch": arch, "seed": seed,
                                "image_id": loaders["test"].dataset.image_ids,
                                "dice": dice.numpy(), "iou": iou.numpy()})
    print(f"Mejor época {best['epoch']} (Dice val {best['dice']:.3f}) | "
          f"test: Dice {test_scores['dice'].mean():.3f} IoU {test_scores['iou'].mean():.3f} | "
          f"{(time.time() - start) / 60:.1f} min")

    if save:
        save_run(arch, seed, best, pd.DataFrame(history), test_scores)
    return pd.DataFrame(history), test_scores


def update_csv(name: str, rows: pd.DataFrame, keys: list[str]):
    """Acumula las filas en un CSV, reemplazando las de una corrida repetida."""
    path = SEG_RESULTS_DIR / name
    previous = pd.read_csv(path) if path.exists() else pd.DataFrame()
    table = pd.concat([previous, rows]).drop_duplicates(keys, keep="last").sort_values(keys)
    table.to_csv(path, index=False)


def save_run(arch, seed, best, history, test_scores):
    """Guarda los pesos y acumula las métricas de la corrida."""
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"arch": arch, "seed": seed, "epoch": best["epoch"], "val_dice": best["dice"],
                "channels": CHANNELS, "strides": STRIDES, "image_size": IMAGE_SIZE,
                "state_dict": best["state"]},
               MODELS_DIR / f"{arch}_seed{seed}.pt")

    summary = pd.DataFrame([{
        "arch": arch, "seed": seed, "best_epoch": best["epoch"], "epochs_run": len(history),
        "val_dice": round(float(best["dice"]), 4),
        "test_dice": round(float(test_scores["dice"].mean()), 4),
        "test_dice_std": round(float(test_scores["dice"].std()), 4),
        "test_dice_min": round(float(test_scores["dice"].min()), 4),
        "test_iou": round(float(test_scores["iou"].mean()), 4),
        "minutes": round(float(history["minutes"].iloc[-1]), 1),
    }])
    update_csv("history.csv", history, ["arch", "seed", "epoch"])
    update_csv("test_per_image.csv", test_scores, ["arch", "seed", "image_id"])
    update_csv("test_metrics.csv", summary, ["arch", "seed"])


def main():
    parser = argparse.ArgumentParser(description="Entrena una red de segmentación pulmonar")
    parser.add_argument("--arch", choices=list(ARCHITECTURES), default="unet")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true",
                        help="prueba rápida: 2 épocas con 24 imágenes, sin guardar nada")
    args = parser.parse_args()

    SEG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        train(args.arch, args.seed, epochs=2, limit=24, save=False)
    else:
        train(args.arch, args.seed)


if __name__ == "__main__":
    main()
