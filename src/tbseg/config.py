"""Rutas y parámetros compartidos por todo el proyecto."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
TBX11K_DIR = DATA_DIR / "raw" / "TBX11K"
TBX11K_IMG_DIR = TBX11K_DIR / "imgs"
METADATA_DIR = DATA_DIR / "metadata"
SUBSET_CSV = METADATA_DIR / "subset.csv"

# Dataset externo con máscaras de pulmón (Montgomery + Shenzhen) para entrenar la segmentación
SEGTRAIN_RAW_DIR = DATA_DIR / "external" / "montgomery_shenzhen"
SEGTRAIN_DIR = DATA_DIR / "processed" / "segtrain"
SEGTRAIN_CSV = METADATA_DIR / "segtrain_split.csv"
SEGTRAIN_SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

MODELS_DIR = ROOT / "models"
MASKS_DIR = ROOT / "masks"
RESULTS_DIR = ROOT / "results"

SEED = 42
N_FOLDS = 5
IMAGE_SIZE = 512

# Carpetas de TBX11K con etiqueta pública -> imágenes a muestrear para el subconjunto
SUBSET_SIZES = {"tb": 800, "health": 400, "sick": 400}
POSITIVE_CLASS = "tb"
