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
# 80/20 por paciente; del 80 % de entrenamiento se aparta un 10 % solo para la detención temprana
SEGTRAIN_TEST_SIZE = 0.20
SEGTRAIN_VAL_SIZE = 0.10
# Índice con metadatos clínicos y de imagen generado por notebooks/01b_exploracion_montgomery_shenzhen
MS_INDEX_CSV = METADATA_DIR / "montgomery_shenzhen_index.csv"

MODELS_DIR = ROOT / "models"
MASKS_DIR = ROOT / "masks"
RESULTS_DIR = ROOT / "results"

SEED = 42
N_FOLDS = 5
IMAGE_SIZE = 512

# Clases de TBX11K que entran al experimento -> imágenes a muestrear. Solo TB contra sanas,
# como en Montgomery-Shenzhen: las enfermas sin TB (sick/) quedan fuera.
SUBSET_SIZES = {"tb": 800, "health": 800}
POSITIVE_CLASS = "tb"
# Carpetas de TBX11K con etiqueta pública (incluye sick/, solo para la exploración)
TBX11K_LABELED_CLASSES = ("tb", "health", "sick")
# Imágenes del subconjunto en gris, 512x512, con nombre <image_id>.png (las usan predict y extract)
TBX11K_PROCESSED_DIR = DATA_DIR / "processed" / "tbx11k"
# Partición de TBX11K para los clasificadores: 80 % desarrollo / 20 % prueba, estratificada
# por clase. No hay validación: los segmentadores no se entrenan con TBX11K.
SUBSET_TEST_SIZE = 0.20
