# Segmentación pulmonar en tuberculosis

Proyecto de investigación para evaluar diferentes técnicas de segmentación
pulmonar y su impacto en la extracción de características radiómicas
a partir de radiografías de tórax.

## Objetivo

Evaluar distintas estrategias de segmentación pulmonar y su efecto sobre la
estabilidad, la reproducibilidad y la utilidad predictiva de las características
radiómicas para la detección de tuberculosis.

## Datasets

El proyecto usa dos datasets con roles distintos:

**Montgomery + Shenzhen** — conjunto de trabajo principal.
704 radiografías con máscaras pulmonares trazadas manualmente (345 TB, 359 normales).
Se usan para entrenar los segmentadores y para extraer las características radiómicas
con las que se entrenan y evalúan los clasificadores.

> Descargar desde Kaggle:
> [Chest Xray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)

**TBX11K** — ampliación futura.
8 400 radiografías etiquetadas (800 TB, 7 600 no-TB), sin máscaras pulmonares.
TBX11K no tiene máscaras, así que los segmentadores se entrenan primero en
Montgomery + Shenzhen y luego se aplican sobre TBX11K.

> Descargar desde:
> [https://github.com/yun-liu/Tuberculosis](https://github.com/yun-liu/Tuberculosis)

Ninguno de los dos datasets se incluye en este repositorio.

## Configuración

Requiere Python 3.10+ y `git`. Una GPU con CUDA es opcional
(solo se usa para entrenar las redes de segmentación).

### 1. Clonar el repositorio

```bash
git clone git@github.com:JulianCarvajal/tuberculosis-lung-segmentation.git
cd tuberculosis-lung-segmentation
```

### 2. Crear el entorno virtual

```bash
python3 -m venv .venv
```

### 3. Activar el entorno

```bash
source .venv/bin/activate        # Linux / macOS / WSL
.venv\Scripts\activate           # Windows
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` instala el código del proyecto (`src/tbseg`) como paquete,
de modo que se puede importar desde scripts y notebooks (`from tbseg.data import load_subset`).
`pyradiomics` se instala desde su etiqueta de GitHub porque la versión de PyPI
no compila con numpy 2.

### 5. Descargar los datasets

Seguir las instrucciones del notebook `00_descarga_datasets.ipynb`, que guía
la descarga de ambos datasets, indica dónde colocar cada carpeta y verifica
que la estructura sea correcta antes de continuar.

### 6. Preparar el dataset de segmentación

```bash
python -m tbseg.segmentation.prepare
```

Escribe las imágenes y máscaras a 512 × 512 en `data/processed/segtrain/`
y la partición train/val/test (70/15/15 por paciente, estratificada por fuente
y diagnóstico) en `data/metadata/segtrain_split.csv` (versionado).

### 7. Construir el subconjunto de estudio de TBX11K

```bash
python -m tbseg.data
```

Escribe `data/metadata/subset.csv` (versionado): 1 600 imágenes (800 TB,
400 sanas, 400 enfermas no-TB), duplicados exactos eliminados, con 5 folds
estratificados. Ejecutarlo de nuevo produce el mismo archivo.

### 8. Ejecutar los tests

```bash
pytest
```

## Estructura del proyecto

```
data/raw/                    datasets originales (no versionados)
data/external/               datasets externos, p. ej. máscaras pulmonares (no versionados)
data/processed/              imágenes y máscaras preprocesadas (no versionados)
data/metadata/               subconjuntos y particiones (versionados)
models/                      pesos entrenados (no versionados)
masks/<método>/              máscaras pulmonares por método de segmentación (no versionadas)
results/                     características, métricas y figuras
src/tbseg/                   código del proyecto
tests/                       tests
notebooks/
  00_descarga_datasets.ipynb                 descarga y verificación de los datasets
  01a_exploracion_tbx11k.ipynb               exploración del dataset TBX11K
  01b_exploracion_montgomery_shenzhen.ipynb  exploración de Montgomery + Shenzhen
  02_segmentacion.ipynb                      entrenamiento y evaluación de segmentadores
```
