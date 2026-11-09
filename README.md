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
704 radiografías con máscaras pulmonares trazadas manualmente (345 TB, 359 normales;
Montgomery 138, Shenzhen 566), de la U.S. National Library of Medicine [Jaeger et al., 2014].
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
(solo se usa para entrenar las redes de segmentación y para ejecutar MedSAM).

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
y la partición en `data/metadata/segtrain_split.csv` (versionado): 80/20 train/test
por paciente, estratificada por fuente y diagnóstico. Del 80 % se aparta un 10 %
como validación, solo para la detención temprana (72/8/20 en total). La misma
partición se usa para las redes de segmentación y los clasificadores de TB.

### 7. (Opcional) Construir el subconjunto de estudio de TBX11K

```bash
python -m tbseg.data
```

Escribe `data/metadata/subset.csv` (versionado): 1 600 imágenes (800 TB,
400 sanas, 400 enfermas no-TB), duplicados exactos eliminados, con 5 folds
estratificados. Ejecutarlo de nuevo produce el mismo archivo.

### 8. Entrenar las redes y generar las máscaras pulmonares

```bash
python -m tbseg.segmentation.train --arch unet
python -m tbseg.segmentation.train --arch attention_unet
python -m tbseg.segmentation.predict --arch unet
python -m tbseg.segmentation.predict --arch attention_unet
```

`--seed` es 42 por defecto. El entrenamiento guarda los pesos en `models/<arch>_seed<semilla>.pt`
y las métricas en `results/segmentation/`. La predicción escribe las máscaras tal como
salen de la red (umbral 0,5, sin posprocesamiento) en `masks/<dataset>/<método>/` y un
control de calidad por imagen (Dice contra la máscara manual, componentes conectados, área)
en `results/segmentation/mask_qc_<dataset>.csv`.

MedSAM no se entrena, pero necesita una caja por pulmón:

```bash
python -m tbseg.segmentation.medsam
python -m tbseg.segmentation.predict --arch medsam
```

El primer comando descarga los pesos a `models/medsam/` si no existen, calcula la caja a priori con
las máscaras de train, elige los márgenes y la estrategia (caja a priori o caja refinada en una segunda
pasada) con el Dice de val y evalúa en test solo la opción elegida. Guarda la elección en
`results/segmentation/medsam_prompt.json` y las opciones probadas en `medsam_selection.csv`.
El segundo comando escribe las máscaras y el control de calidad con el mismo formato que las redes.

MedSAM se desarrolló con este mismo dataset (Tabla 3 del suplemento de Ma et al., 2024), así que su
Dice en estas imágenes puede ser optimista. La caja a priori es igual para todas las imágenes: lo que
se evalúa es MedSAM con esa caja, no MedSAM en general (`02_segmentacion.ipynb`, sección 6.3).

### 9. Extraer las características radiómicas de primer orden

```bash
python -m tbseg.radiomics.extract --method unet
python -m tbseg.radiomics.extract --method attention_unet
python -m tbseg.radiomics.extract --method medsam
```

Escribe una tabla por método de segmentación en
`results/radiomics/<dataset>_firstorder_<método>.csv` (versionado): una fila por imagen
(mismas filas y mismo orden en todos los archivos) con `image_id, patient_id, source, tb, split`
y las 18 características de primer orden de pyradiomics. Todos los métodos comparten los
mismos parámetros, `src/tbseg/radiomics/firstorder.yaml`, así que entre archivos solo cambia la máscara.

### 10. Entrenar y evaluar los clasificadores de TB

Ejecutar el notebook `04_deteccion_tb.ipynb`. Entrena SVM, Random Forest y una red neuronal (MLP)
una vez por técnica de segmentación, con los CSV del paso anterior y la partición de
`segtrain_split.csv`. Elige los hiperparámetros con 5 folds internos en train + val, evalúa una sola
vez en test y compara todos los pares de técnicas con intervalos de confianza por bootstrap pareado.
Se detiene si falta el CSV de alguna técnica de `METHODS` (sección 1 del notebook).

Escribe en `results/classification/<dataset>/<familias>/`: `cv_metrics.csv`, `best_params.csv`,
`test_predictions.csv`, `test_metrics.csv`, `test_comparison.csv`, `importances.csv` y, de la red
neuronal, `mlp_selection.csv`, `mlp_seeds.csv` y `mlp_variants.csv`.

### 11. Ejecutar los tests

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
masks/<dataset>/<método>/    máscaras pulmonares por método de segmentación (no versionadas)
results/segmentation/        métricas de entrenamiento y control de calidad de las máscaras
results/radiomics/           características de primer orden, un CSV por método
results/classification/      métricas, predicciones e importancias de los clasificadores de TB
src/tbseg/                   código del proyecto: data.py (TBX11K), segmentation/, radiomics/
tests/                       tests
Documentation/               tutoriales de U-Net y MedSAM
notebooks/
  00_descarga_datasets.ipynb                 descarga y verificación de los datasets
  01a_exploracion_tbx11k.ipynb               exploración del dataset TBX11K
  01b_exploracion_montgomery_shenzhen.ipynb  exploración de Montgomery + Shenzhen
  02_segmentacion.ipynb                      entrenamiento, evaluación y control de calidad de segmentadores
  03_radiomica.ipynb                         revisión de las características radiómicas antes de los clasificadores
  04_deteccion_tb.ipynb                      entrenamiento y evaluación de los clasificadores de TB, comparación entre técnicas
```

## Referencias

S. Jaeger, S. Candemir, S. Antani, Y.-X. J. Wáng, P.-X. Lu, and G. Thoma, "Two public
chest X-ray datasets for computer-aided screening of pulmonary diseases," *Quant.
Imaging Med. Surg.*, vol. 4, no. 6, pp. 475–477, 2014.

J. Ma, Y. He, F. Li, L. Han, C. You, and B. Wang, "Segment anything in medical images,"
*Nat. Commun.*, vol. 15, art. 654, 2024.
