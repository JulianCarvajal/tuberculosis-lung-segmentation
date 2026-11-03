# Tuberculosis Lung Segmentation

Proyecto de investigación para evaluar diferentes técnicas de segmentación
pulmonar y su impacto en la extracción de características radiómicas
a partir de radiografías de tórax.

## Objective

Evaluate different lung segmentation strategies and their effect on the
stability, reproducibility and predictive usefulness of radiomic features
for tuberculosis detection.

## Datasets

Datasets are not included in this repository.

**Montgomery + Shenzhen** (working dataset): 704 chest X-rays with manual lung
masks and TB labels (Montgomery 138, Shenzhen 566; 345 TB, 359 normal), from the
U.S. National Library of Medicine [Jaeger et al., 2014]. It is used both to train the
segmentation networks and to train the TB classifiers. We use the Kaggle version
[Chest Xray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels).

**TBX11K** (planned, optional): https://github.com/yun-liu/Tuberculosis. It has no lung
masks, so it would only be segmented with the trained networks (images they never saw),
to train or validate the TB classifiers.

## Setup

Requires Python 3.10+ and `git`. A CUDA GPU is optional (only used to train the segmentation networks).

### 1. Clone the repository

```bash
git clone git@github.com:JulianCarvajal/tuberculosis-lung-segmentation.git
cd tuberculosis-lung-segmentation
```

### 2. Create the virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

```bash
source .venv/bin/activate        # Linux / macOS / WSL
.venv\Scripts\activate           # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the project code (`src/tbseg`) as a package, so it can be
imported from scripts and notebooks (`from tbseg.config import SEGTRAIN_CSV`).
`pyradiomics` is installed from its GitHub tag because the PyPI release does not build with numpy 2.

### 5. Download Montgomery + Shenzhen and prepare it

Download [Chest Xray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
and place its `CXR_png/`, `masks/` and `ClinicalReadings/` folders in
`data/external/montgomery_shenzhen/`. Then run:

```bash
python -m tbseg.segmentation.prepare
```

Writes 512x512 images and binary masks to `data/processed/segtrain/` and the
split to `data/metadata/segtrain_split.csv` (versioned): 80/20 train/test by patient,
stratified by source and diagnosis; 10% of the training patients are held out as
validation, used only for early stopping (72/8/20 overall). The same split is used
for the segmentation networks and the TB classifiers.

### 6. Train the segmentation networks and generate the lung masks

```bash
python -m tbseg.segmentation.train --arch unet
python -m tbseg.segmentation.train --arch attention_unet
python -m tbseg.segmentation.predict --arch unet
python -m tbseg.segmentation.predict --arch attention_unet
```

`--seed` defaults to 42. Training saves the weights to `models/<arch>_seed<seed>.pt` and the metrics to `results/segmentation/`.
Prediction writes the raw masks (threshold 0.5, no post-processing) to
`masks/<dataset>/<method>/` and a per-image quality check (Dice against the manual
mask, connected components, area) to `results/segmentation/mask_qc_<dataset>.csv`.

### 7. Extract the first-order radiomic features

```bash
python -m tbseg.radiomics.extract --method unet
python -m tbseg.radiomics.extract --method attention_unet
```

Writes one table per segmentation method to
`results/radiomics/<dataset>_firstorder_<method>.csv` (versioned): one row per image
(same rows and order in every file) with `image_id, patient_id, source, tb, split` and
the 18 pyradiomics first-order features. All methods share the same parameters,
`src/tbseg/radiomics/firstorder.yaml`, so only the mask changes between files.

### 8. Run the tests

```bash
pytest
```

### Optional: TBX11K subset

Download TBX11K and extract it so that images end up in
`data/raw/TBX11K/imgs/{tb,health,sick,test,extra}`. Then run:

```bash
python -m tbseg.data
```

Writes `data/metadata/subset.csv` (versioned): 1600 images (800 TB, 400 healthy,
400 sick non-TB), exact duplicates removed, with 5 stratified folds. Running it
again produces the same file.

## Project structure

```
data/raw/                  TBX11K (not versioned)
data/external/             Montgomery + Shenzhen as downloaded (not versioned)
data/processed/            512x512 images and masks (not versioned)
data/metadata/             splits, subset and dataset index (versioned)
models/                    trained weights (not versioned)
masks/<dataset>/<method>/  predicted lung masks per segmentation method (not versioned)
results/segmentation/      training metrics and mask quality checks
results/radiomics/         first-order features, one CSV per segmentation method
notebooks/                 exploration and analysis (01 EDA, 02 segmentation)
src/tbseg/                 project code: data.py (TBX11K), segmentation/, radiomics/
tests/                     tests
```

## References

S. Jaeger, S. Candemir, S. Antani, Y.-X. J. Wáng, P.-X. Lu, and G. Thoma, "Two public
chest X-ray datasets for computer-aided screening of pulmonary diseases," *Quant.
Imaging Med. Surg.*, vol. 4, no. 6, pp. 475–477, 2014.
