# Tuberculosis Lung Segmentation

Proyecto de investigación para evaluar diferentes técnicas de segmentación
pulmonar y su impacto en la extracción de características radiómicas
a partir de radiografías de tórax.

## Objective

Evaluate different lung segmentation strategies and their effect on the
stability, reproducibility and predictive usefulness of radiomic features
for tuberculosis detection.

## Dataset

TBX11K:
https://github.com/yun-liu/Tuberculosis

The dataset is not included in this repository.

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
imported from scripts and notebooks (`from tbseg.data import load_subset`).
`pyradiomics` is installed from its GitHub tag because the PyPI release does not build with numpy 2.

### 5. Download the dataset

Download TBX11K from the link above and extract it so that images end up in
`data/raw/TBX11K/imgs/{tb,health,sick,test,extra}`.

### 6. Build the study subset

```bash
python -m tbseg.data
```

Writes `data/metadata/subset.csv` (versioned): 1600 images (800 TB, 400 healthy,
400 sick non-TB), exact duplicates removed, with 5 stratified folds. Running it
again produces the same file.

### 7. Prepare the lung mask dataset (segmentation training)

TBX11K has no lung masks, so the segmentation networks are trained on
Montgomery + Shenzhen (704 images with manual lung masks). Download
[Chest Xray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
and place its `CXR_png/`, `masks/` and `ClinicalReadings/` folders in
`data/external/montgomery_shenzhen/`. Then run:

```bash
python -m tbseg.segmentation.prepare
```

Writes 512x512 images and binary masks to `data/processed/segtrain/` and the
train/val/test split (70/15/15, stratified by source and diagnosis) to
`data/metadata/segtrain_split.csv` (versioned).

### 8. Run the tests

```bash
pytest
```

## Project structure

```
data/raw/            original datasets (not versioned)
data/external/       external datasets, e.g. lung masks (not versioned)
data/metadata/       subset and splits (versioned)
models/              trained weights (not versioned)
masks/<method>/      lung masks per segmentation method (not versioned)
results/             features, metrics and figures
notebooks/           exploration and analysis
src/tbseg/           project code
tests/               tests
```
