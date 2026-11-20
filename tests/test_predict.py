import numpy as np

from tbseg.segmentation.predict import mask_qc


def two_lungs(size=64):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[10:50, 5:25] = 255   # pulmón derecho del paciente (izquierda de la imagen)
    mask[10:50, 39:59] = 255  # pulmón izquierdo
    return mask


def test_mask_qc_two_lungs_matching_manual():
    mask = two_lungs()
    qc = mask_qc(mask, manual=mask)
    assert qc["n_components"] == 2 and qc["two_sides"]
    assert qc["extra_area_frac"] == 0 and qc["dice"] == 1.0


def test_mask_qc_detects_single_lung_and_blobs():
    mask = two_lungs()
    mask[:, 32:] = 0           # se pierde un pulmón
    mask[60:62, 2:4] = 255     # mancha suelta del mismo lado
    qc = mask_qc(mask, manual=two_lungs())
    assert qc["n_components"] == 2 and not qc["two_sides"]
    assert qc["dice"] < 0.7


def test_mask_qc_empty_mask():
    qc = mask_qc(np.zeros((32, 32), dtype=np.uint8))
    assert qc["area_frac"] == 0 and qc["n_components"] == 0 and not qc["two_sides"]


def test_datasets_share_label_columns():
    # extract.py y los clasificadores esperan las mismas columnas en todos los datasets
    from tbseg.segmentation.predict import DATASETS, LABEL_COLUMNS

    for name, dataset in DATASETS.items():
        labels = dataset["labels"]()
        assert list(labels.columns) == LABEL_COLUMNS, name
        assert labels["image_id"].is_unique, name
        assert set(labels["split"]) <= {"train", "val", "test"}, name
