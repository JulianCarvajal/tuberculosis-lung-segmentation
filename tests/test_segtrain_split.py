import pandas as pd

from tbseg.segmentation.prepare import assign_split


def make_index():
    rows = [
        {"image_id": f"{src}_{i:04d}_{tb}", "source": src, "tb": tb}
        for src, n in [("shenzhen", 280), ("montgomery", 70)]
        for tb in (0, 1)
        for i in range(n)
    ]
    index = pd.DataFrame(rows)
    index["patient_id"] = index["image_id"]
    # Un paciente con tres radiografías (como los controles de Montgomery)
    repeated = ["montgomery_0001_1", "montgomery_0002_1", "montgomery_0003_1"]
    index.loc[index["image_id"].isin(repeated), "patient_id"] = repeated[0]
    return index


def test_split_proportions_per_stratum():
    split = assign_split(make_index())
    props = pd.crosstab([split["source"], split["tb"]], split["split"], normalize="index")
    # 80/20, y del 80 % un 10 % para validación: 72/8/20
    assert (props["train"].between(0.70, 0.74)).all()
    assert (props["val"].between(0.06, 0.10)).all()
    assert (props["test"].between(0.18, 0.22)).all()


def test_same_patient_same_split():
    split = assign_split(make_index())
    assert (split.groupby("patient_id")["split"].nunique() == 1).all()


def test_split_is_deterministic():
    index = make_index()
    pd.testing.assert_frame_equal(assign_split(index), assign_split(index))
