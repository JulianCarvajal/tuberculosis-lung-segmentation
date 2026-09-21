import pandas as pd

from tbseg.segmentation.prepare import assign_split


def make_index():
    rows = [
        {"image_id": f"{src}_{i:04d}_{tb}", "source": src, "tb": tb}
        for src, n in [("shenzhen", 280), ("montgomery", 70)]
        for tb in (0, 1)
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def test_split_proportions_per_stratum():
    split = assign_split(make_index())
    props = pd.crosstab([split["source"], split["tb"]], split["split"], normalize="index")
    assert (props["train"].between(0.68, 0.72)).all()
    assert (props["val"].between(0.13, 0.17)).all()
    assert (props["test"].between(0.13, 0.17)).all()


def test_split_is_deterministic():
    index = make_index()
    pd.testing.assert_frame_equal(assign_split(index), assign_split(index))
