import pandas as pd

from tbseg.config import SUBSET_SIZES, SUBSET_TEST_SIZE
from tbseg.data import build_subset


def make_index(n_per_class=1000, n_dup_pairs=50):
    rows = [
        {"image_id": f"{cls}{i:04d}", "rel_path": f"{cls}/{cls}{i:04d}.png",
         "source_class": cls, "md5": f"{cls}{i}", "label": int(cls == "tb")}
        for cls in SUBSET_SIZES
        for i in range(n_per_class)
    ]
    index = pd.DataFrame(rows)
    # Las primeras n_dup_pairs imágenes sanas son copias de las siguientes
    health = index.index[index["source_class"] == "health"]
    index.loc[health[:n_dup_pairs], "md5"] = index.loc[health[n_dup_pairs:2 * n_dup_pairs], "md5"].to_numpy()
    return index


def test_subset_sizes_and_no_duplicates():
    index = make_index()
    subset = build_subset(index)
    assert subset["source_class"].value_counts().to_dict() == SUBSET_SIZES
    assert subset["image_id"].is_unique
    md5 = index.set_index("image_id").loc[subset["image_id"], "md5"]
    assert md5.is_unique


def test_split_is_stratified_per_class():
    subset = build_subset(make_index())
    assert set(subset["split"]) == {"train", "test"}
    counts = pd.crosstab(subset["split"], subset["source_class"])
    for cls, n in SUBSET_SIZES.items():
        assert counts.loc["test", cls] == round(n * SUBSET_TEST_SIZE)


def test_subset_is_deterministic():
    index = make_index()
    pd.testing.assert_frame_equal(build_subset(index), build_subset(index))
