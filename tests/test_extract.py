import numpy as np
import pytest

from tbseg.radiomics.extract import build_extractor, extract_features

# Sin normalizar, para comparar directamente con numpy
RAW = {"imageType": {"Original": {}}, "featureClass": {"firstorder": []},
       "setting": {"force2D": True, "label": 1, "binWidth": 25}}


def make_case(size=64):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (size, size)).astype(np.uint8)
    image[0, 0] = 255                      # máximo cerca de 255: provoca el desborde en uint8
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[8:56, 4:28] = 255                 # máscaras guardadas como 0/255
    mask[8:56, 36:60] = 255
    return image, mask


def test_uint8_image_and_255_mask_give_valid_features():
    image, mask = make_case()
    f = extract_features(build_extractor(RAW), image, mask)
    # Con uint8 sin convertir, pyradiomics devuelve Entropy≈0 y Uniformity=1
    assert f["Entropy"] > 1 and f["Uniformity"] < 0.5


def test_features_match_numpy_inside_mask():
    image, mask = make_case()
    f = extract_features(build_extractor(RAW), image, mask)
    pixels = image[mask > 0].astype(float)
    assert f["Mean"] == pytest.approx(pixels.mean())
    assert f["Minimum"] == pixels.min() and f["Maximum"] == pixels.max()
    assert f["Variance"] == pytest.approx(pixels.var())


def test_pixels_outside_mask_do_not_matter():
    image, mask = make_case()
    extractor = build_extractor(RAW)
    changed = image.copy()
    changed[mask == 0] = 0
    assert extract_features(extractor, image, mask) == extract_features(extractor, changed, mask)


def test_project_params_give_18_first_order_features():
    image, mask = make_case()
    f = extract_features(build_extractor(), image, mask)
    assert len(f) == 18 and "StandardDeviation" not in f
