import numpy as np
import pytest
import torch

from tbseg.segmentation import medsam


def two_lungs(size=64, shift=0):
    mask = np.zeros((size, size), dtype=bool)
    mask[10 + shift:50 + shift, 5:25] = True
    mask[10 + shift:50 + shift, 39:59] = True
    return mask


def test_lung_boxes_one_per_lung_left_to_right():
    assert medsam.lung_boxes(two_lungs()).tolist() == [[5, 10, 24, 49], [39, 10, 58, 49]]


def test_lung_boxes_splits_joined_lungs_by_central_column():
    mask = two_lungs()
    mask[20:25, 25:39] = True
    assert medsam.lung_boxes(mask).tolist() == [[5, 10, 31, 49], [32, 10, 58, 49]]


def test_lung_boxes_splits_when_two_largest_regions_share_a_side():
    mask = two_lungs()
    mask[30, 5:25] = False     # el pulmón izquierdo queda dividido en dos regiones
    mask[:, 39:] = False
    mask[10:12, 40:42] = True  # el otro lado solo conserva una mancha pequeña
    assert medsam.lung_boxes(mask).tolist() == [[5, 10, 24, 49], [40, 10, 41, 11]]


def test_lung_boxes_without_lung_on_one_side_fails():
    mask = two_lungs()
    mask[:, 32:] = False
    with pytest.raises(ValueError):
        medsam.lung_boxes(mask)


def test_expand_stays_inside_image():
    assert medsam.expand([[5, 10, 24, 60]], 8, size=64).tolist() == [[0, 2, 32, 63]]


def test_prior_boxes_is_coordinate_median():
    prior = medsam.prior_boxes(two_lungs(shift=s) for s in (0, 2, 10))
    assert prior.tolist() == [[5, 12, 24, 51], [39, 12, 58, 51]]


def test_largest_component_and_empty_mask():
    mask = two_lungs()
    mask[:, 32:] = False
    mask[60:62, 50:52] = True
    assert medsam.largest_component(mask).sum() == 40 * 20
    assert not medsam.largest_component(np.zeros((8, 8), dtype=bool)).any()


def test_preprocess_shape_range_and_channels():
    image = np.random.default_rng(0).integers(0, 256, (64, 64)).astype(np.uint8)
    x = medsam.preprocess(image)
    assert x.shape == (1, 3, 1024, 1024) and x.min() == 0 and x.max() == 1
    assert torch.equal(x[0, 0], x[0, 2])


def test_preprocess_constant_image_does_not_fail():
    assert torch.isfinite(medsam.preprocess(np.full((32, 32), 7, dtype=np.uint8))).all()


def test_overlap_matches_definition():
    pred, target = np.zeros((4, 4), dtype=bool), np.zeros((4, 4), dtype=bool)
    pred[:2], target[1:3] = True, True
    assert medsam.overlap(pred, target) == pytest.approx((0.5, 1 / 3))
    assert medsam.overlap(target, target) == (1.0, 1.0)


def test_prompt_prior_uses_expanded_prior_boxes():
    prompt = {"prior_boxes": [[10, 10, 20, 20], [30, 10, 40, 20]], "margin": 5, "strategy": "prior", "refine_margin": None}
    assert medsam.prompt_boxes(None, None, prompt, "cpu").tolist() == [[5, 5, 25, 25], [25, 5, 45, 25]]


def test_refine_uses_mask_box_and_keeps_box_without_mask(monkeypatch):
    def segment(model, embedding, boxes, device, size=64):
        probs = np.zeros((2, size, size))
        probs[0, 12:18, 14:19] = 1  # la segunda caja no encuentra pulmón
        return probs

    monkeypatch.setattr(medsam, "segment", segment)
    boxes = np.array([[5, 5, 25, 25], [30, 5, 50, 25]])
    assert medsam.refine_boxes(None, None, boxes, 2, "cpu").tolist() == [[12, 10, 20, 19], [30, 5, 50, 25]]


def selection_case():
    target = np.zeros((512, 512), dtype=bool)
    target[100:400, 80:230] = True
    target[100:400, 280:430] = True
    return {"mask": target, "embedding": None}


def box_segment(model, embedding, boxes, device, size=512):
    probs = np.zeros((len(boxes), size, size))
    for k, (x0, y0, x1, y1) in enumerate(np.asarray(boxes).astype(int)):
        probs[k, y0:y1 + 1, x0:x1 + 1] = 1
    return probs


def test_select_prompt_picks_best_margin_and_prefers_prior_on_tie(monkeypatch):
    monkeypatch.setattr(medsam, "segment", box_segment)
    prior = np.array([[90, 110, 219, 389], [290, 110, 419, 389]])  # 10 px más pequeña que el pulmón
    prompt, selection = medsam.select_prompt(None, prior, [selection_case()], "cpu")
    assert prompt["margin"] == 10 and prompt["strategy"] == "prior" and prompt["refine_margin"] is None
    assert len(selection) == len(medsam.PRIOR_MARGINS) + len(medsam.REFINE_MARGINS)
    assert selection["selected"].sum() == 1
    assert selection.loc[selection["selected"], "val_dice"].iloc[0] == pytest.approx(prompt["val_dice"])


def test_select_prompt_picks_refinement_when_it_improves(monkeypatch):
    def segment(model, embedding, boxes, device, size=512):
        # Con la caja a priori la máscara recorta el pulmón 5 px; la caja refinada con margen 5 lo cubre exacto
        boxes = np.asarray(boxes).astype(int)
        if boxes[0].tolist() == [80, 100, 230, 400]:
            probs = np.zeros((2, size, size))
            probs[0, 105:395, 85:225] = 1
            probs[1, 105:395, 285:425] = 1
            return probs
        return box_segment(model, embedding, boxes, device, size)

    monkeypatch.setattr(medsam, "segment", segment)
    monkeypatch.setattr(medsam, "PRIOR_MARGINS", (40,))
    monkeypatch.setattr(medsam, "REFINE_MARGINS", (0, 5))
    prior = np.array([[120, 140, 190, 360], [320, 140, 390, 360]])
    prompt, _ = medsam.select_prompt(None, prior, [selection_case()], "cpu")
    assert prompt["strategy"] == "refined" and prompt["margin"] == 40 and prompt["refine_margin"] == 5


def test_prompt_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(medsam, "PROMPT_JSON", tmp_path / "prompt.json")
    with pytest.raises(FileNotFoundError):
        medsam.load_prompt()
    prompt = {"prior_boxes": [[1, 2, 3, 4], [5, 6, 7, 8]], "margin": 10, "strategy": "refined", "refine_margin": 5, "val_dice": 0.9}
    medsam.save_prompt(prompt)
    assert medsam.load_prompt() == prompt
