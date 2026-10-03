import numpy as np
import torch

from tbseg.segmentation.train import augment, build_model, overlap_metrics


def make_pair(size=64):
    img = np.random.default_rng(0).random((size, size)).astype(np.float32)
    mask = np.zeros((size, size), dtype=np.float32)
    mask[16:48, 16:48] = 1.0
    return img, mask


def test_augment_keeps_mask_binary_and_image_in_range():
    img, mask = make_pair()
    rng = np.random.default_rng(0)
    for _ in range(10):
        a_img, a_mask = augment(img, mask, rng)
        assert a_img.shape == img.shape and a_mask.shape == mask.shape
        assert set(np.unique(a_mask)) <= {0.0, 1.0}
        assert 0.0 <= a_img.min() and a_img.max() <= 1.0


def test_overlap_metrics_extremes():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4, :] = 1
    perfect = torch.where(target > 0, 10.0, -10.0)   # logits: predicción idéntica
    inverted = -perfect                              # predicción disjunta

    dice, iou = overlap_metrics(perfect, target)
    assert dice.item() == 1.0 and iou.item() == 1.0
    dice, iou = overlap_metrics(inverted, target)
    assert dice.item() == 0.0 and iou.item() == 0.0


def test_models_return_one_channel_same_size():
    x = torch.randn(1, 1, 64, 64)
    for arch in ("unet", "attention_unet"):
        assert build_model(arch)(x).shape == (1, 1, 64, 64)
