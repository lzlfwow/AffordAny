from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils


def decode_segmentation_mask(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        merged = mask_utils.merge(rles) if isinstance(rles, list) else rles
        mask = mask_utils.decode(merged)
    elif isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        mask = mask_utils.decode(rle)
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segmentation).__name__}")

    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)
