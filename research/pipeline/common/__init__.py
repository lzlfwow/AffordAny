from .segmentation import decode_segmentation_mask

__all__ = ["decode_segmentation_mask", "prepare_live_chain_object"]


def prepare_live_chain_object(*args, **kwargs):
    from .live_chain import prepare_live_chain_object as _prepare_live_chain_object

    return _prepare_live_chain_object(*args, **kwargs)
