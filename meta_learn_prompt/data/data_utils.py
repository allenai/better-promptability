from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Union

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

PAD_TYPE = Union[int, float, bool]


def _find_max_shapes(
    batch: list[dict[str, np.ndarray]], allow_keys: Iterable[str]
) -> dict[str, np.ndarray]:
    max_shapes = {}
    for e in batch:
        for k, v in e.items():
            if k not in allow_keys:
                continue
            shape = np.array(v.shape)
            if k not in max_shapes:
                max_shapes[k] = shape
            else:
                try:
                    max_shapes[k] = np.maximum(max_shapes[k], shape)
                except ValueError:  # more informed error message
                    raise ValueError(f"Different shapes for {k}: {max_shapes[k]} vs. {shape}")
    return max_shapes


def _pad_last_dim(sequence: list[list], padding_token: PAD_TYPE, padding_side: str):
    """
    In-place pads the last dimension of a 2d list.
    """
    assert padding_side in {"left", "right"}
    max_len = max(len(e) for e in sequence)
    for i, e in enumerate(sequence):
        pad_len = max_len - len(e)
        sequence[i] = (
            ([padding_token] * pad_len if padding_side == "left" else [])
            + e
            + ([padding_token] * pad_len if padding_side == "right" else [])
        )


def _pad(
    sequence: np.ndarray, padding_token: PAD_TYPE, padding_shape: np.ndarray, padding_side: str
) -> np.ndarray:
    assert padding_side in {"left", "right"}
    if sequence is None:
        return None
    padding = [(p, 0) if padding_side == "left" else (0, p) for p in padding_shape]
    return np.pad(sequence, padding, constant_values=padding_token)


def _tensorize(sequence: np.ndarray, name: str) -> torch.Tensor:
    dtype = torch.long
    if "_mask" in name:
        dtype = torch.bool
    return torch.tensor(sequence, dtype=dtype)


def collate_fn(
    batch: list[dict[str, list]],
    pad_token_map: Mapping[str, PAD_TYPE],
    padding_side: str,
) -> dict[str, torch.Tensor]:
    """
    Input:
        pad_token_map: specifies the padding for each key. Only keys including in this map
            will be included in the batch.
    """
    # This is a bit ad-hoc to deal with 3d elements, but it works
    for e in batch:
        for k, v in e.items():
            if k in pad_token_map and isinstance(v[0], list):
                _pad_last_dim(v, pad_token_map[k], padding_side)

    batch = [{k: np.array(v) for k, v in e.items()} for e in batch]
    max_shapes = _find_max_shapes(batch, pad_token_map.keys())
    for i, e in enumerate(batch):
        batch[i] = {
            k: _pad(e[k], pad_token, max_shapes[k] - np.array(e[k].shape), padding_side)
            for k, pad_token in pad_token_map.items()
        }
        batch[i] = {k: _tensorize(v, k) for k, v in batch[i].items()}
    return default_collate(batch)


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()
