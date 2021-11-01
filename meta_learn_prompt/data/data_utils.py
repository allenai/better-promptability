from typing import Iterable, Union

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


def _pad(
    sequence: np.ndarray, padding_token: PAD_TYPE, padding_shape: np.ndarray, padding_side: str
) -> np.ndarray:
    assert padding_side in {"left", "right"}
    if sequence is None:
        return None
    padding = [(p, 0) if padding_side == "left" else (0, p) for p in padding_shape]
    return np.pad(sequence, padding, constant_values=padding_token)


def _tensorize(sequence: np.ndarray, name: str, output_mode: str, label_key: str) -> torch.Tensor:
    # TODO: this can be smarter
    dtype = torch.long
    if name == label_key and output_mode == "regression":
        dtype = torch.float
    elif "attention_mask" in name:
        dtype = torch.bool
    return torch.tensor(sequence, dtype=dtype)


def collate_fn(
    batch: list[dict[str, list]],
    label_key: str,
    pad_token_map: dict[str, PAD_TYPE],
    padding_side: str,
    output_mode: str,
) -> dict[str, torch.Tensor]:
    """
    Input:
        pad_token_map: specifies the padding for each key. Only keys including in this map plus the
            label will be included in the batch. By default, the labels will NOT be padded, but if
            it needs to be padded, simply pass it as a part of pad_token_map.
    """
    batch = [{k: np.array(v) for k, v in e.items()} for e in batch]
    max_shapes = _find_max_shapes(batch, pad_token_map.keys())
    for i, e in enumerate(batch):
        batch[i] = {label_key: e[label_key]} | {
            k: _pad(e[k], pad_token, max_shapes[k] - np.array(e[k].shape), padding_side)
            for k, pad_token in pad_token_map.items()
        }  # dict concatenation overrides label_key if present
        batch[i] = {k: _tensorize(v, k, output_mode, label_key) for k, v in batch[i].items()}
    return default_collate(batch)