from __future__ import annotations
import hashlib
import random
from typing import Iterable, Mapping, Union, TypeVar, Sequence, Optional

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


T = TypeVar("T")


class MixerDataset(Sequence[T]):
    """
    This dataset mixes multiple other datasets into a single :class:`Dataset`.

    The `sampling_cap` argument sets an artificial size limit for all of the datasets which
    controls the sampling probability for each. This is useful when you have a mix of small
    and large datasets. When using `sampling_cap`, you should call :meth:`resample()` after every
    epoch to randomize the examples that get picked from the undersampled datasets, i.e. the datasets
    that are bigger than `sampling_cap`.
    """

    def __init__(
        self,
        datasets: list[Sequence[T]],
        sampling_cap: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self._datasets: list[Sequence[T]] = []
        self._total_size: int = 0
        self._dataset_boundaries: list[tuple[int, int]] = []
        for dataset in datasets:
            start_boundary = 0 if not self._dataset_boundaries else self._dataset_boundaries[-1][-1]
            if sampling_cap is not None and len(dataset) > sampling_cap:
                self._total_size += sampling_cap
                self._dataset_boundaries.append((start_boundary, start_boundary + sampling_cap))
                self._datasets.append(_UndersampledDataset(dataset, sampling_cap, seed=seed))
            else:
                self._total_size += len(dataset)
                self._dataset_boundaries.append((start_boundary, start_boundary + len(dataset)))
                self._datasets.append(dataset)

    def __getitem__(self, i: int) -> T:  # type: ignore[override]
        for dataset_idx, (start, end) in enumerate(self._dataset_boundaries):
            if start <= i < end:
                return self._datasets[dataset_idx][i - start]
        raise IndexError("index out of bounds")

    def __len__(self) -> int:
        return self._total_size

    def resample(self, seed: Optional[int] = None):
        for dataset in self._datasets:
            if isinstance(dataset, _UndersampledDataset):
                dataset.resample(seed)


class _UndersampledDataset(Sequence[T]):
    def __init__(self, dataset: Sequence[T], sampling_cap: int, seed: Optional[int] = None):
        assert sampling_cap < len(dataset)
        self._dataset = dataset
        self._sampling_cap = sampling_cap
        self._indices = list(range(len(self._dataset)))
        self._max_taken = sampling_cap
        if seed is not None:
            random.seed(seed)
        #  random.shuffle(self._indices)

    def __getitem__(self, i: int) -> T:  # type: ignore[override]
        if i > self._sampling_cap:
            raise IndexError("index out of bounds")
        return self._dataset[self._indices[i]]

    def __len__(self) -> int:
        return self._sampling_cap

    def resample(self, seed: Optional[int]):
        if seed is not None:
            random.seed(seed)
        if self._max_taken + self._sampling_cap <= len(self._dataset):
            # Re-organize `self._indices` so that the latest used chunk is pulled off and put on the end.
            self._indices = (
                self._indices[self._sampling_cap :] + self._indices[: self._sampling_cap]
            )
            self._max_taken += self._sampling_cap
        else:
            # Re-shuffle `self._indices` in a way that ensures the last chunk we have got to is
            # used next.
            used = (
                self._indices[: self._sampling_cap]
                + self._indices[self._sampling_cap + (len(self._dataset) - self._max_taken) :]
            )
            unused = self._indices[
                self._sampling_cap : self._sampling_cap + (len(self._dataset) - self._max_taken)
            ]
            # `used` will be sliced up and moved around before being added back into `self._indices`,
            # so we shuffle it now to add randomness.
            random.shuffle(used)

            # `next_up` is the next chunk of `self._sampling_cap` which will include all
            # of `unused` and however many examples from `used` that we need to reach
            # `self._sampling_cap` instances.
            next_up = unused + used[: self._sampling_cap - len(unused)]
            random.shuffle(next_up)

            # Put everything back together into `self._indices`.
            self._indices = next_up + used[self._sampling_cap - len(unused) :]

            # clean up to hopefully help GC
            del used, unused, next_up

            self._max_taken = self._sampling_cap
