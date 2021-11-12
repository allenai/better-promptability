from itertools import accumulate, cycle, islice
import math
import random
from typing import Iterable, Mapping, Union, TypeVar, Sequence, Optional, Iterator, Generic

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
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


def _tensorize(sequence: np.ndarray, name: str, output_mode: str, label_key: str) -> torch.Tensor:
    # TODO: this can be smarter
    dtype = torch.long
    if name == label_key and output_mode == "regression":
        dtype = torch.float
    elif "attention_mask" in name or "label_mask" in name:
        dtype = torch.bool
    return torch.tensor(sequence, dtype=dtype)


def collate_fn(
    batch: list[dict[str, list]],
    label_key: str,
    pad_token_map: Mapping[str, PAD_TYPE],
    padding_side: str,
    output_mode: str,
) -> dict[str, torch.Tensor]:
    """
    Input:
        pad_token_map: specifies the padding for each key. Only keys including in this map plus the
            label will be included in the batch. By default, the labels will NOT be padded, but if
            it needs to be padded, simply pass it as a part of pad_token_map.
    """
    # This is a bit ad-hoc to deal with 3d elements, but it works
    for e in batch:
        for k, v in e.items():
            if k in pad_token_map and isinstance(v[0], list):
                _pad_last_dim(v, pad_token_map[k], padding_side)

    batch = [{k: np.array(v) for k, v in e.items()} for e in batch]
    max_shapes = _find_max_shapes(batch, pad_token_map.keys())
    for i, e in enumerate(batch):
        batch[i] = {label_key: e[label_key]} | {
            k: _pad(e[k], pad_token, max_shapes[k] - np.array(e[k].shape), padding_side)
            for k, pad_token in pad_token_map.items()
        }  # dict concatenation overrides label_key if present
        batch[i] = {k: _tensorize(v, k, output_mode, label_key) for k, v in batch[i].items()}
    return default_collate(batch)


T = TypeVar("T")


class MixerStreamDataset(IterableDataset[T]):
    """
    This dataset mixes multiple other datasets into a single :class:`IterableDataset`,
    which becomes an infinite stream of elements from the original datasets, sampled
    according to the given probabilities.
    """

    def __init__(
        self,
        datasets: list[Sequence[T]],
        probabilities: Optional[list[float]] = None,
        seed: int = 27,
    ):
        self.seed = seed
        self.datasets = datasets
        self.probabilities = (
            probabilities
            if probabilities is not None
            else [1.0 / len(self.datasets) for _ in range(len(self.datasets))]
        )
        assert len(self.datasets) == len(self.probabilities)
        self.cumulative_dist = list(accumulate(self.probabilities))
        assert math.isclose(self.cumulative_dist[-1], 1.0)

    def __iter__(self) -> Iterator[T]:
        data_streams: list[Iterator[T]] = [
            cycle(_SingleStreamIterator(dataset, seed=self.seed)) for dataset in self.datasets
        ]
        return self.shard_iterable((next(data_streams[idx]) for idx in self.indices_stream()))

    def next_idx(self) -> int:
        p = random.random()
        for idx, cutoff in enumerate(self.cumulative_dist):
            if p < cutoff:
                return idx
        return len(self.cumulative_dist) - 1

    def indices_stream(self):
        while True:
            yield self.next_idx()

    def shard_iterable(self, iterable: Iterable[T]) -> Iterator[T]:
        """
        Helper method that determines which items in an iterable object to skip based
        on the current node rank (for distributed training) and worker ID (for multi-process data loading).
        """
        sharded_slice: Iterator[T] = iter(iterable)

        if dist.is_available() and dist.is_initialized():
            sharded_slice = islice(sharded_slice, dist.get_rank(), None, dist.get_world_size())

        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            sharded_slice = islice(sharded_slice, worker_info.id, None, worker_info.num_workers)

        return sharded_slice


class _SingleStreamIterator(Generic[T]):
    def __init__(self, dataset: Sequence[T], seed: int = 27):
        self.dataset = dataset
        self.epochs = 0
        self.seed = seed

    def __iter__(self) -> Iterator[T]:
        indices = list(range(len(self.dataset)))
        random.seed(self.seed + self.epochs)
        random.shuffle(indices)
        self.epochs += 1
        return (self.dataset[i] for i in indices)
