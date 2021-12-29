from __future__ import annotations

import math
import random
from typing import Callable

from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter


class MixerDataLoader(DataLoader):
    """
    A dataloader that encapsulates multiple dataloaders. At each iteration, yields the next batch
    from a random dataloader.
    """

    def __init__(
        self,
        dataloaders: list[DataLoader],
        meta_batch_size: int,
        batch_postprocessor: Callable[[list], list] = lambda b: b,
    ):
        self._dataloader_iters = [iter(dataloader) for dataloader in dataloaders]
        self._meta_batch_size = meta_batch_size
        self._batch_postprocessor = batch_postprocessor
        self._total_len = int(
            math.ceil(sum(len(dataloader) for dataloader in dataloaders) / meta_batch_size)
        )
        self._weights = [len(dataloader) for dataloader in dataloaders]

        self.num_workers = 0  # TODO: multiprocessing
        self.collate_fn = None
        self.dataset = None

    def __iter__(self) -> _BaseDataLoaderIter:
        while True:
            batches = []
            for _ in range(self._meta_batch_size):
                # TODO: this might not be robust in distributed training
                dataloader_idx = random.choices(range(len(self._dataloader_iters)), self._weights)[
                    0
                ]
                self._weights[dataloader_idx] -= 1
                dataloader_iter = self._dataloader_iters[dataloader_idx]
                batches.append(next(dataloader_iter))
                assert all(w >= 0 for w in self._weights)
                if all(w == 0 for w in self._weights):
                    return
            assert len(batches) > 0
            yield self._batch_postprocessor(batches)

    def __len__(self) -> int:
        return self._total_len
