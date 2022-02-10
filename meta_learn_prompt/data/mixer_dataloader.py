from __future__ import annotations

import math
import random
from typing import Callable

import torch.distributed as dist
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
        self._meta_batch_size = self._meta_batch_size_per_device = meta_batch_size
        self._batch_postprocessor = batch_postprocessor
        if dist.is_initialized():
            self._world_size = dist.get_world_size()
            self._rank = dist.get_rank()
            assert self._meta_batch_size % self._world_size == 0
            self._meta_batch_size_per_device = self._meta_batch_size // self._world_size

        num_batches = sum(len(dataloader) for dataloader in dataloaders)
        if dist.is_initialized():
            self._total_len = num_batches // meta_batch_size
            if num_batches % meta_batch_size > self._rank:
                self._total_len += 1
        else:
            self._total_len = int(math.ceil(num_batches / meta_batch_size))
        self._weights = [len(dataloader) for dataloader in dataloaders]
        self._seed = 1

        self.num_workers = 0  # TODO: multiprocessing
        self.collate_fn = None
        self.dataset = None

    def sample_one_batch(self):
        dataloader_idx = random.choices(range(len(self._dataloader_iters)), self._weights)[0]
        self._weights[dataloader_idx] -= 1
        assert all(w >= 0 for w in self._weights)
        dataloader_iter = self._dataloader_iters[dataloader_idx]
        return next(dataloader_iter)

    def __iter__(self) -> _BaseDataLoaderIter:
        while True:
            batches = []
            for _ in range(self._meta_batch_size_per_device):
                if dist.is_initialized():
                    rngstate = random.getstate()
                    self._seed += 1
                    random.seed(self._seed)
                    for i in range(min(self._world_size, sum(self._weights))):
                        sample = self.sample_one_batch()
                        if i == self._rank:
                            batches.append(sample)
                    random.setstate(rngstate)
                else:
                    batches.append(self.sample_one_batch())

                if all(w == 0 for w in self._weights):  # early stopping
                    if len(batches) > 0:
                        yield self._batch_postprocessor(batches)
                    return
            assert len(batches) > 0
            yield self._batch_postprocessor(batches)

    def __len__(self) -> int:
        return self._total_len
