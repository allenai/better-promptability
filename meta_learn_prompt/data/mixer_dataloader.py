from __future__ import annotations

import random

from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter


class MixerDataLoader(DataLoader):
    def __init__(self, dataloaders: list[DataLoader], meta_batch_size: int):
        self._dataloaders = dataloaders
        self._meta_batch_size = meta_batch_size
        self._total_len = sum(len(dataloader) for dataloader in dataloaders) // meta_batch_size
        self._weights = [len(dataloader) for dataloader in dataloaders]

    def __iter__(self) -> _BaseDataLoaderIter:
        batches = []
        for _ in range(self._meta_batch_size):
            assert all(w >= 0 for w in self._weights)
            if all(w == 0 for w in self._weights):
                break
            # TODO: this might not be robust in distributed training
            dataloader_idx = random.choices(range(len(self._dataloaders)), self._weights)[0]
            self._weights[dataloader_idx] -= 1
            dataloader = self._dataloaders[dataloader_idx]
            batches.append(next(dataloader))
        assert len(batches) > 0
        yield batches

    def __len__(self) -> int:
        return self._total_len
