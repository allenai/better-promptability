from __future__ import annotations

import random
from typing import Any, Optional, Union

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from tango.common import Tqdm


class MixerDataset(Dataset):
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
        datasets: list[HFDataset],
        sampling_cap: Optional[int] = None,
        seed: int = 3,  # this is important during distributed training
    ):
        self._datasets: list[Union[Dataset, HFDataset]] = []
        self._total_size: int = 0
        for dataset in Tqdm.tqdm(datasets, desc="Mixing datasets"):
            if sampling_cap is not None and len(dataset) > sampling_cap:
                self._total_size += sampling_cap
                self._datasets.append(_UndersampledDataset(dataset, sampling_cap, seed=seed))
            else:
                self._total_size += len(dataset)
                self._datasets.append(dataset)

    def __getitem__(self, key: int) -> Any:  # type: ignore[override]
        for dataset in self._datasets:
            if key < len(dataset):
                return dataset[key]
            key -= len(dataset)
        raise IndexError("index out of bounds")

    def __len__(self) -> int:
        return self._total_size

    def get_all_example_lens(self) -> list[int]:
        lens = []
        for dataset in Tqdm.tqdm(self._datasets, desc="Getting lengths for sampler"):
            if isinstance(dataset, HFDataset):
                lens.extend(dataset["sort_key_len"])
            elif isinstance(dataset, _UndersampledDataset):
                lens.extend(dataset.get_active_example_lens())
            else:
                assert False
        return lens

    def resample(self):
        for dataset in self._datasets:
            if isinstance(dataset, _UndersampledDataset):
                dataset.resample()


class _UndersampledDataset(Dataset):
    def __init__(
        self,
        dataset: HFDataset,
        sampling_cap: int,
        seed: int = 3,
    ):
        assert sampling_cap < len(dataset)
        self._dataset = dataset
        self._sampling_cap = sampling_cap
        self._indices = list(range(len(self._dataset)))
        self._num_taken = sampling_cap
        self._seed = seed

        # It's important that we can shuffle deterministically in order to guarantee
        # that different processes shuffle the data in exactly the same way during distributed
        # data parallel training, so we always set the seed before shuffling in this class.
        # However, we don't want to mess with the RNG state outside of this class, so
        # we make sure to reset it right after we shuffle.
        state = random.getstate()
        random.seed(self._seed)
        random.shuffle(self._indices)
        random.setstate(state)

    def __getitem__(self, i: int) -> Any:  # type: ignore[override]
        if i > self._sampling_cap:
            raise IndexError("index out of bounds")
        return self._dataset[self._indices[i]]

    def __len__(self) -> int:
        return self._sampling_cap

    def get_active_example_lens(self) -> list[int]:
        return self._dataset.select(self._indices[: self._sampling_cap])["sort_key_len"]

    def resample(self):
        self._seed += 1
        state = random.getstate()
        random.seed(self._seed)
        if self._num_taken + self._sampling_cap <= len(self._dataset):
            # Re-organize `self._indices` so that the latest used chunk is pulled off and put on the end.
            self._indices = (
                self._indices[self._sampling_cap :] + self._indices[: self._sampling_cap]
            )
            self._num_taken += self._sampling_cap
        else:
            # Re-shuffle `self._indices` in a way that ensures the last chunk we have got to is
            # used next.
            used = (
                self._indices[: self._sampling_cap]
                + self._indices[self._sampling_cap + (len(self._dataset) - self._num_taken) :]
            )
            unused = self._indices[
                self._sampling_cap : self._sampling_cap + (len(self._dataset) - self._num_taken)
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

            self._num_taken = self._sampling_cap
        random.setstate(state)

    def fast_forward(self, num_epochs):
        # Technically we can manipulate self._seed, self._indices, and self._num_taken directly,
        # but this is easier and I think not much slower
        for _ in range(num_epochs):
            self.resample()
