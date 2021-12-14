from __future__ import annotations

import random
from typing import Optional, Sequence, TypeVar

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
    ):
        self._datasets: list[Sequence[T]] = []
        self._total_size: int = 0
        self._dataset_boundaries: list[tuple[int, int]] = []
        for dataset in datasets:
            start_boundary = 0 if not self._dataset_boundaries else self._dataset_boundaries[-1][-1]
            if sampling_cap is not None and len(dataset) > sampling_cap:
                self._total_size += sampling_cap
                self._dataset_boundaries.append((start_boundary, start_boundary + sampling_cap))
                self._datasets.append(_UndersampledDataset(dataset, sampling_cap))
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

    def resample(self):
        for dataset in self._datasets:
            if isinstance(dataset, _UndersampledDataset):
                dataset.resample()


class _UndersampledDataset(Sequence[T]):
    def __init__(
        self,
        dataset: Sequence[T],
        sampling_cap: int,
    ):
        assert sampling_cap < len(dataset)
        self._dataset = dataset
        self._sampling_cap = sampling_cap
        self._indices = list(range(len(self._dataset)))
        self._num_taken = sampling_cap
        random.shuffle(self._indices)

    def __getitem__(self, i: int) -> T:  # type: ignore[override]
        if i > self._sampling_cap:
            raise IndexError("index out of bounds")
        return self._dataset[self._indices[i]]

    def __len__(self) -> int:
        return self._sampling_cap

    def resample(self):
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
