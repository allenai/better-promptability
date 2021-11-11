import random

import pytest

from meta_learn_prompt.data.data_utils import MixerStreamDataset


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_mixer_stream_dataset(seed):
    random.seed(seed)

    dataset1 = ["a", "b", "c"]
    dataset2 = ["1", "2", "3", "4"]
    mixer = MixerStreamDataset([dataset1, dataset2], [0.1, 0.9])
    counts: dict[str, int] = {"dataset1": 0, "dataset2": 0}
    for i, x in enumerate(iter(mixer)):
        assert isinstance(x, str)
        if x in dataset1:
            counts["dataset1"] += 1
        else:
            counts["dataset2"] += 1
        if i > 20:
            break
    assert counts["dataset2"] > counts["dataset1"]
