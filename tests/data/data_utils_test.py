from itertools import islice
import random

from meta_learn_prompt.data.data_utils import MixerStreamDataset


def test_mixer_stream_dataset():
    random.seed(1)

    dataset1 = ["a", "b", "c"]
    dataset2 = ["1", "2", "3", "4"]
    mixer = MixerStreamDataset([dataset1, dataset2], [0.1, 0.9])
    counts: dict[str, int] = {"dataset1": 0, "dataset2": 0}
    for i, x in enumerate(islice(iter(mixer), 300)):
        assert isinstance(x, str)
        if x in dataset1:
            counts["dataset1"] += 1
        else:
            counts["dataset2"] += 1
    # Make sure the ratio of items sampled from dataset2 compared to dataset1 is about 9-to-1.
    assert 8.5 < counts["dataset2"] / counts["dataset1"] < 9.5
