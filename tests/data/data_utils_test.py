import pytest

from meta_learn_prompt.data.data_utils import MixerDataset


@pytest.fixture
def datasets():
    return [["a1", "a2", "a3"], ["b1", "b2", "b3", "b4", "b5", "b6"]]


def test_mixer_dataset(datasets):
    mixer = MixerDataset(datasets)
    assert len(mixer) == 9
    assert [x for x in mixer] == [x for dataset in datasets for x in dataset]


def test_mixer_dataset_with_size_limit(datasets):
    mixer = MixerDataset(datasets, seed=0, max_size=3)
    assert len(mixer) == 6
    assert [x for x in mixer][:3] == ["a1", "a2", "a3"]
    for x in [x for x in mixer][3:]:
        assert x in datasets[1]
