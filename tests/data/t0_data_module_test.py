from meta_learn_prompt.data import T0Mixture


@pytest.mark.parametrize(
    "dataset_name",
    [
        "hellaswag",
    ],
)
def test_t0_mixture(dataset_name: str):
    mix = T0Mixture(dataset_name=dataset_name)

    assert len(mix.data_modules) == 11
