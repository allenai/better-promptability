from meta_learn_prompt.data import T0Mixture, T0DataModule
from meta_learn_prompt.data.config import Config


# def test_t0_mixture():
#     dataset_name = "story_cloze";
#     subset_name = "2016";
#     template_name = "Answer_Given_options_score_eval"
#     mix = T0Mixture(dataset_name=dataset_name, subset_name=subset_name, template_name=template_name)

#     print(mix.data_modules)
#     assert len(mix.data_modules) == 11


def test_t0_data_module():
    dataset_name = "unittest"
    subset_name = "unittest"
    template_name = "unittest"

    t0 = T0DataModule(
        transformer_model="google/t5-small-lm-adapt",
        num_prefix=1,
        dataset_name=dataset_name,
        subset_name=subset_name,
        template_name=template_name,
        config=Config(),
        data_dir="test_fixtures/data/sst2"
    )
    print(t0.__dict__)
    assert False
