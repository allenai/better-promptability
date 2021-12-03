import pytest
from meta_learn_prompt.data.config import Config
from meta_learn_prompt.data.t0_data_module import T0DataModule
from meta_learn_prompt.models.prefix_transformer import PrefixTransformer

from transformers.models import t5 as hf_t5
from tango.common.params import Params


@pytest.fixture(scope="module")
def model_name():
    return "google/t5-small-lm-adapt"


@pytest.fixture(scope="module")
def tokenizer(model_name):
    return hf_t5.T5Tokenizer.from_pretrained(model_name)


def test_prefix_transformer_forward(model_name, tokenizer):

    dataset = T0DataModule(
        transformer_model=model_name,
        num_prefix=1,
        dataset_name="unittest",
        subset_name="unittest",
        template_name="unittest",
        config=Config(),
        data_dir="test_fixtures/data/unittest",
    )

    conf = {
        "optimizer": {"type": "torch::AdamW", "lr": 0.001, "eps": 1e-08},
        "config": Config(),
        "dataset": dataset,
        "transformer_model": "t5-small",
    }
    prefix_model = PrefixTransformer.from_params(params_=Params(conf))

    input_ids = tokenizer(
        ["The <extra_id_0> walks in <extra_id_1> park", "The <extra_id_0> barked"],
        return_tensors="pt",
        padding=True,
    ).input_ids
    assert input_ids.tolist() == [
        [37, 32099, 10681, 16, 32098, 2447, 1],
        [37, 32099, 1207, 5100, 1, 0, 0],
    ]

    attention_mask = ~(input_ids == 0)

    labels = tokenizer(
        ["<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", "<extra_id_0> dog"],
        return_tensors="pt",
        padding=True,
    ).input_ids
    assert labels.tolist() == [
        [32099, 5295, 1782, 32098, 8, 32097, 1],
        [32099, 1782, 1, 0, 0, 0, 0],
    ]

    decoder_attention_mask = ~(labels == 0)

    prefix_model.forward(
        batch={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": labels,
            "decoder_attention_mask": decoder_attention_mask,
        }
    )
