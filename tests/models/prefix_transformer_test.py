import pytest
import dill

from meta_learn_prompt.common.testing import MetaLearnPromptTestCase
from meta_learn_prompt.data.config import Config
from meta_learn_prompt.data.t0_data_module import T0Mixture, T0DataModule
from meta_learn_prompt.models.prefix_transformer import PrefixTransformer

from tango.common.params import Params


class TestPrefixTransformer(MetaLearnPromptTestCase):
    def test_prefix_transformer_forward(self):

        with open(str(self.FIXTURES_ROOT / "data" / "hellaswag.dill"), "rb") as f:
            unpickler = dill.Unpickler(file=f)
            dataset = unpickler.load()

        assert False

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

        prefix_model.forward(batch={"input_ids": input_ids, "attention_mask": attention_mask})
