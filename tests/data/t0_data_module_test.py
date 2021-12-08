from meta_learn_prompt.data.config import Config
from meta_learn_prompt.data import T0DataModule
from meta_learn_prompt.common.testing import MetaLearnPromptTestCase


class T0DataModuleTest(MetaLearnPromptTestCase):
    def test_t0_data_module(self):
        t0 = T0DataModule(
            config=Config(),
            data_dir=str(self.FIXTURES_ROOT / "data"),
            num_prefix=1,
            transformer_model="google/t5-small-lm-adapt",
            task_name="hellaswag_complete_first_then_score_eval",
            dataset_name="hellaswag",
            subset_name=None,
            template_name="complete_first_then_score_eval",
            t0_data_cache=str(self.FIXTURES_ROOT / "data" / "processed_cache"),
        )

        t0.setup()
        data = t0.load()
        assert "train" in data
