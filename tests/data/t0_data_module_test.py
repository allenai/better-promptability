from better_promptability.data.config import Config
from better_promptability.data import T0Module
from better_promptability.common.testing import BetterPromptabilityTestCase


class T0ModuleTest(BetterPromptabilityTestCase):
    def test_t0_module_green(self):
        t0 = T0Module(
            config=Config(),
            data_dir=str(self.FIXTURES_ROOT / "data"),
            num_prefix=1,
            transformer_model="google/t5-small-lm-adapt",
            mixture_name="green",
            task_name="hellaswag_complete_first_then_score_eval",
            t0_data_cache=str(self.FIXTURES_ROOT / "data" / "processed_cache"),
        )

        t0.setup()
        data = t0.load()
        assert "train" in data

        train_batch = list(t0.train_dataloader())[0]
        assert train_batch["target_ids"].dim() == 2

        val_batch = list(t0.val_dataloader()[0])[0]
        assert val_batch["target_ids"].dim() == 3

    def test_t0_module_green_story_cloze(self):

        # Story_cloze special case.

        t0 = T0Module(
            config=Config(),
            data_dir=str(self.FIXTURES_ROOT / "data"),
            num_prefix=1,
            transformer_model="google/t5-small-lm-adapt",
            mixture_name="green",
            task_name="story_cloze_2016_Story_Continuation_and_Options_score_eval",
            t0_data_cache=str(self.FIXTURES_ROOT / "data" / "processed_cache"),
        )

        t0.setup()
        data = t0.load()
        assert "train" in data

        train_batch = list(t0.train_dataloader())[0]
        assert train_batch["target_ids"].dim() == 2

        val_batch = list(t0.val_dataloader()[0])[0]
        assert val_batch["target_ids"].dim() == 3

    def test_t0_module_d4_train(self):
        t0 = T0Module(
            config=Config(),
            data_dir=str(self.FIXTURES_ROOT / "data"),
            num_prefix=1,
            transformer_model="google/t5-small-lm-adapt",
            mixture_name="d4_train",
            task_name="adversarial_qa_dbert_based_on",
            t0_data_cache=str(self.FIXTURES_ROOT / "data" / "cache"),
        )

        t0.setup()
        data = t0.load()
        assert "train" in data

        train_batch = list(t0.train_dataloader())[0]
        assert train_batch["target_ids"].dim() == 2

        val_batch = list(t0.val_dataloader()[0])[0]
        assert val_batch["target_ids"].dim() == 2

    def test_t0_module_d4_dev(self):
        t0 = T0Module(
            config=Config(),
            data_dir=str(self.FIXTURES_ROOT / "data"),
            num_prefix=1,
            transformer_model="google/t5-small-lm-adapt",
            mixture_name="d4_dev",
            task_name="openbookqa_main_choices",
            t0_data_cache=str(self.FIXTURES_ROOT / "data" / "cache"),
        )

        t0.setup()
        data = t0.load()
        assert "train" in data

        train_batch = list(t0.train_dataloader())[0]
        assert train_batch["target_ids"].dim() == 2

        val_batch = list(t0.val_dataloader()[0])[0]
        assert val_batch["target_ids"].dim() == 3
