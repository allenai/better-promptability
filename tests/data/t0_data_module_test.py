from meta_learn_prompt.data.config import Config
from meta_learn_prompt.data import T0Module
from meta_learn_prompt.common.testing import MetaLearnPromptTestCase


class T0ModuleTest(MetaLearnPromptTestCase):
    def test_t0_module_green(self):
        t0 = T0Module(
            config=Config(),
            data_dir=str(self.FIXTURES_ROOT / "data"),
            num_prefix=1,
            transformer_model="google/t5-small-lm-adapt",
            mixture_name="green",
            task_name="hellaswag_complete_first_then_score_eval",
            dataset_name="hellaswag",
            subset_name=None,
            template_name="complete_first_then_score_eval",
            t0_data_cache=str(self.FIXTURES_ROOT / "data" / "processed_cache"),
        )

        t0.setup()
        data = t0.load()
        assert "train" in data

        train_batch = list(t0.train_dataloader())[0]
        assert train_batch["target_ids"].dim() == 2

        val_batch = list(t0.val_dataloader()[0])[0]
        assert val_batch["target_ids"].dim() == 3

        test_batch = list(t0.test_dataloader()[0])[0]
        assert test_batch["target_ids"].dim() == 3

    def test_t0_module_d4_train(self):
        t0 = T0Module(
            config=Config(),
            data_dir=str(self.FIXTURES_ROOT / "data"),
            num_prefix=1,
            transformer_model="google/t5-small-lm-adapt",
            mixture_name="d4_train",
            task_name="adversarial_qa_dbert_based_on",
            dataset_name="adversarial_qa",
            subset_name="dbert",
            template_name="based_on",
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
            dataset_name="openbookqa",
            subset_name="main",
            template_name="choices",
            t0_data_cache=str(self.FIXTURES_ROOT / "data" / "cache"),
        )

        t0.setup()
        data = t0.load()
        assert "train" in data

        train_batch = list(t0.train_dataloader())[0]
        assert train_batch["target_ids"].dim() == 2

        val_batch = list(t0.val_dataloader()[0])[0]
        assert val_batch["target_ids"].dim() == 3
