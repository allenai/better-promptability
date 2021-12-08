from meta_learn_prompt.steps.process_dataset import ProcessDataset
from meta_learn_prompt.common.testing import MetaLearnPromptTestCase


class ProcessDatasetTest(MetaLearnPromptTestCase):
    def test_process_dataset(self):
        step = ProcessDataset()
        result = step.run(
            old_data_path=str(
                self.FIXTURES_ROOT / "data" / "hellaswag_complete_first_then_score_eval"
            ),
            new_data_path=str(
                self.FIXTURES_ROOT / "data" / "hellaswag_complete_first_then_score_eval_processed"
            ),
        )

        assert len(result["train"]) == 8
        assert len(result["train"][0]["targets"]) == 4
        assert len(result["train"][0]["targets_pretokenized"]) == 4
        assert len(result["train"][0]["is_correct"]) == 4
