from better_promptability.steps.process_dataset import ProcessDataset
from better_promptability.common.testing import BetterPromptabilityTestCase


class ProcessDatasetTest(BetterPromptabilityTestCase):
    def test_process_dataset(self):
        step = ProcessDataset()
        result = step.run(
            old_data_path=str(
                self.FIXTURES_ROOT / "data" / "cache" / "hellaswag_complete_first_then_score_eval"
            ),
            new_data_path=str(
                self.FIXTURES_ROOT
                / "data"
                / "processed_cache"
                / "hellaswag_complete_first_then_score_eval"
            ),
            process_if_exists=True,
        )

        assert len(result["train"]) == 7
        assert len(result["train"][0]["targets"]) == 4
        assert len(result["train"][0]["targets_pretokenized"]) == 4
        assert len(result["train"][0]["is_correct"]) == 4
