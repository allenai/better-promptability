from better_promptability.steps.process_story_cloze import ProcessStoryCloze
from better_promptability.common.testing import MetaLearnPromptTestCase


class ProcessStoryClozeTest(MetaLearnPromptTestCase):
    def test_process_story_cloze(self):
        step = ProcessStoryCloze()
        result = step.run(
            old_data_path=str(
                self.FIXTURES_ROOT
                / "data"
                / "cache"
                / "story_cloze_2016_Story_Continuation_and_Options_score_eval"
            ),
            new_data_path=str(
                self.FIXTURES_ROOT
                / "data"
                / "processed_cache"
                / "story_cloze_2016_Story_Continuation_and_Options_score_eval"
            ),
            process_if_exists=True,
        )

        assert len(result["train"]) == 28

        assert len(result["train"][0]["targets"]) == 2
        assert len(result["train"][0]["targets_pretokenized"]) == 2
        assert len(result["train"][0]["is_correct"]) == 2

        assert "validation" in result
        assert "test" not in result
