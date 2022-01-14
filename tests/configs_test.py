from tango.common import Params


def test_few_shot_baseline_all():
    d = Params.from_file("configs/fewshot_baseline_all.jsonnet").as_dict()
    assert "result_anli_GPT_3_style_r1_score_eval" in d["steps"]
    assert "aggregated_results" in d["steps"]
