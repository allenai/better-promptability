import os

from tango.common import Params


def test_few_shot_baseline_all():
    os.environ["CKPT"] = "null"
    d = Params.from_file("configs/fewshot_eval_all_green.jsonnet").as_dict()
    del os.environ["CKPT"]
    assert "result_anli_GPT_3_style_r1_score_eval" in d["steps"]
    assert "aggregated_results" in d["steps"]
