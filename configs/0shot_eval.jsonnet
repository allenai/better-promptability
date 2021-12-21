local config = {
    "type": "default",
    "seed": 100,
    "gpus": 1,
    "fp16": false,
};
local model = "google/t5-small-lm-adapt";
local mixture_name = "green";
local task_name = "hellaswag_Randomized_prompts_template_score_eval";
// local mixture_name = "d4_dev";
// local task_name = "race_high_Read_the_article_and_answer_the_question_no_option_";

{
    "steps": {
        "output_model": {
            "type": "eval_step",
            "config": config,
            "trainer": {
                "type": "default",
            },
            "datamodule": {
                "type": "t0",
                "mixture_name": mixture_name,
                "task_name": task_name,
                "data_dir": "data",
                "t0_data_cache": "/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache",
                "transformer_model": model,
                "num_prefix": 0,
            },
            "model": {
                "transformer_model": model,
            }
        }
    }
}
