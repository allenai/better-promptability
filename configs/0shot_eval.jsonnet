local config = {
    "type": "default",
    "seed": 100,
    "gpus": 1,
    "precision": 32,
};
local model_name = "google/t5-small-lm-adapt";
local mixture_name = "green";
local task_name = "hellaswag_Randomized_prompts_template_score_eval";
// local mixture_name = "d4_dev";
// local task_name = "race_high_Read_the_article_and_answer_the_question_no_option_";
local num_prefix = 0;

// Set to null if you don't want to load a checkpoint.
local checkpoint = null;

local model = if checkpoint == "null" then {
    "type": "prefix_transformer",
    "transformer_model": model_name,
} else {
    "type": "prefix_transformer_from_checkpoint",
    "transformer_model": model_name,
    "checkpoint_path": checkpoint,
    "strict": false,
};

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
                "t0_data_cache": "/data/cl/user/zfw/better-promptability/t0_cache",
                "transformer_model": model_name,
                "num_prefix": num_prefix,
                "num_workers": 0,
            },
            "model": model,
        }
    }
}
