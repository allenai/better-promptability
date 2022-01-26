local config = {
    "type": "default",
    "seed": 100,
    "gpus": 1,
    "fp16": false,
};
local model_name = "google/t5-small-lm-adapt";
local mixture_name = "green";
local task_name = "hellaswag_Randomized_prompts_template_score_eval";
// local mixture_name = "d4_dev";
// local task_name = "race_high_Read_the_article_and_answer_the_question_no_option_";
local subsample_indices_file = "data/" + mixture_name + "_training_indices_16shot_100seed.pkl";
local optstates_dir = "/net/nfs2.allennlp/zhaofengw/optstates";

// Set to null if you don't want to load a checkpoint.
local checkpoint = "/net/nfs2.allennlp/zhaofengw/meta-learn-prompt/unitout4/runs/sacred-alpaca/output_model/work/last.ckpt";
// local checkpoint = null;

local optimizer = {
    "type": "adafactor",
    "lr": 0.001,
    "scale_parameter": false,
    "relative_step": false,
};

local model = if checkpoint == null then {
    "type": "prefix_transformer",
    "transformer_model": model_name,
    "optimizer": optimizer,
    "optstates_dir": optstates_dir,
} else {
    "type": "prefix_transformer_from_checkpoint",
    "transformer_model": model_name,
    "optimizer": optimizer,
    "checkpoint_path": checkpoint,
    "optstates_dir": optstates_dir,
    "strict": false,
};

{
    "steps": {
        "output_model": {
            "type": "train_step",
            "config": config,
            "trainer": {
                "type": "default",
                "max_epochs": 100,
                "gradient_clip_val": 1.0,
                "accumulate_grad_batches": 1.0,
                "log_every_n_steps": 50,
                "logger": [
                    {"type": "pytorch_lightning::TensorBoardLogger"},
                ],
                "callbacks": [
                    "pytorch_lightning::ModelCheckpoint",
                    "my_logger",
                ],
                "replace_sampler_ddp": false,
            },
            "datamodule": {
                "type": "t0",
                "mixture_name": mixture_name,
                "task_name": task_name,
                "data_dir": "data",
                "t0_data_cache": "/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache",
                "transformer_model": model_name,
                "num_prefix": 20,
                "subsample_indices_file": subsample_indices_file,
                "num_workers": 4,
            },
            "model": model,
        }
    }
}
