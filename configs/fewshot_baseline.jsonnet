local config = {
    "type": "default",
    "seed": 100,
    "gpus": 1,
    "fp16": false,
};
local model = "google/t5-small-lm-adapt";
local task_name = "super_glue_wic_affirmation_true_or_false_score_eval";

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
                "log_every_n_steps": 3,
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
                "mixture_name": "green",
                "task_name": task_name,
                "data_dir": "data",
                "t0_data_cache": "/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache",
                "transformer_model": model,
                "num_prefix": 20,
                "subsample_indices_file": "data/green_training_indices_16shot_100seed.pkl",
            },
            "model": {
                "transformer_model": model,
                "optimizer": {
                    "type": "adafactor",
                    "lr": 0.001,
                    "scale_parameter": false,
                    "relative_step": false,
                },
            }
        }
    }
}
