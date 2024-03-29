local config = {
    "type": "default",
    "seed": 100,
    "gpus": 1,
    "fp16": false,
};
local model = "google/t5-small-lm-adapt";
local dataset_name = "story_cloze";
local subset_name = "2016";
local template_name = "Answer_Given_options_score_eval";

{
    "steps": {
        "output_model": {
            "type": "train_step",
            "config": config,
            "trainer": {
                "type": "default",
                "max_epochs": 1,
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
                "dataset_name": dataset_name,
                "subset_name": subset_name,
                "template_name": template_name,
                "subsample_indices_file": "data/green_training_indices_16shot_100seed.pkl",
                "data_dir": "data/" + dataset_name + "_" + subset_name + "_" + template_name,
                "transformer_model": model,
                "num_prefix": 20,
            },
            "model": {
                "transformer_model": model,
                "optimizer": {
                    "type": "adafactor",
                    "lr": 0.001,
                    "scale_parameter": false,
                    "relative_step": false,
                },
                "weight_decay": 1e-5,
            }
        }
    }
}
