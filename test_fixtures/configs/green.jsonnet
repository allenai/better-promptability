local config = {
    "type": "default",
    "seed": 100,
    "gpus": null,
    "fp16": false,
};
local model = "google/t5-small-lm-adapt";
local task_name = "hellaswag_complete_first_then_score_eval";

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
                "mixture_name": "green",
                "task_name": task_name,
                "data_dir": "test_fixtures/data",
                "t0_data_cache": "test_fixtures/data/processed_cache",
                "transformer_model": model,
                "num_prefix": 1,
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
