local config = {
    "type": "default",
    "seed": 100,
    "gpus": null,
    "fp16": false,
};
local model = "google/t5-small-lm-adapt";
local dataset_name = "unittest";
local subset_name = "unittest";
local template_name = "unittest";

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
                "data_dir": "test_fixtures/data/sst2",
                "transformer_model": model,
                "num_prefix": 20,
            },
            "model": {
                "transformer_model": model,
                "optimizer": {
                    "type": "transformers_adamw",
                    "lr": 0.001,
                    "eps": 1e-8,
                },
                "weight_decay": 1e-5,
            }
        }
    }
}
