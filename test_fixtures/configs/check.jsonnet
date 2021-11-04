local config = {
    "type": "default",
    "seed": 100,
    "gpus": 1,
    "fp16": false,
};

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
                "dataset": "sst-2",
                "data_dir": "test_fixtures/data/sst2",
                "template_idx": 0,
                "transformer_model": "gpt2",
                "num_prefix": 20,
            },
            "model": {
                "type": "prefix_transformer",
                "transformer_model": "gpt2",
                "optimizer": {
                    "type": "transformers_adamw",
                    "lr": 0.001,
                    "eps": 1e-8,
                },
                "weight_decay": 0.0,
            }
            
        }
    }
}
