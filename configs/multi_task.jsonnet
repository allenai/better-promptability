local config = {
    "type": "default",
    "seed": 100,
    "gpus": 2,
    "precision": 32,
};
local model = "google/t5-small-lm-adapt";
local train_full_model = true;
local effective_batch_size = 4096;
local batch_size = 2;
local ckpt_interval = 512;

{
    "steps": {
        "output_model": {
            "type": "train_step",
            "config": config,
            "trainer": {
                "type": "default",
                "max_epochs": 1,
                "gradient_clip_val": 1.0,
                "accumulate_grad_batches": effective_batch_size / batch_size / config.gpus,
                "num_sanity_val_steps": 0,
                "log_every_n_steps": 50,
                "val_check_interval": ckpt_interval * effective_batch_size / batch_size / config.gpus,
                "logger": [
                    {"type": "pytorch_lightning::TensorBoardLogger"},
                ],
                "callbacks": [
                    # We need separate ModelCheckpoints for per-step and per-epoch checkpointing.
                    # See https://github.com/PyTorchLightning/pytorch-lightning/issues/11645
                    # and https://github.com/PyTorchLightning/pytorch-lightning/issues/11930
                    {
                        "type": "pytorch_lightning::ModelCheckpoint",
                        "save_last": true,
                        "save_top_k": -1,
                        "filename": "{epoch}-{step}-{categorical_accuracy:.4f}",
                        "save_on_train_epoch_end": false,
                    },
                    {
                        "type": "pytorch_lightning::ModelCheckpoint",
                        "save_last": true,
                        "save_top_k": -1,
                        "filename": "{epoch}-{step}-endofepoch-{categorical_accuracy:.4f}",
                        "save_on_train_epoch_end": true,
                    },
                    "my_logger",
                    "t0_multitask",
                ],
                "replace_sampler_ddp": false,
            },
            "model": {
                "type": "prefix_transformer",
                "transformer_model": model,
                "optimizer": {
                    "type": "transformers::adafactor",
                    "lr": 0.001,
                    "scale_parameter": false,
                    "relative_step": false,
                },
                "train_full_model": train_full_model,
            },
            "datamodule": {
                "type": "t0_multitask",
                "mixture_name": "d4_train",
                "data_dir": "data",
                "t0_data_cache": "/data/cl/user/zfw/better-promptability/t0_cache/",
                "transformer_model": model,
                "batch_size": batch_size,
                "eval_batch_size": 64,
                "num_prefix": 20,
                "num_workers": 4,
                "deep": true,
            },
        }
    }
}
