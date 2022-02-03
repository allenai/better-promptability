local config = {
    "type": "default",
    "seed": 100,
    "gpus": 4,
    "fp16": false,
};
local model = "google/t5-large-lm-adapt";
local train_full_model = true;
local effective_batch_size = 4096;
local batch_size = 1;

{
    "steps": {
        "output_model": {
            "type": "train_step",
            "config": config,
            "trainer": {
                "type": "default",
                "max_epochs": 100,
                "gradient_clip_val": 1.0,
                "accumulate_grad_batches": effective_batch_size / batch_size,
                "num_sanity_val_steps": 0,
                "log_every_n_steps": 50,
                "logger": [
                    {"type": "pytorch_lightning::TensorBoardLogger"},
#                    {
#                        "type": "pytorch_lightning::WandbLogger",
#                        "project": "multi-task",
#                        "entity": "meta-learn-prompt",
#                    },
                ],
                "callbacks": [
                    # We need separate ModelCheckpoints for per-step and per-epoch checkpointing.
                    # See https://github.com/PyTorchLightning/pytorch-lightning/issues/11645
                    {
                        "type": "pytorch_lightning::ModelCheckpoint",
                        "save_last": true,
                        "save_top_k": -1,
                        "every_n_train_steps": 500,
                    },
                    {
                        "type": "pytorch_lightning::ModelCheckpoint",
                        "save_last": true,
                        "save_top_k": -1,
                        "filename": "{epoch}-{step}-endofepoch",
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
                    "type": "transformers_adamw",
                    "lr": 0.0001,
                },
                "train_full_model": train_full_model,
            },
            "datamodule": {
                "type": "t0_multitask",
                "mixture_name": "debug_train",
                "data_dir": "data",
                "t0_data_cache": "/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache",
                "transformer_model": model,
                "batch_size": batch_size,
                "num_prefix": 20,
                "num_workers": 8,
            },
        }
    }
}
