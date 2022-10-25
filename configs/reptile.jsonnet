local config = {
    "type": "default",
    "seed": 100,
    "gpus": 2,
    "precision": 32,
};
local model = "google/t5-small-lm-adapt";

local meta_batch_size = 128;
local adaptation_steps = 7;
local ckpt_interval = 65536 / meta_batch_size;

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
                "num_sanity_val_steps": 0,
                "log_every_n_steps": 6,
                "val_check_interval": ckpt_interval / config.gpus,
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
            "datamodule": {
                "type": "t0_meta_learning",
                "meta_batch_size": meta_batch_size,
                "mixture_name": "d4_train",
                "data_dir": "data",
                "t0_data_cache": "/data/cl/user/zfw/better-promptability/t0_cache/",
                "transformer_model": model,
                "batch_size": 16 * (adaptation_steps + 1),  # this is the effective batch size; ONLY change meta_accumulate_grad_batches when adjusting for GPU sizes
                "support_batch_size": 16 * adaptation_steps,  # ditto
                "eval_batch_size": 64,
                "num_prefix": 20,
                "num_workers": 4,
                "deep": true,
            },
            "model": {
                "type": "meta_learner",
                "model": {
                    "transformer_model": model,
                    "optimizer": {
                        "type": "transformers::adafactor",
                        "lr": 0.001,
                        "scale_parameter": false,
                        "relative_step": false,
                    },
                },
                "adaptation_steps": adaptation_steps,
                "algorithm": "reptile",
                "different_inner_loop_batches": true,
                "meta_optimizer": {
                    "type": "transformers::adafactor",
                    "lr": 0.001,
                    "scale_parameter": false,
                    "relative_step": false,
                },
                "meta_accumulate_grad_batches": 16,
            }  // "model" (meta_learner)
        }  // "output_model"
    }
}
