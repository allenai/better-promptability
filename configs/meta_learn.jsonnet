local config = {
    "type": "default",
    "seed": 100,
    "gpus": 8,
    "precision": 32,
};
local model = "google/t5-xl-lm-adapt";

local meta_batch_size = 128;
local ckpt_interval = 64000 / meta_batch_size;

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
                "val_check_interval": ckpt_interval,
                "logger": [
                    {"type": "pytorch_lightning::TensorBoardLogger"},
                    {
                        "type": "pytorch_lightning::WandbLogger",
                        "project": "meta-learning",
                        "entity": "meta-learn-prompt",
                    },
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
                // "t0_data_cache": "/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache",
                "t0_data_cache": "/net/nfs.cirrascale/allennlp/zhaofengw/t0/data_cache/",
                "transformer_model": model,
                "batch_size": 32,
                "support_batch_size": 16,
                "num_prefix": 20,
                "num_workers": 4,
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
                    "load_opt_states": false,
                },
                "adaptation_steps": 7,  # though in few-shot learning we have only one batch/epoch, but we train for many epochs
                "algorithm": "fomaml",
                "meta_optimizer": {
                    "type": "transformers::adafactor",
                    "lr": 0.001,
                    "scale_parameter": false,
                    "relative_step": false,
                },
                "load_opt_states": false,
                "meta_accumulate_grad_batches": 1,
            }  // "model" (meta_learner)
        }  // "output_model"
    }
}
