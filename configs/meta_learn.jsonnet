local config = {
    "type": "default",
    "seed": 100,
    "gpus": 1,
    "fp16": false,
};
local model = "google/t5-small-lm-adapt";

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
                "logger": [
                    {"type": "pytorch_lightning::TensorBoardLogger"},
                    {
                        "type": "pytorch_lightning::WandbLogger",
                        "project": "meta-learning",
                        "entity": "meta-learn-prompt",
                    },
                ],
                "callbacks": [
                    {
                        "type": "pytorch_lightning::ModelCheckpoint",
                        "save_last": true,
                        "save_top_k": -1,
                    },
                    "my_logger",
                    "t0_multitask",
                ],
                "replace_sampler_ddp": false,
            },
            "datamodule": {
                "type": "t0_meta_learning",
                "meta_batch_size": 8,
                "mixture_name": "d4_train",
                "data_dir": "data",
                "t0_data_cache": "/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache",
                "transformer_model": model,
                "batch_size": 32,
                "support_batch_size": 16,
                "num_prefix": 20,
                "num_workers": 2,
            },
            "model": {
                "type": "meta_learner",
                "model": {
                    "transformer_model": model,
                    "optimizer": {
                        "type": "adafactor",
                        "lr": 0.001,
                        "scale_parameter": false,
                        "relative_step": false,
                    },
                    "load_opt_states": false,
                },
                "adaptation_steps": 7,  # though in few-shot learning we have only one batch/epoch, but we train for many epochs
                "algorithm": "fomaml",
                "meta_optimizer": {
                    "type": "adafactor",
                    "lr": 0.001,
                    "scale_parameter": false,
                    "relative_step": false,
                },
            }  // "model" (meta_learner)
        }  // "output_model"
    }
}
