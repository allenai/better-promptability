local config = {
    "type": "default",
    "seed": 100,
    "gpus": 1,
    "fp16": false,
};
local model = "google/t5-small-lm-adapt";
local train_full_model = true;

{
    "steps": {
        "trained_model": {
            "type": "train_step",
            "config": config,
            "trainer": {
                "type": "default",
                "max_epochs": 10,
                "gradient_clip_val": 1.0,
                "accumulate_grad_batches": 1.0,
                "log_every_n_steps": 3,
                "logger": [
                    {
                        "type": "pytorch_lightning::WandbLogger",
                        "project": "meta-learn-prompt-multi-task",
                        "entity": "allennlp",
                    },
                ],
                "callbacks": [
                    "pytorch_lightning::ModelCheckpoint",
                    "my_logger",
                    "t0_multitask",
                ],
                "replace_sampler_ddp": false,
            },
            "model": {
                "transformer_model": model,
                "optimizer": {
                    "type": "adafactor",
                    "lr": 0.001,
                    "scale_parameter": false,
                    "relative_step": false,
                },
                "train_full_model": train_full_model,
            },
            "datamodule": {
                "type": "t0_multitask",
                "mixture_name": "d4_train",
                "data_dir": "/net/nfs2.allennlp/petew/meta-learn-prompt/cache",
                "t0_data_cache": "/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache",
                "transformer_model": model,
                "batch_size": 32,
                "num_prefix": 20,
            },
        }
    }
}
