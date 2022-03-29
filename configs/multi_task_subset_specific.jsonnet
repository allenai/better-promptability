local config = {
    "type": "default",
    "seed": 100,
    "gpus": 1,
    "precision": 32,
};
local model = "bigscience/T0_3B";
local train_full_model = true;
local effective_batch_size = 4096;
local batch_size = 2;
local ckpt_interval = 512;
local max_epochs = 10;
local warmup_steps = 500;
local train_steps = 10000000;
local subsample_indices_files = [
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_0_indices.pkl",
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_1_indices.pkl",
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_2_indices.pkl",
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_3_indices.pkl",
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_4_indices.pkl",
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_5_indices.pkl",
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_6_indices.pkl",
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_7_indices.pkl",
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_8_indices.pkl",
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_9_indices.pkl",
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_10_indices.pkl",
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_11_indices.pkl",
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_12_indices.pkl",
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_13_indices.pkl",
    "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/test_cluster_14_indices.pkl"];
local train_subsample_indices_file = "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices_inc_dev/train_cluster_0_indices.pkl";


{
    "steps": {
        "output_model": {
            "type": "train_step",
            "config": config,
            "trainer": {
                "type": "default",
                "max_steps": train_steps,
                "gradient_clip_val": 1.0,
                "accumulate_grad_batches": effective_batch_size / batch_size / config.gpus,
                "num_sanity_val_steps": 0,
                "log_every_n_steps": 50,
                "logger": [
                    {"type": "pytorch_lightning::TensorBoardLogger"},
                    {
                        "type": "pytorch_lightning::WandbLogger",
                        "project": "test",
                        "entity": "hamishivi",
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
                        "filename": "{subsample_indices_file}-{epoch}-{step}-endofepoch-{categorical_accuracy:.4f}",
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
                    "type": "transformers::AdamW",
                    "lr": 0.0001
                    #"type": "deepspeed::cpu_adam",
                    #"type": "deepspeed::fused_adam",
                    #"type": "deepspeed::fused_lamb",
                    #"type": "transformers::adamw",
                },
                "LRScheduler": {
                    "type": "linear_with_warmup",
                    "num_warmup_steps": warmup_steps,
                    "num_training_steps": train_steps
                },
                "train_full_model": train_full_model,
                "load_opt_states": false,
            },
            "datamodule": {
                "type": "t0_multitask",
                "mixture_name": "green",
                "data_dir": "data",
                "t0_data_cache": "/net/nfs.cirrascale/allennlp/zhaofengw/t0/data_cache/",
                "transformer_model": model,
                "batch_size": batch_size,
                "eval_batch_size": 32,
                "num_prefix": 0,
                "subsample_indices_files": subsample_indices_files,
                "train_subsample_indices_file": train_subsample_indices_file,
                "num_workers": 4,
            },
        },
    }
}
