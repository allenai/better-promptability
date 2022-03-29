local config = {
    "type": "default",
    "seed": 100,
    "gpus": 1,
    "precision": 32,
};
local model_name = "bigscience/T0_3B";
local mixture_name = "green";
// local mixture_name = "d4_dev";
// local task_name = "race_high_Read_the_article_and_answer_the_question_no_option_";
local num_prefix = 0;
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

// Set to null if you don't want to load a checkpoint.
local checkpoint = null;
// local checkpoint = null;

local model = if checkpoint == null then {
    "type": "prefix_transformer",
    "transformer_model": model_name,
} else {
    "type": "prefix_transformer_from_checkpoint",
    "transformer_model": model_name,
    "checkpoint_path": checkpoint,
    "strict": false,
};

{
    "steps": {
        "output_model": {
            "type": "eval_step",
            "config": config,
            "trainer": {
                "type": "default",
            },
            "datamodule": {
                "type": "t0_multitask",
                "mixture_name": mixture_name,
                "data_dir": "data",
                #"t0_data_cache": "/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache",
                "t0_data_cache": "/net/nfs.cirrascale/allennlp/zhaofengw/t0/data_cache/",
                "transformer_model": model_name,
                "batch_size": 1,
                "eval_batch_size": 64,
                "num_prefix": 0,
                "num_workers": 0,
                "subsample_indices_files": subsample_indices_files,
            },
            "model": model,
        }
    }
}
