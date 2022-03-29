local config = {
    "type": "default",
    "seed": 100,
    "gpus": 1,
    "precision": 32,
};
local model_name = "bigscience/T0_3B";
local mixture_name = "mixturename";
local task_name = "taskname";
// local mixture_name = "d4_dev";
// local task_name = "race_high_Read_the_article_and_answer_the_question_no_option_";
local num_prefix = 0;
local subsample_indices_file = "subsampleindicesfile";

// Set to null if you don't want to load a checkpoint.
local checkpoint = "/net/nfs.cirrascale/allennlp/hamishi/meta-learn-prompt/output_t0_3b_dev_inc_dev_cluster_0/cache/TrainStep-004-WPJT9FATUtsf8VVPhdH4PEQS7zRQTMdV/work/subsample_indices_file=0-epoch=0-step=22-endofepoch-categorical_accuracy=0.0000.ckpt";
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
                "type": "t0",
                "mixture_name": mixture_name,
                "task_name": task_name,
                "data_dir": "data",
                "t0_data_cache": "/net/nfs.cirrascale/allennlp/zhaofengw/t0/data_cache",
                "transformer_model": model_name,
                "num_prefix": num_prefix,
                "subsample_indices_file": subsample_indices_file,
                "num_workers": 0,
            },
            "model": model,
        }
    }
}
