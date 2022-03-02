local t0_mixtures = import 't0_mixtures.jsonnet';
local t0_task_info = import 't0_task_info.jsonnet';

// ------------------------------------ //
// --- Mixture, datasets, and tasks --- //
// ------------------------------------ //

local mixture_name = "d4_dev";

local datasets = std.set([
    t0_task_info["tasks"][task_name]["dataset_name"] for task_name in t0_mixtures[mixture_name]
]);
local tasks = t0_mixtures[mixture_name];

// ----------------------- //
// --- Hyperparameters --- //
// ----------------------- //

local config = {
    "type": "default",
    "seed": 100,
    "gpus": 1,
    "precision": 32,
};
local model_name = "google/t5-small-lm-adapt";
local checkpoint = std.extVar("CKPT");

// Set to null if you don't want to load a checkpoint.
// local checkpoint = "/net/nfs.cirrascale/allennlp/zhaofengw/meta-learn-prompt/output/mtl_small_nooptstate/runs/pumped-kodiak/output_model/work/last.ckpt";
local checkpoint = null;

local model = if checkpoint == null then {
    "type": "prefix_transformer",
    "transformer_model": model_name,
} else {
    "type": "prefix_transformer_from_checkpoint",
    "transformer_model": model_name,
    "checkpoint_path": checkpoint,
    "strict": false,
};

// Function that returns the eval step for a given task.
local EvalStep(task_name) = {
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
        "num_prefix": 0,
        "num_workers": 4,
    },
    "model": model,
};

// Function that returns the name of the eval step for a task.
local EvalStepName(task_name) = "result_" + task_name;

{
    steps: {
        [EvalStepName(task_name)]: EvalStep(task_name) for task_name in tasks
    } + {
        "aggregated_results": {
            type: "aggregate_results",
            results: {
                [task_name]: {type: "ref", ref: EvalStepName(task_name)}
                for task_name in tasks
            }
        }
    }
}
