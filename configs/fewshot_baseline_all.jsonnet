// This config is for running evaluations on a mixture of tasks and aggregating the results
// by dataset and subset.
// The aggregated results for each dataset will be in a step called "aggregated_results_{dataset_name}",
// and the aggregated results for each subset will be in a step called "aggregated_results_{dataset_name}_{subset_name}".
//
// Testing:
// --------
// 
// To do a test run, first modify this config like so:
//  1. Set `model_name` to a small model like "google/t5-small-lm-adapt",
//  2. Set `epochs` to a small number like 5,
//  3. Override `datasets` to only list a couple of datasets in the mixture.
//  4. Override `tasks` to only list one or two tasks for each dataset.
//
// Then run:
//
// $ tango run configs/fewshot_baseline_all.jsonnet -d /tmp/test-run

local t0_mixtures = import 't0_mixtures.jsonnet';
local t0_task_info = import 't0_task_info.jsonnet';

// ------------------------------------ //
// --- Mixture, datasets, and tasks --- //
// ------------------------------------ //

local mixture_name = "green";

local datasets = std.set([
    t0_task_info["tasks"][task_name]["dataset_name"] for task_name in t0_mixtures[mixture_name]
]);
local tasks = t0_mixtures[mixture_name];
assert std.count(datasets, "anli") == 1;  // confidence check
assert std.count(tasks, "anli_GPT_3_style_r1_score_eval") == 1; // confidence check

// For debugging:
// local datasets = ["anli", "hellaswag"];
// local tasks = [
//     "anli_GPT_3_style_r1_score_eval",
//     "anli_GPT_3_style_r2_score_eval",
//     "hellaswag_Predict_ending_with_hint_score_eval",
//     "hellaswag_Randomized_prompts_template_score_eval",
// ];

// ----------------------- //
// --- Hyperparameters --- //
// ----------------------- //

local config = {
    type: "default",
    seed: 100,
    gpus: 1,
    fp16: false,
};

local epochs = 100;

local model_name = "google/t5-small-lm-adapt";

local checkpoint = null;
// local checkpoint = "/net/nfs.cirrascale/allennlp/zhaofengw/meta-learn-prompt/output/mtl_small_nooptstate/runs/pumped-kodiak/output_model/work/last.ckpt";

local optimizer = {
    type: "adafactor",
    lr: 0.001,
    scale_parameter: false,
    relative_step: false,
};

// Set to "true" to enable validation after every training epoch, otherwise we only validate
// after the final epoch.
local validate_every_epoch = false;

// ------------------------------------------------------------ //
// --- Data cache - edit according to the machine you're on --- //
// ------------------------------------------------------------ //

// Cirrascale machines:
local t0_data_cache = "/net/nfs.cirrascale/allennlp/zhaofengw/t0/data_cache";
local optstates_dir = "/net/nfs.cirrascale/allennlp/zhaofengw/optstates";

// ----------------------------------------------------------- //
// --- ! You probably don't need to edit below this line ! --- //
// ----------------------------------------------------------- //

local model = {
    [if checkpoint == null then null else "type"]: "from_checkpoint",
    [if checkpoint == null then null else "checkpoint_path"]: checkpoint,
    transformer_model: model_name,
    optimizer: optimizer,
    optstates_dir: optstates_dir,
};

// Function that returns the train + eval step for a given task.
local TrainStep(task_name) = {
    type: "train_step",
    config: config,
    trainer: {
        type: "default",
        max_epochs: epochs,
        gradient_clip_val: 1.0,
        accumulate_grad_batches: 1.0,
        log_every_n_steps: 50,
        logger: [
            {type: "pytorch_lightning::TensorBoardLogger"},
        ],
        callbacks: [
            "pytorch_lightning::ModelCheckpoint",
            "my_logger",
        ],
        replace_sampler_ddp: false,
        check_val_every_n_epoch: if validate_every_epoch then 1 else epochs,
    },
    datamodule: {
        type: "t0",
        mixture_name: mixture_name,
        task_name: task_name,
        data_dir: "data",
        t0_data_cache: t0_data_cache,
        transformer_model: model_name,
        num_prefix: 20,
        subsample_indices_file: "data/" + mixture_name + "_training_indices_16shot_100seed.pkl",
        num_workers: 4,
    },
    model: model,
};

// Function that returns the name of the train+eval step for a task.
local TrainStepName(task_name) = "result_" + task_name;

// Function that checks if a task comes from the given dataset.
local TaskInDataset(task_name, dataset_name) = t0_task_info["tasks"][task_name]["dataset_name"] == dataset_name;

// Function that checks if a task comes from the given subset of a dataset.
local TaskInSubset(task_name, dataset_name, subset_name) = TaskInDataset(task_name, dataset_name) && t0_task_info["tasks"][task_name]["subset_name"] == subset_name;

// Function that returns an array of tasks for a given dataset.
local TasksForDataset(dataset_name) = std.filter(function(task) TaskInDataset(task, dataset_name), tasks);

// Function that returns an array of tasks for a given subset of a dataset.
local TasksForSubset(dataset_name, subset_name) = std.filter(function(task) TaskInSubset(task, dataset_name, subset_name), tasks);

// Function that returns the aggregation step for a given dataset.
local AggregationByDatasetStep(dataset_name) = {
    type: "aggregate_results",
    results: [
        {type: "ref", ref: TrainStepName(task_name)} for task_name in TasksForDataset(dataset_name)
    ],
};

// Function that returns the aggregation step for a given subset of a dataset.
local AggregationBySubsetStep(dataset_name, subset_name) = {
    type: "aggregate_results",
    results: [
        {type: "ref", ref: TrainStepName(task_name)} for task_name in TasksForSubset(dataset_name, subset_name)
    ],
};

// Function that returns the name of the aggregation step for a dataset.
local AggregationByDatasetStepName(dataset_name) = "aggregated_results_" + dataset_name;

// Function that returns the name of the aggregation step for a subset of a dataset.
local AggregationBySubsetStepName(dataset_name, subset_name) = "aggregated_results_" + dataset_name + "_" + subset_name;

// Function to get all of the subsets for a dataset.
local SubsetsForDataset(dataset_name) = std.set([
    t0_task_info["tasks"][task_name]["subset_name"]
    for task_name in tasks
    if t0_task_info["tasks"][task_name]["dataset_name"] == dataset_name && t0_task_info["tasks"][task_name]["subset_name"] != null
]);

local subsets = std.flatMap(
    function(dataset_name) [{"dataset_name": dataset_name, "subset_name": subset_name} for subset_name in SubsetsForDataset(dataset_name)],
    datasets
);
assert std.count(subsets, {"dataset_name": "anli", "subset_name": "r1"}) == 1;  // confidence check

{
    subsets: subsets,
    steps: {
        [TrainStepName(task_name)]: TrainStep(task_name) for task_name in tasks
    } + {
        [AggregationByDatasetStepName(dataset_name)]: AggregationByDatasetStep(dataset_name) for dataset_name in datasets
    } + {
        [AggregationBySubsetStepName(x["dataset_name"], x["subset_name"])]: AggregationBySubsetStep(x["dataset_name"], x["subset_name"]) for x in subsets
    }
}
