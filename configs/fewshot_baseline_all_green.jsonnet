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
// $ tango run configs/fewshot_baseline_all_green.jsonnet -d /tmp/test-run

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
    precision: 32,
};

local epochs = 100;

local model_name = "google/t5-small-lm-adapt";

local checkpoint = std.extVar("CKPT");

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
local t0_data_cache = "/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache";
local optstates_dir = "/net/nfs2.allennlp/zhaofengw/optstates";

// ----------------------------------------------------------- //
// --- ! You probably don't need to edit below this line ! --- //
// ----------------------------------------------------------- //

local model = {
    "type": if checkpoint == "null" then "prefix_transformer" else "prefix_transformer_from_checkpoint",
    [if checkpoint == "null" then null else "checkpoint_path"]: checkpoint,
    transformer_model: model_name,
    optimizer: optimizer,
    optstates_dir: optstates_dir,
    load_opt_states: false,
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
            "my_logger",
        ],
        enable_checkpointing: false,
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

{
    steps: {
        [TrainStepName(task_name)]: TrainStep(task_name) for task_name in tasks
    } + {
        "aggregated_results": {
            type: "aggregate_results",
            results: {
                [task_name]: {type: "ref", ref: TrainStepName(task_name)}
                for task_name in tasks
            }
        }
    }
}
