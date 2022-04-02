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

# local tasks = t0_mixtures[mixture_name];

// For debugging:
local tasks = [
        "super_glue_rte_GPT_3_style_score_eval",
        "super_glue_rte_MNLI_crowdsource_score_eval",
        "super_glue_rte_based_on_the_previous_passage_score_eval",
        "super_glue_rte_can_we_infer_score_eval",
        "super_glue_rte_does_it_follow_that_score_eval",
        "super_glue_rte_does_this_imply_score_eval",
        "super_glue_rte_guaranteed_true_score_eval",
        "super_glue_rte_justified_in_saying_score_eval",
        "super_glue_rte_must_be_true_score_eval",
        "super_glue_rte_should_assume_score_eval",
];

// ----------------------- //
// --- Hyperparameters --- //
// ----------------------- //

local config = {
    type: "default",
    seed: 100,
    gpus: 1,
    precision: 32,
};

local epochs = 3;

local model_name = "google/t5-xl-lm-adapt";

#local checkpoint = "/net/nfs.cirrascale/allennlp/zhaofengw/meta-learn-prompt/output/mtl_xl/runs/fleet-tarpon/output_model/work/epoch=0-step=7704-endofepoch-categorical_accuracy=0.6343.ckpt";
local checkpoint = "null";

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
    train_full_model: true
};

local grad_acc = 4;

// Function that returns the train + eval step for a given task.
local TrainStep(task_name) = {
    type: "train_step",
    config: config,
    trainer: {
        type: "default",
        max_epochs: epochs,
        gradient_clip_val: 1.0,
        accumulate_grad_batches: grad_acc,
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
        batch_size: std.floor(32 / grad_acc),
        #num_prefix: 20,
        #num_prefix: 3,
        num_workers: 1,
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
