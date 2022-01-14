// This config is for running evaluations on a set of tasks and aggregating the results.
//
// Testing:
// --------
// 
// To do a test run, set `model_name` to a small model like "google/t5-small-lm-adapt"
// and `epochs` to a small number, like 5. Also limit the number of tasks listed in `tasks`.
//
// Then run:
//
// $ tango run configs/fewshot_baseline_all.jsonnet -d /tmp/test-run
//
// You'll be able to see the aggregated results with:
//
// $ cat /tmp/test-run/runs/*/all_results/data.json | jq

// ------------- //
// --- Tasks --- //
// ------------- //

local mixture_name = "green";

// These are all of the tasks you want to evaluate on. They should all be members
// of the mixture specified above by "mixture_name".
// You can find all valid tasks names for a given mixture in the file "data/{mixture_name}_tasks.txt".
local tasks = [
    "anli_GPT_3_style_r1_score_eval",
    "anli_GPT_3_style_r2_score_eval",
    "anli_GPT_3_style_r3_score_eval",
];

// ----------------------- //
// --- Hyperparameters --- //
// ----------------------- //

local config = {
    type: "default",
    seed: 100,
    gpus: 1,
    fp16: false,
};

local epochs = 10;

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

local TrainStepName(task_name) = "result_" + task_name;

{
    steps: {
        [TrainStepName(task_name)]: TrainStep(task_name) for task_name in tasks
    } + {
        all_results: {
           type: "aggregate_results",
           results: [
               {type: "ref", ref: TrainStepName(task_name)} for task_name in tasks
           ],
        },
    }
}
