# Continued Pretraining for Better Zero- and Few-Shot Promptability

The official implementation for our paper:

```bibtex
@inproceedings{wu-etal-2022-continued,
    title = "Continued Pretraining for Better Zero- and Few-Shot Promptability",
    author = "Zhaofeng Wu and Robert L. Logan IV and Pete Walsh and Akshita Bhagia and Dirk Groeneveld and Sameer Singh and Iz Beltagy",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

We provide a somewhat cleaned version of the codebase in the `main` branch. If you run into any issue, you can check out the `archive` branch for the original version that we used. For historical reasons, this repository is slightly over-engineered. For example, because eventually we just performed continued pretraining for one epoch, a lot of checkpointing related logic is unused.

## Pretrained Models

We release our pretrained models at https://huggingface.co/ZhaofengWu/better-promptability.

## Environment Setup

1. Create a new Python virtual environment with Python 3.7.
2. Install PyTorch 1.10.1 according to the [official instructions](https://pytorch.org/get-started/locally/). You may need to install `torchvision==0.11.2` this way too.
3. Run

    ```
    pip install -e .
    ```
    Sometimes you might need the flags `--trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org`.

You can verify that your environment is set up properly by running:

```
tango run configs/check_install.yml
```

## Data Preparation

We use the P3 datasets for training and evaluation. In our codebase, we refer to the training datasets as `d4_train`, following the naming convention in their codebase, and the evaluation datasets as `green`, because they are colored green in the T0 paper. You may also see mentions of `d4_dev`, which is a set of datasets (mutually exclusive with `d4_train` and `green`) that we used for development.

Most of these datasets are publicly available, with the exception of Story Cloze, which we separately obtained from BigScience. You could try doing the same, or processing the data yourself from the original source. The processed Story Cloze data should be in a directory with folders `story_cloze_2016_{Answer_Given_options,Choose_Story_Ending,Movie_What_Happens_Next,Novel_Correct_Ending,Story_Continuation_and_Options}_score_eval`, each one with files

```
COMPLETED  info.test.json  info.validation.json  stats.test.json  stats.validation.json  test.tfrecord-00000-of-00001  validation.tfrecord-00000-of-00001
```

You should update the `STORY_CLOZE_PATH` variable in `scripts/download_t0_training_set.py` to point to this directory. Then to download and process the rest of the datasets, you can run the following commands. Depending on your network speed, etc., they could take a few days (~2 days on our machine).

```bash
mkdir t0_cache unprocessed_green_cache
python scripts/download_t0_training_set.py d4_train t0_cache
python scripts/download_t0_training_set.py green unprocessed_green_cache
python scripts/process_green_datasets.py unprocessed_green_cache t0_cache
```

## Training

All existing configs use T5-small for illustration. You might want to replace it with other sized T5 models.

### Continued Pretraining

Change the value of `"t0_data_cache"` in each config to the path to the `t0_cache` directory above. Then you can run multi-task training or meta-learning with one of the following commands. When run for the first time, these commands may take a few hours for further dataset processing.

```bash
tango run -d ${continued_pretrained_model} configs/multi_task.jsonnet
tango run -d ${continued_pretrained_model} configs/fomaml.jsonnet
tango run -d ${continued_pretrained_model} configs/reptile.jsonnet
```

For multi-task training, you can change the flags `"train_full_model"`, `"num_prefix"`, and `"deep"`, to reproduce our various configurations in the paper. By default, the config file reproduces our best model that trains all components, with a deep prompt. Feel free to change the other flags too -- in particular, you probably want to change the number of GPUs used. These scripts support distributed training. Note that the tqdm estimates of these scripts are over-estimations in the beginning. Wait for at least >10% or so for a more accurate estimate.

### 0-shot/few-shot Evaluation

For 0-shot/few-shot evaluation, you can run (remember to set the `"t0_data_cache"` path, like above):

```bash
CKPT=${checkpoint_path} tango run -d ${output_dir} configs/0shot_eval_all_green.jsonnet  # or configs/fewshot_eval_all_green.jsonnet
```

where `${checkpoint_path}` is the checkpoint you want to evaluate in `${continued_pretrained_model}`. It should look something like `${continued_pretrained_model}/cache/TrainStep-*/work/epoch=0-step=*-endofepoch-categorical_accuracy=*.ckpt`. Set `CKPT=null` if you want to evaluate the model without any continued pretraining.

You need to set the flags `"model_name"`, `"num_prefix"`, and `"deep"` to match the values used during continued pretraining. For example, for the model `mtl_large_deep`, you want `"model_name" = "google/t5-large-lm-adapt"`, `"num_prefix" = 20`, and `"deep" = true`.

`configs/0shot_eval.jsonnet` and `configs/fewshot_eval.jsonnet` evaluate a speicific dataset instead of aggregating over all datasets.

These configs don't directly print out the ARG. To compute that, you can print out the per-dataset accuracy using something like `for d in $(ls -d ${output_dir}/runs/*/result_* | sort); do cat $d/data.json | python -c "import sys, json; print(json.load(sys.stdin)[1][-1]['best_categorical_accuracy'], end='')"; echo -n ","; done`, and then paste the resulting string into `boostrap.py` for the ARG. `bootstrap.py` is also used for significance testing.

### Evaluating Official T0 Checkpoints

T0 was trained without EOS (at least so it seems). To accomodate for this, change `t0_module.py`'s' `assemble_prompt()` to not add EOS (in addition to, of course, changing the `"model_name"` to T0 in the relevant config).
