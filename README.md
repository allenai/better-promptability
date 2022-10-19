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

## Environment Setup

1. Create a new Python virtual environment with Python 3.7.
2. Install PyTorch 1.10.1 according to the [official instructions](https://pytorch.org/get-started/locally/).
3. Run

    ```
    pip install -e .[dev]
    ```
    or in zsh
    ```
    pip install -e .\[dev\]
    ```
    Sometimes you might need the flags `--trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org`.

You can verify that your environment is set up properly by running:

```
tango run configs/check_install.yml
```

## Commands

### Intermediate training

`tango run -d output configs/multi_task.jsonnet` and `tango run -d output configs/meta_learn.jsonnet`.

### Prompt Tuning or 0-shot Evaluation

For few-shot learning without intermediate training, run `CKPT=null tango run -d output configs/fewshot_baseline.jsonnet`. To run and aggregate over all datasets/templates, use `CKPT=null tango run -d output configs/fewshot_baseline_{green,d4_dev}.jsonnet`.

If you're not on AI2 NFS, you probably need to pass in the location of the data cache and the pretrained optimizer states with `--overrides "{\"steps.output_model.datamodule.t0_data_cache\": \"${DATA_CACHE_PATH}\"}"`

The set up to run 0-shot evaluation without a soft prompt is very similar, with the config `configs/0shot_eval.jsonnet`.

Change `CKPT` if you want to run these with an existing model checkpoint.

### Evaluating Official T0 checkpoints

T0 was trained w/o EOS (at least so it seems). To accomodate for this, change `t0_module.py`'s' `assemble_prompt()` to not add EOS.
