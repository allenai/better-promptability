# Meta-learning for prompting

## Setup

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

For few-shot learning without intermediate training, run `tango run -d output configs/fewshot_baseline.jsonnet`.
Or to run things in a loop:
```bash
for name in $(cat data/d4_dev_tasks.txt); do
    echo $name
    tango run -d output/${name} configs/fewshot_baseline.jsonnet --overrides "{\"steps.output_model.datamodule.task_name\": \"${name}\"}"
done
```
If you're not on AI2 NFS, you probably need to pass in the location of the data cache and the pretrained optimizer states with `--overrides "{\"steps.output_model.datamodule.t0_data_cache\": \"${DATA_CACHE_PATH}\", \"steps.output_model.model.optstates_dir\": \"${OPT_STATES_PATH}\"}"`
