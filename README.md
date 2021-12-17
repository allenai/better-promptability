# Meta-learning for prompting

## Setup

1. Create a new Python virtual environment with Python 3.7.
2. Install PyTorch 1.9.1 according to the [official instructions](https://pytorch.org/get-started/locally/).
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
