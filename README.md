# Meta-learning for prompting

## Setup

1. Create a new Python virtual environment with Python 3.9 or greater.
2. Install PyTorch according to the [official instructions](https://pytorch.org/get-started/locally/).
3. Run

    ```
    pip install -e .[dev]
    ```

You can verify that your environment is set up properly by running:

```
tango --no-logging run configs/check_install.yml -i meta_learn_prompt
```
