import sys

import dill
from tango.common.logging import initialize_logging
from tango.common.util import import_extra_module

from better_promptability.train.train import _train_step


def main():
    initialize_logging()

    _, kwargs_file, results_file = sys.argv
    with open(kwargs_file, "rb") as f:
        training_kwargs = dill.load(f)
    for module in training_kwargs["extra_modules"]:
        import_extra_module(module)
    results = _train_step(
        training_kwargs["work_dir"],
        training_kwargs["config"],
        training_kwargs["trainer"],
        training_kwargs["strategy"],
        training_kwargs["model"],
        datamodule=training_kwargs["datamodule"],
    )
    with open(results_file, "wb") as f:
        dill.dump(results, f)


if __name__ == "__main__":
    main()
