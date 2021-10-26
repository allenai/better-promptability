import datasets
from tango import Step


@Step.register("load_super_glue")
class LoadSuperGlue(Step):
    """
    A step for loading tasks from [SuperGLUE](https://super.gluebenchmark.com/).
    """

    DETERMINISTIC = True
    CACHEABLE = False  # we'll use 'datasets' caching mechanism, not Tango's.

    DIAGNOSTIC_TASKS = {
        "axb",  # broadcoverage diagnostics
        "axg",  # winogender schema diagnostics
    }

    REGULAR_TASKS = {
        "cb",  # commitment  bank
        "copa",  # choise of plausible alternatives
        "multirc",  # multi-sentence reading comprehension
        "rte",  # recognizing textual entailment
        "wic",  # words in context
        "wsc",  # winograd schema challenge
        "boolq",  # BoolQ
        "record",  # reading comprehension with commonsense reasoning
    }

    def run(self, include_diagnostic_tasks: bool = True) -> datasets.DatasetDict:
        """
        Load all data into a `DatasetDict`. All splits for all tasks (regular and diagnostic)
        are loaded by default, although you can exclude diagnostic tasks by setting
        `include_diagnostic_tasks` to `False`.

        The keys in the `DatasetDict` returned are a concatenation of the (lowercase) task ID
        (e.g. "copa") and the split name (e.g. "train"), with an underscore in between
        (e.g. "copa_train").
        """
        output = datasets.DatasetDict()
        task_names = (
            self.REGULAR_TASKS
            if not include_diagnostic_tasks
            else (self.REGULAR_TASKS | self.DIAGNOSTIC_TASKS)
        )
        for name in sorted(task_names):
            task_data = datasets.load_dataset("super_glue", name=name)
            for split, dataset in task_data.items():
                output[f"{name}_{split}"] = dataset
        print(f"Loaded {len(task_names)} superGLUE tasks")
        return output
