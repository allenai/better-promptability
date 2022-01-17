from collections import defaultdict
from typing import Any, Dict, List, Tuple, Set, Optional

import numpy as np
from tango import Format, JsonFormat, Step
from tango.common import Params
import torch


@Step.register("aggregate_results")
class AggregateResults(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT: Format = JsonFormat()
    VERSION = "002"

    def run(self, results: Dict[str, Tuple[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
        """
        Aggregate the results of a bunch of `TrainStep`s. `results` is a mapping of `task_name`
        the output from the corresponding `TrainStep`.
        """
        t0_task_info = Params.from_file("configs/t0_task_info.jsonnet")["tasks"].as_dict(quiet=True)

        def accuracy_for_task(task_name: str) -> float:
            acc = results[task_name][1][-1]["best_categorical_accuracy"]
            if isinstance(acc, (float, int)):
                return float(acc)
            elif isinstance(acc, torch.Tensor):
                return acc.item()
            else:
                raise TypeError(acc)

        def stats_for_tasks(tasks: Set[str]) -> Dict[str, Optional[float]]:
            accuracies = [accuracy_for_task(task_name) for task_name in tasks]
            return {
                "mean": np.mean(accuracies),
                "std": None if len(accuracies) <= 1 else np.std(accuracies),
            }

        dataset_to_tasks: Dict[str, Set[str]] = defaultdict(set)
        dataset_to_subset_to_tasks: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for task_name in results:
            dataset_name = t0_task_info[task_name]["dataset_name"]
            subset_name = t0_task_info[task_name]["subset_name"]
            dataset_to_tasks[dataset_name].add(task_name)
            dataset_to_subset_to_tasks[dataset_name][subset_name].add(task_name)

        # For direct copying into a spreadsheet
        flattened_results = []
        for dataset_name, subset_to_tasks in dataset_to_subset_to_tasks.items():
            for subset_name in subset_to_tasks:
                stats = stats_for_tasks(subset_to_tasks[subset_name])
                flattened_results.extend([stats["mean"], stats["std"]])

        return {
            "categorical_accuracy_all": stats_for_tasks(set(results.keys())),
            "categorical_accuracy_by_dataset": {
                dataset_name: stats_for_tasks(tasks)
                for dataset_name, tasks in dataset_to_tasks.items()
            },
            "categorical_accuracy_by_dataset_and_subset": {
                dataset_name: {
                    subset_name: stats_for_tasks(subset_to_tasks[subset_name])
                    for subset_name in subset_to_tasks
                }
                for dataset_name, subset_to_tasks in dataset_to_subset_to_tasks.items()
            },
            "flattened": ",".join([str(n * 100) if n is not None else 0 for n in flattened_results])
        }
