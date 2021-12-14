import logging
from typing import Dict
from datasets import Dataset, DatasetDict
from tango.step import Step

logger = logging.getLogger(__name__)


@Step.register("process_dataset")
class ProcessDataset(Step):

    DETERMINISTIC: bool = True
    CACHEABLE = False  # use datasets caching.

    def run(self, old_data_path: str, new_data_path: str) -> DatasetDict:  # type: ignore[override]
        dataset_dict = DatasetDict.load_from_disk(old_data_path)
        new_splits = {}

        for split_name in dataset_dict:
            split = dataset_dict[split_name]

            new_instances: Dict = {
                "inputs": [],
                "inputs_pretokenized": [],
                "targets": [],
                "targets_pretokenized": [],
                "is_correct": [],
            }

            instance: Dict = {
                "inputs": None,
                "inputs_pretokenized": None,
                "targets": [],
                "targets_pretokenized": [],
                "is_correct": [],
            }

            # TODO: assert for presence of the right keys in the dataset.
            for row in split:
                if row["idx"][1] == 0 and instance["inputs"] is not None:
                    new_instances["inputs"].append(instance["inputs"])
                    new_instances["inputs_pretokenized"].append(instance["inputs_pretokenized"])
                    new_instances["targets"].append(instance["targets"])
                    new_instances["targets_pretokenized"].append(instance["targets_pretokenized"])
                    new_instances["is_correct"].append(instance["is_correct"])

                    instance = {
                        "inputs": None,
                        "inputs_pretokenized": None,
                        "targets": [],
                        "targets_pretokenized": [],
                        "is_correct": [],
                    }

                instance["inputs"] = row["inputs"]
                instance["inputs_pretokenized"] = row["inputs_pretokenized"]
                instance["targets"].append(row["targets"])
                instance["targets_pretokenized"].append(row["targets_pretokenized"])
                instance["is_correct"].append(row["is_correct"])

            new_instances["inputs"].append(instance["inputs"])
            new_instances["inputs_pretokenized"].append(instance["inputs_pretokenized"])
            new_instances["targets"].append(instance["targets"])
            new_instances["targets_pretokenized"].append(instance["targets_pretokenized"])
            new_instances["is_correct"].append(instance["is_correct"])

            new_splits[split_name] = Dataset.from_dict(new_instances)

        new_dataset_dict: DatasetDict = DatasetDict(new_splits)
        new_dataset_dict.save_to_disk(new_data_path)
        return new_dataset_dict
