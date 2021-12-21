import logging
import os
from typing import Dict
from datasets import Dataset, DatasetDict
from tango.step import Step

from allennlp.common import cached_transformers

logger = logging.getLogger(__name__)


@Step.register("process_story_cloze")
class ProcessStoryCloze(Step):

    DETERMINISTIC: bool = True
    CACHEABLE = False  # use datasets caching.

    def run(
        self,
        old_data_path: str,
        new_data_path: str,
        process_if_exists: bool = False,
        tokenizer_model: str = "google/t5-small-lm-adapt",
    ) -> DatasetDict:  # type: ignore[override]

        if not process_if_exists and os.path.exists(new_data_path):
            logger.info(
                f"The processed dataset already exists at {new_data_path}. "
                "Set `process_if_exists` to `True` if you want to process again. "
                "Returning existing dataset."
            )
            return DatasetDict.load_from_disk(new_data_path)

        tokenizer = cached_transformers.get_tokenizer(tokenizer_model)

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

            for instance in split:
                actual_targets_pretokenized = instance["targets_pretokenized"]

                is_correct = [
                    choice.strip() == actual_targets_pretokenized.strip()
                    for choice in (instance["answer_choices"])
                ]

                targets = [
                    tokenizer(choice, add_special_tokens=False)["input_ids"]
                    for choice in instance["answer_choices"]
                ]

                targets_pretokenized = instance["answer_choices"]

                new_instances["inputs"].append(instance["inputs"])
                new_instances["inputs_pretokenized"].append(instance["inputs_pretokenized"])
                new_instances["targets"].append(targets)
                new_instances["targets_pretokenized"].append(targets_pretokenized)
                new_instances["is_correct"].append(is_correct)

            if split_name == "validation":
                split_name = "train"
            if split_name == "test":
                split_name = "validation"
            new_splits[split_name] = Dataset.from_dict(new_instances)

        new_dataset_dict: DatasetDict = DatasetDict(new_splits)
        logger.info(f"Saving processed dataset at {new_data_path}.")
        new_dataset_dict.save_to_disk(new_data_path)
        return new_dataset_dict
