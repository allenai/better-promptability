from __future__ import annotations
from typing import Any, List, Mapping, Optional
import numpy as np

from tango.common.aliases import PathOrStr

from pathlib import Path
import pickle

from allennlp.training.metrics import Metric
import datasets
from tango.common import DatasetDict

from .data_utils import md5, PAD_TYPE
from .prompt_data_module import PromptDataModule
from .config import Config


@PromptDataModule.register("t0", exist_ok=True)
class T0Module(PromptDataModule):
    """
    Represents a single dataset AND template, but all the splits.
    """

    def __init__(
        self,
        config: Config,
        num_prefix: int,
        transformer_model: PathOrStr,
        mixture_name: str,
        task_name: str,
        t0_data_cache: PathOrStr = "/net/nfs2.allennlp/petew/meta-learn-prompt/t0/cache",
        sequence_length: Optional[Mapping[str, int]] = None,
        subsample_indices_file: Optional[str] = None,
        **kwargs,
    ):
        from .t0_mixture import read_task_info

        super().__init__(config, num_prefix, transformer_model, **kwargs)

        self.mixture_name = mixture_name
        self.task_name = task_name
        self.dataset_name, self.subset_name, self.template_name = read_task_info()[self.task_name]
        self.t0_data_cache = Path(t0_data_cache)
        self.sequence_length = sequence_length
        self.subsample_indices = None
        if subsample_indices_file is not None:
            self.subsample_indices = pickle.load(open(subsample_indices_file, "rb"))[task_name]

    @property
    def hash_fields(self) -> list[Any]:
        return super().hash_fields + [self.task_name]

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
        if self.subsample_indices is not None:
            indices, checksum = self.subsample_indices
            dataset = self.dataset_dict[self.train_split].select(indices)
            assert md5("".join(str(ex["inputs"] + ex["targets"]) for ex in dataset)) == checksum
            # TODO(petew): this is not blocking, but this is not elegant, and we might reconsider
            # https://github.com/allenai/tango/pull/112
            self.dataset_dict.splits[self.train_split] = dataset

    @property
    def dev_splits(self) -> list[str]:
        # Story Cloze doesn't have a training split, so we use the dev split for training
        if self.dataset_name != "story_cloze":
            for split in ("dev", "validation"):
                if split in self.dataset_dict:
                    return [split]
        return []

    @property
    def test_splits(self) -> list[str]:
        # We don't need the test sets. The test set labels of some datasets are hidden
        # (e.g., superglue), and T0 only evaluated on the dev sets.
        return []

    @property
    def metric_names(self) -> list[str]:
        # For all the green (i.e., d4_score_eval) datasets, all tasks have accuracy as the metric.
        return ["categorical_accuracy"]

    def instantiate_metric(self, metric_name: str, split: str) -> Metric:
        # return t5.evaluation.metrics.rank_classification
        return Metric.by_name(metric_name)()

    @property
    def metric_watch_mode(self) -> str:
        return "max"

    @property
    def sort_key(self) -> str:
        return "inputs"

    def load(self) -> DatasetDict:
        data_path = self.t0_data_cache / self.task_name
        assert data_path.is_dir()

        dataset_dict = datasets.load_from_disk(data_path)

        if self.dataset_name == "story_cloze":
            # Story Cloze doesn't have a training split, so we use the validation split for training
            dataset_dict[self.train_split] = dataset_dict["validation"]
            del dataset_dict["validation"]

        # See comment in test_splits(), above
        del dataset_dict["test"]

        return dataset_dict

    def tokenize(self, example: dict[str, Any], split: str) -> dict[str, Any]:
        inputs = example["inputs"][: self.inputs_max_length]

        # Make sure there are no other EOS in `inputs` and `targets`.
        # The EOS token is really the only special token we are concerned about with T5.
        # T5 has no BOS token. There might be UNK tokens in the inputs though, but that's okay.
        assert self.tokenizer.eos_token_id not in inputs

        single_target: bool = False
        is_correct: Optional[List[bool]] = None
        targets = example["targets"]

        if self.mixture_name == "d4_train":
            single_target = True
        elif self.mixture_name == "d4_dev" and split == self.train_split:
            single_target = True

        # We want to evaluate d4_dev datasets same way as the green ones.
        # Some d4_dev datasets do not have answer_choices at all
        # (eg. "web_questions_get_the_answer" simply wants a knowledge-based answer).
        # We ignore these datasets.

        elif (self.mixture_name == "d4_dev" and split != self.train_split) or (
            self.mixture_name == "green"
            and split != self.train_split
            and self.dataset_name == "story_cloze"
        ):
            single_target = False
            # The format in d4_dev is the same as train (there is no is_correct).
            # To get multiple targets, we need to use "answer_choices", and tokenize them.
            is_correct = [
                choice.strip() == example["targets_pretokenized"].strip()
                for choice in (example["answer_choices"])
            ]
            targets = [
                self.tokenizer(choice, add_special_tokens=False)["input_ids"]
                for choice in example["answer_choices"]
            ]

        elif self.mixture_name == "green" and split == self.train_split:
            single_target = True

            # Actually getting the single target.

            if self.dataset_name != "story_cloze":
                correct_idx = np.argmax(example["is_correct"])
                targets = targets[correct_idx]

        else:  # green dev
            single_target = False
            is_correct = example["is_correct"]

        if single_target:
            targets = targets[:-1][  # exclude EOS in example['targets'] (we add later)
                : self.targets_max_length
            ]
            assert self.tokenizer.eos_token_id not in targets
            input_ids, target_ids, input_mask, target_mask = assemble_prompt(
                inputs, targets, self.tokenizer.eos_token_id, self.task_token_ids
            )
        else:
            input_ids = []
            input_mask = []
            target_mask = []
            target_ids = []

            for target in targets:
                target = target[:-1][  # exclude EOS in example['targets'] (we add later)
                    : self.targets_max_length
                ]
                assert self.tokenizer.eos_token_id not in target

                _input_ids, _target_ids, _input_mask, _target_mask = assemble_prompt(
                    inputs, target, self.tokenizer.eos_token_id, self.task_token_ids
                )
                input_ids.append(_input_ids)
                input_mask.append(_input_mask)
                target_ids.append(_target_ids)
                target_mask.append(_target_mask)

        return_dict = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }

        if not single_target:
            assert is_correct is not None and sum(is_correct) == 1
            return_dict["is_correct"] = is_correct
            return_dict["is_correct_mask"] = [True] * len(is_correct)
        return return_dict

    def pad_token_map(self, split: str) -> Mapping[str, PAD_TYPE]:  # type: ignore
        """
        Specifies the padding for each key. Only keys including in this map will be
        included in the batch.
        """
        pad_token_map_ = {
            "input_ids": 0,
            "input_mask": False,
            "target_ids": 0,
            "target_mask": False,
        }

        if self.mixture_name == "green" and split != self.train_split:
            pad_token_map_["is_correct"] = False
            pad_token_map_["is_correct_mask"] = False
        return pad_token_map_


def assemble_prompt(inputs, targets, eos_token_id, task_token_ids):
    input_ids = task_token_ids + inputs + [eos_token_id]
    target_ids = targets + [eos_token_id]
    input_mask = [True] * len(input_ids)
    target_mask = [True] * len(target_ids)
    return input_ids, target_ids, input_mask, target_mask
