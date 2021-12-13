from __future__ import annotations
import numpy as np
from typing import Any, Mapping

from tango.common.aliases import PathOrStr
from transformers import T5Tokenizer, PreTrainedTokenizerBase

from .data_utils import PAD_TYPE
from .data_module import DataModule
from .config import Config


class PromptDataModule(DataModule):
    def __init__(
        self,
        config: Config,
        data_dir: PathOrStr,
        num_prefix: int,
        transformer_model: PathOrStr,
        mixture_name: str,
        *args,
        **kwargs,
    ):
        self.num_prefix = num_prefix
        self.transformer_model = transformer_model
        self.mixture_name = mixture_name

        self.task_tokens = ["<TASK{}>".format(str(i).zfill(2)) for i in range(self.num_prefix)]

        super().__init__(config, data_dir, *args, **kwargs)

        # Following T0 paper
        self.inputs_max_length = 1024
        self.targets_max_length = 256

    @property
    def hash_fields(self) -> list[Any]:
        return super().hash_fields + [
            self.num_prefix,
            self.inputs_max_length,
            self.targets_max_length,
        ]

    def setup_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer = T5Tokenizer.from_pretrained(self.transformer_model)
        tokenizer.add_tokens(self.task_tokens)
        task_token_ids = tokenizer(
            " ".join(self.task_tokens), return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        assert task_token_ids.shape[-1] == self.num_prefix
        self.task_token_ids = task_token_ids.squeeze(0).tolist()
        return tokenizer

    def tokenize(self, example: dict[str, Any], split: str) -> dict[str, Any]:
        inputs = example["inputs"][: self.inputs_max_length]

        # Make sure there are no other EOS in `inputs` and `targets`.
        # The EOS token is really the only special token we are concerned about with T5.
        # T5 has no BOS token. There might be UNK tokens in the inputs though, but that's okay.
        assert self.tokenizer.eos_token_id not in inputs

        single_target: bool = False
        # is_correct: Optional[List[bool]] = None
        targets = example["targets"]

        if self.mixture_name == "d4_train":
            single_target = True
        elif self.mixture_name == "d4_dev":  # and split == self.train_split:
            single_target = True

        # # This is what we would do if we wanted to evaluate d4_dev datasets
        # # the same way as the green ones. But some d4_dev datasets do not have answer_choices
        # # at all (eg. "web_questions_get_the_answer" simply wants a knowledge-based answer).
        # elif self.mixture_name == "d4_dev" and split != self.train_split:
        #    single_target = False
        #    # The format in d4_dev is the same as train (there is no is_correct).
        #    # To get multiple targets, we need to use "answer_choices", and tokenize them.
        #    answer_choices = example["answer_choices"]
        #    is_correct = [choice == example["targets"] for choice in (answer_choices)]
        #    targets = [self.tokenizer(choice, add_special_tokens=False)["input_ids"] for choice in answer_choices]

        elif self.mixture_name == "green" and split == self.train_split:
            single_target = True

            # Actually getting the single target.
            correct_idx = np.argmax(example["is_correct"])
            targets = targets[correct_idx]

        else:  # green dev/test
            single_target = False

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

        if self.mixture_name == "green" and split != self.train_split:
            return_dict["is_correct"] = example["is_correct"]
        return return_dict

    def pad_token_map(self, split: str) -> Mapping[str, PAD_TYPE]:  # type: ignore
        """
        Specifies the padding for each key. Only keys including in this map will be
        included in the batch.
        """
        pad_token_map_ = {
            "input_ids": 0,
            "input_mask": False,
            "target_ids": -100,
            "target_mask": False,
        }

        if self.mixture_name == "green" and split != self.train_split:
            pad_token_map_["is_correct"] = 0
        return pad_token_map_


def assemble_prompt(inputs, targets, eos_token_id, task_token_ids):
    input_ids = task_token_ids + inputs + [eos_token_id]
    target_ids = targets + [eos_token_id]
    input_mask = [True] * len(input_ids)
    target_mask = [True] * len(target_ids)
    return input_ids, target_ids, input_mask, target_mask
