from __future__ import annotations
from typing import Any, Mapping

from tango.common.aliases import PathOrStr
from transformers import T5Tokenizer, PreTrainedTokenizerBase

from .data_utils import PAD_TYPE
from .data_module import DataModule


class PromptDataModule(DataModule):
    def __init__(self, num_prefix: int, transformer_model: PathOrStr, *args, **kwargs):
        self.num_prefix = num_prefix
        self.transformer_model = transformer_model

        self.task_tokens = ["<TASK{}>".format(str(i).zfill(2)) for i in range(self.num_prefix)]

        super().__init__(*args, **kwargs)

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
        targets = example["targets"][:-1][  # exclude EOS in example['targets'] (we add later)
            : self.targets_max_length
        ]

        # Make sure there are no other EOS in `inputs` and `targets`.
        # The EOS token is really the only special token we are concerned about with T5.
        # T5 has no BOS token. There might be UNK tokens in the inputs though, but that's okay.
        assert self.tokenizer.eos_token_id not in inputs
        assert self.tokenizer.eos_token_id not in targets

        input_ids, target_ids, input_mask, target_mask = assemble_prompt(
            inputs, targets, self.tokenizer.eos_token_id, self.task_token_ids
        )

        return_dict = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }
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
        return pad_token_map_


def assemble_prompt(inputs, targets, eos_token_id, task_token_ids):
    input_ids = task_token_ids + inputs + [eos_token_id]
    target_ids = targets + [eos_token_id]
    input_mask = [True] * len(input_ids)
    target_mask = [True] * len(target_ids)
    return input_ids, target_ids, input_mask, target_mask
