from __future__ import annotations
from typing import Any, Mapping

from tango.common.aliases import PathOrStr
from transformers import GPT2Tokenizer, PreTrainedTokenizerBase

from .data_utils import PAD_TYPE
from .data_module import DataModule
from .templates import get_possible_labels, templatize


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
    def output_mode(self) -> str:
        return "token_classification"

    def setup_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer = GPT2Tokenizer.from_pretrained(self.transformer_model)
        tokenizer.add_tokens(self.task_tokens)
        task_token_ids = tokenizer(" ".join(self.task_tokens), return_tensors="pt")["input_ids"]
        assert task_token_ids.shape[-1] == self.num_prefix
        self.task_token_ids = task_token_ids.squeeze(0).tolist()
        return tokenizer

    def tokenize(self, example: dict[str, Any], split: str) -> dict[str, Any]:
        # For T0 datasets, they are already tokenized in seqio, but maybe it'd be great to do them
        # again as a sanity check esp. considering differences between tf vs. huggingface tokenizers
        inputs = self.tokenizer(example["inputs_pretokenized"])["input_ids"][
            : self.inputs_max_length
        ]
        # TODO(petew): add_prefix_space should be False for T5
        targets = self.tokenizer(example["target_pretokenized"], add_prefix_space=True)[
            "input_ids"
        ][: self.targets_max_length]
        assert inputs == example["inputs"] and targets == example["targets"]

        input_ids, attention_mask, label_mask, label = assemble_prompt(
            inputs, targets, self.tokenizer.eos_token_id, self.task_token_ids
        )

        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_mask": label_mask,
        }
        if split == self.train_split:
            return_dict["label"] = label
        else:
            return_dict["sequence_label"] = label
            return_dict["label"] = example[self.label_key]
        return return_dict

    def pad_token_map(self, split: str) -> Mapping[str, PAD_TYPE]:  # type: ignore
        """
        Specifies the padding for each key. Only keys including in this map plus the label will be
        included in the batch.
        """
        pad_token_map_ = {"input_ids": 0, "attention_mask": False, "label_mask": False}
        pad_token_map_["label" if split == self.train_split else "sequence_label"] = 0
        return pad_token_map_


def assemble_prompt(prefix, input, eos_token_id, task_token_ids):
    # Why don't we need BOS? I'm not sure -- this is following Min et al. (2021).
    # I think it has something to do with GPT-2 not being trained with it
    # see https://github.com/huggingface/transformers/issues/3311
    input_ids = prefix + input + [eos_token_id]
    label_mask = [False] * len(prefix) + [True] * (len(input) + 1)
    # T: soft task tokens; P: prompt tokens; X: sentence
    # input_ids  : P1 P2 P3 X1 X2 X3 EOS
    # label_mask : 0  0  0  1  1  1  1

    n_task_tokens = len(task_token_ids)
    new_input_ids = task_token_ids + input_ids[:-1]
    labels = [0] * (n_task_tokens - 1) + input_ids
    new_label_mask = [False] * (n_task_tokens - 1) + label_mask
    # new_input_ids  : T1 T2 T3 P1 P2 P3 X1 X2 X3
    # pred           : T2 T3 P1 P2 P3 X1 X2 X3 EOS       <-- this is what the model will predict
    # labels         : 0  0  P1 P2 P3 X1 X2 X3 EOS
    # new_label_mask : 0  0  0  0  0  1  1  1  1

    assert len(new_input_ids) == len(labels) == len(new_label_mask)
    attention_mask = [True] * len(labels)
    return new_input_ids, attention_mask, new_label_mask, labels
