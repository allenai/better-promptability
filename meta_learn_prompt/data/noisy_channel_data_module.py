from typing import Any, Mapping

from tango.common.aliases import PathOrStr
from transformers import GPT2Tokenizer, PreTrainedTokenizerBase

from .data_utils import PAD_TYPE
from .data_module import DataModule
from .templates import get_possible_labels, templatize


LONG_DATASETS = {
    "cr",
    "subj",
    "agnews",
    "amazon",
    "yelp_full",
    "yelp_binary",
    "boolq",
    "dbpedia",
    "yahoo",
}


class NoisyChannelDataModule(DataModule):
    def __init__(
        self,
        num_prefix: int,
        template_idx: int,
        transformer_model: PathOrStr,
        *args,
        **kwargs,
    ):
        self.num_prefix = num_prefix
        self.template_idx = template_idx
        self.transformer_model = transformer_model

        self.task_tokens = ["<TASK{}>".format(str(i).zfill(2)) for i in range(self.num_prefix)]

        super().__init__(*args, **kwargs)

        # TODO
        self.max_length = 256# if self.dataset in LONG_DATASETS else 128

    @property
    def hash_fields(self) -> list[Any]:
        return super().hash_fields + [self.template_idx]

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
        def prepare(label):
            prefix, input = templatize(self.dataset, self.template_idx, example, label)
            prefix = self.tokenizer(prefix)["input_ids"]
            input = self.tokenizer(input)["input_ids"][: self.max_length - 16]
            return assemble_prompt(prefix, input, self.tokenizer.eos_token_id, self.task_token_ids)

        if split == self.train_split:
            input_ids, attention_mask, label_mask, label = prepare(example[self.label_key])
        else:
            input_ids, attention_mask, label_mask, label = [], [], [], []
            for possible_label in get_possible_labels(self.dataset, example):
                _input_ids, _attention_mask, _label_mask, _label = prepare(possible_label)
                input_ids.append(_input_ids)
                attention_mask.append(_attention_mask)
                label_mask.append(_label_mask)
                label.append(_label)

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
