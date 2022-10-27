from __future__ import annotations
from typing import Any, Mapping
from urllib.error import HTTPError

from tango.common.aliases import PathOrStr
from transformers import T5Tokenizer, PreTrainedTokenizerBase

from .data_utils import PAD_TYPE
from .data_module import DataModule
from .config import Config


class PromptDataModule(DataModule):
    def __init__(
        self,
        config: Config,
        num_prefix: int,
        transformer_model: PathOrStr,
        deep: bool = False,
        **kwargs,
    ):
        self.num_prefix = num_prefix
        self.transformer_model = transformer_model
        self.deep = deep

        if not self.deep:
            self.task_tokens = ["<TASK{}>".format(str(i).zfill(2)) for i in range(self.num_prefix)]

        super().__init__(config, **kwargs)

        self.inputs_max_length = 768
        self.targets_max_length = 192

    @property
    def hash_fields(self) -> list[Any]:
        return super().hash_fields + [
            self.num_prefix,
            self.deep,
            self.inputs_max_length,
            self.targets_max_length,
        ]

    def setup_tokenizer(self, retry=10) -> PreTrainedTokenizerBase:
        while True:
            try:
                tokenizer = T5Tokenizer.from_pretrained(self.transformer_model)
                break
            except HTTPError as e:
                if retry == 0:
                    raise e
            retry -= 1

        if not self.deep:
            tokenizer.add_tokens(self.task_tokens)
            task_token_ids = tokenizer(
                " ".join(self.task_tokens), return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            assert task_token_ids.shape[-1] == self.num_prefix
            self.task_token_ids = task_token_ids.squeeze(0).tolist()

        return tokenizer

    def tokenize(self, example: dict[str, Any], split: str):
        return NotImplementedError

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
        return pad_token_map_
