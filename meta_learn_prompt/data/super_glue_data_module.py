from typing import Union

from datasets import DatasetDict
from tango.common.aliases import PathOrStr
from tango.integrations.pytorch_lightning.data import LightningDataModule
from transformers import PreTrainedTokenizerBase

from .data_module import DataModule


SUPER_GLUE_DATASETS = {
    "axb",  # broadcoverage diagnostics
    "axg",  # winogender schema diagnostics
    "cb",  # commitment  bank
    "copa",  # choise of plausible alternatives
    "multirc",  # multi-sentence reading comprehension
    "rte",  # recognizing textual entailment
    "wic",  # words in context
    "wsc",  # winograd schema challenge
    "boolq",  # BoolQ
    "record",  # reading comprehension with commonsense reasoning
}


@LightningDataModule.register("super_glue_pretrain")
class SuperGlueDataModule(DataModule):
    def __init__(
        self,
        transformer_model: PathOrStr,
        *args,
        datasets_to_include: set[str] = {"cb", "copa", "multirc", "rte", "wic", "boolq", "record"},
        **kwargs,
    ):
        for name in datasets_to_include:
            if name not in SUPER_GLUE_DATASETS:
                raise ValueError(f"Bad dataset name '{name}'")

        super().__init__(*args, **kwargs)

        self.datasets_to_include = datasets_to_include
        self.transformer_model = transformer_model

    @property
    def metric_names(self) -> list[str]:
        pass

    @property
    def metric_to_watch(self) -> str:
        pass

    @property
    def metric_watch_mode(self) -> str:
        return "max"

    @property
    def output_mode(self) -> str:
        pass

    @property
    def num_labels(self) -> Union[int, None]:
        pass

    def load(self) -> DatasetDict:
        pass

    def setup_tokenizer(self) -> PreTrainedTokenizerBase:
        pass
