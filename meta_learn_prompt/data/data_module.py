from __future__ import annotations
import hashlib
import logging
import os
from abc import abstractmethod, abstractproperty
from collections.abc import ItemsView
from typing import Any, Mapping, Optional, Union

from allennlp.training.metrics import Metric
import datasets
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from tango.common.aliases import PathOrStr
from tango.integrations.pytorch_lightning.data import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import LengthGroupedSampler

from .config import Config
from .data_utils import PAD_TYPE, collate_fn


# Sometimes we want to change the implementation of methods, etc., which cache ignores.
# We maintain our own cache so this is not very useful anyway.
datasets.set_caching_enabled(False)


logger = logging.getLogger(__name__)


class DataModule(LightningDataModule):
    """
    Abstract class representing a lightning data module using HF datasets, relevant properties,
    and a tokenizer.
    """

    def __init__(
        self,
        config: Config,
        data_dir: PathOrStr,
        max_length: Optional[int] = None,
        preprocess_and_save: bool = True,
        batch_size: int = 32,
        eval_batch_size: int = 32,
    ):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.max_length = max_length
        self.preprocess_and_save = preprocess_and_save

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

    def setup(self, stage: Optional[str] = None):

        if self.preprocess_and_save:
            self.tokenizer = self.setup_tokenizer()
            if os.path.exists(self.cache_path):
                logger.info(f"Reusing cache at {self.cache_path}")
                self.dataset_dict = DatasetDict.load_from_disk(self.cache_path)
                return

        self.dataset_dict = self.load()
        if self.preprocess_and_save:
            self.dataset_dict = self.preprocess(self.dataset_dict)
            logger.info(f"Saving dataset cache at {self.cache_path}")
            self.dataset_dict.save_to_disk(self.cache_path)

    def _to_params(self):
        return {}

    def __getitem__(self, key: str) -> HFDataset:
        return self.dataset_dict[key]

    @property
    def hash_fields(self) -> list[Any]:
        """For cache purpose"""
        return [self.config.seed, self.tokenizer.__repr__()]

    @property
    def cache_path(self) -> str:
        def hash(s):
            return hashlib.md5(s.encode("utf-8")).hexdigest()

        hash_fields = "".join([str(f) for f in self.hash_fields])
        return os.path.join(
            self.data_dir,
            f"{self.__class__.__name__}_{hash(hash_fields)}.datacache",
        )

    @property
    def train_split(self) -> str:
        return "train"

    @property
    def dev_splits(self) -> list[str]:
        return ["dev"]

    @property
    def test_splits(self) -> list[str]:
        return ["test"]

    @property
    def text_key(self) -> str:
        """The key in the example dictionary for the main text."""
        return "text"

    @property
    def second_text_key(self) -> Union[str, None]:
        """For text pairs, the key in the example dictionary for the second text."""
        return None

    @property
    def label_key(self) -> str:
        """The key in the example dictionary for the label."""
        return "label"

    @property
    def sort_key(self) -> str:
        return self.text_key

    @property
    @abstractproperty
    def metric_names(self) -> list[str]:
        raise NotImplementedError("This is an abstract property. Did you forget to implement it?")

    def instantiate_metric(self, metric_name: str, split: str) -> Metric:
        return Metric.by_name(metric_name)()

    def postprocess_metric(self, metric_name: str, metric: Any) -> Union[int, float]:
        """Postprocesses whatever Metric.get_metric() returns into a number that can be compared."""
        return metric

    @property
    def metric_to_watch(self) -> str:
        if len(self.metric_names) == 1:
            return self.metric_names[0]
        else:
            raise NotImplementedError(
                "This is an abstract property. Did you forget to implement it?"
            )

    @property
    @abstractproperty
    def metric_watch_mode(self) -> str:
        raise NotImplementedError("This is an abstract property. Did you forget to implement it?")

    @property
    @abstractproperty
    def output_mode(self) -> str:
        raise NotImplementedError("This is an abstract property. Did you forget to implement it?")

    @property
    def num_labels(self) -> Union[int, None]:
        if self.output_mode == "classification":
            raise NotImplementedError("You need to implement this property")
        return None

    @abstractmethod
    def load(self) -> DatasetDict:
        raise NotImplementedError("This is an abstract method. Did you forget to implement it?")

    def tokenize(self, examples: dict[str, list], split: str) -> dict[str, list]:
        return self.tokenizer(
            examples[self.text_key],
            text_pair=examples[self.second_text_key] if self.second_text_key is not None else None,
            padding=False,  # we control this in the collator
            truncation=True,
            max_length=self.max_length,
        )

    def preprocess(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = DatasetDict(  # reimplementing DatasetDict.map to provide `split`
            {
                split: dataset.map(
                    lambda examples: self.tokenize(examples, split),
                    batched=False,  # to make tokenization/transformation easier
                    num_proc=1,  # TODO: this can't be > 1 in tango, for some reason.
                )
                for split, dataset in dataset_dict.items()
            }
        )

        # Rename validation -> dev
        if "validation" in dataset_dict and "dev" not in dataset_dict:
            dataset_dict["dev"] = dataset_dict["validation"]
            del dataset_dict["validation"]

        return dataset_dict

    @abstractmethod
    def setup_tokenizer(self) -> PreTrainedTokenizerBase:
        raise NotImplementedError("This is an abstract method. Did you forget to implement it?")

    def items(self) -> ItemsView:
        return self.dataset_dict.items()

    def dataloader(self, split: str, batch_size: int, shuffle=False) -> DataLoader:
        dataset_split = self.dataset_dict[split]
        if shuffle:
            # LengthGroupedSampler sorts from longest to shortest; we want the reverse
            lens = [-len(ids) for ids in dataset_split[self.sort_key]]
            if self.config.gpus <= 1:
                sampler = LengthGroupedSampler(None, batch_size, lengths=lens)
            else:
                # TODO: support this when
                # https://github.com/huggingface/transformers/commit/1b74af76b7e5c259d1470dec9d8d68c303dea5db
                # is released and also remove the None from above
                raise NotImplementedError()
        else:
            sampler = None
        pad_token_map = self.pad_token_map(split)
        assert all(pad is not None for pad in pad_token_map.values())

        dataloader = DataLoader(
            dataset_split,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            # num_workers=1,
            collate_fn=lambda batch: collate_fn(
                batch, self.label_key, pad_token_map, self.tokenizer.padding_side, self.output_mode
            ),
            pin_memory=True,
        )

        return dataloader

    def pad_token_map(self, split: str) -> Mapping[str, PAD_TYPE]:
        """
        Specifies the padding for each key. Only keys including in this map plus the label will be
        included in the batch.
        """
        return {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": False,
            "token_type_ids": self.tokenizer.pad_token_type_id,
        }

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_split, self.batch_size, shuffle=True)

    def val_dataloader(self, shuffle: bool = False):
        return [
            self.dataloader(split, self.eval_batch_size, shuffle=shuffle)
            for split in self.dev_splits
        ]

    def test_dataloader(self, shuffle: bool = False):
        return [
            self.dataloader(split, self.eval_batch_size, shuffle=shuffle)
            for split in self.test_splits
        ]
