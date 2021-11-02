from collections.abc import ItemsView
import hashlib
import logging
import numpy as np
import os
from typing import Any, Mapping, Optional, Union

import datasets
from datasets import DatasetDict, Dataset as HFDataset, load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers import GPT2Tokenizer, PreTrainedTokenizerBase
from transformers.trainer_pt_utils import LengthGroupedSampler

from tango.common.aliases import PathOrStr
from tango.integrations.pytorch_lightning.data import LightningDataModule

from .config import Config
from .data_utils import PAD_TYPE, collate_fn
from .templates import templatize, get_possible_labels

TSV_FORMAT = {"amazon", "sst-2", "agnews", "dbpedia", "yahoo", "yelp_full"}
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
    def metric_names(self) -> list[str]:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def metric_to_watch(self) -> str:
        if len(self.metric_names) == 1:
            return self.metric_names[0]
        else:
            raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def metric_watch_mode(self) -> str:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def output_mode(self) -> str:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def num_labels(self) -> Union[int, None]:
        if self.output_mode == "classification":
            raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")
        return None

    def load(self) -> DatasetDict:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

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

    def setup_tokenizer(self) -> PreTrainedTokenizerBase:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

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


class LocalDataModule(DataModule):
    def __init__(
        self,
        tsv: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tsv = tsv

    def load(self):

        self.split_filename = lambda split: f"{split}.tsv" if self.tsv else f"{split}.csv"
        dataset_dict = load_dataset(
            "csv",
            data_files={
                split: os.path.join(self.data_dir, self.split_filename(split))
                for split in [self.train_split] + self.dev_splits + self.test_splits
            },
            skiprows=1 if self.tsv else 0,
            delimiter="\t" if self.tsv else ",",
            column_names=[self.text_key, self.label_key]
            if self.tsv
            else [self.label_key, self.text_key],
        )
        assert isinstance(dataset_dict, DatasetDict)
        return dataset_dict


@LightningDataModule.register("few_shot")
class FewShotDataset(LocalDataModule):
    def __init__(
        self,
        dataset: str,
        num_prefix: int,
        template_idx: int,
        transformer_model: Union[str, PathOrStr],
        *args,
        **kwargs,
    ):

        self.dataset = dataset.lower()
        self.num_prefix = num_prefix
        self.template_idx = template_idx
        self.transformer_model = transformer_model

        tsv = self.dataset in TSV_FORMAT
        self.task_tokens = ["<TASK{}>".format(str(i).zfill(2)) for i in range(self.num_prefix)]

        super().__init__(tsv, *args, **kwargs)

        self.max_length: int = 256 if self.dataset in LONG_DATASETS else 128

    @property
    def hash_fields(self) -> list[Any]:
        return super().hash_fields + [self.dataset, self.template_idx]

    @property
    def metric_names(self) -> list[str]:
        return ["categorical_accuracy"]

    @property
    def metric_watch_mode(self) -> str:
        return "max"

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
            input = self.tokenizer(input)["input_ids"][:self.max_length - 16]
            return assemble_prompt(
                prefix, input, self.tokenizer.eos_token_id, self.task_token_ids
            )

        if split == self.train_split:
            input_ids, attention_mask, label_mask, label = prepare(example[self.label_key])
        else:
            input_ids, attention_mask, label_mask, label = [], [], [], []
            for possible_label in get_possible_labels(self.dataset):
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
