from __future__ import annotations
import os
from typing import Any

from datasets import DatasetDict, load_dataset

from .noisy_channel_data_module import NoisyChannelDataModule


TSV_FORMAT = {"amazon", "sst-2", "agnews", "dbpedia", "yahoo", "yelp_full"}


@NoisyChannelDataModule.register("few_shot")
class FewShotDataModule(NoisyChannelDataModule):
    def __init__(self, dataset: str, *args, **kwargs):
        self.dataset = dataset.lower()
        self.tsv = self.dataset in TSV_FORMAT
        super().__init__(*args, **kwargs)

    @property
    def hash_fields(self) -> list[Any]:
        return super().hash_fields + [self.dataset]

    @property
    def metric_names(self) -> list[str]:
        return ["categorical_accuracy"]

    @property
    def metric_watch_mode(self) -> str:
        return "max"

    def load(self) -> DatasetDict:
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
