import os

from datasets import DatasetDict, load_dataset

from .data_module import DataModule


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
