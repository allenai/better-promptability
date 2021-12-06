from __future__ import annotations
import csv
import importlib
import json
import os
from pathlib import Path
import pickle
from typing import Any, Mapping, Optional

from allennlp.training.metrics import Metric
import datasets
from datasets import DatasetDict
from tango.common import PathOrStr

from .config import Config
from .data_utils import md5
from .prompt_data_module import PromptDataModule


class T0Mixture:
    """
    This class is used to initialize a collection of T0DataModule.
    """

    def __init__(
        self,
        mixture_name: str,  # should be "d4_train" or "green"
        config: Config,
        data_dir: PathOrStr,
        num_prefix: int,
        transformer_model: PathOrStr,
        t0_data_cache: PathOrStr = "/net/nfs2.allennlp/petew/meta-learn-prompt/t0/cache",
        sequence_length: Optional[Mapping[str, int]] = None,
        subsample_indices_file: Optional[str] = None,
        **data_module_kwargs,
    ):
        assert mixture_name in {"d4_train", "green"}
        self.mixture_name = mixture_name
        self.task_name_to_info: dict[str, tuple[str, Optional[str], str]] = {}  # TODO
        with open("data/t0_task_info.tsv", newline="") as task_info_file:
            reader = csv.DictReader(task_info_file, delimiter="\t")
            for row in reader:
                self.task_name_to_info[row["task_name"]] = (
                    row["dataset_name"],
                    row["subset_name"],
                    row["template_name"],
                )
        self.data_modules: dict[str, T0DataModule] = {}
        for task_name in (line.strip() for line in open(f"data/{self.mixture_name}_tasks.txt")):
            dataset_name, subset_name, template_name = self.task_name_to_info[task_name]
            self.data_modules[task_name] = T0DataModule(
                config=config,
                data_dir=data_dir,
                num_prefix=num_prefix,
                transformer_model=transformer_model,
                task_name=task_name,
                dataset_name=dataset_name,
                subset_name=subset_name,
                template_name=template_name,
                t0_data_cache=t0_data_cache,
                sequence_length=sequence_length,
                subsample_indices_file=subsample_indices_file,
                **data_module_kwargs,
            )
        assert len(self.data_modules) > 0


@PromptDataModule.register("t0", exist_ok=True)
class T0DataModule(PromptDataModule):
    """
    Represents a single dataset AND template, but all the splits.
    """

    def __init__(
        self,
        config: Config,
        data_dir: PathOrStr,
        num_prefix: int,
        transformer_model: PathOrStr,
        task_name: str,
        dataset_name: str,
        subset_name: Optional[str],
        template_name: str,
        t0_data_cache: PathOrStr = "/net/nfs2.allennlp/petew/meta-learn-prompt/t0/cache",
        sequence_length: Optional[Mapping[str, int]] = None,
        subsample_indices_file: Optional[str] = None,
        **kwargs,
    ):
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.template_name = template_name
        self.t0_data_cache = Path(t0_data_cache)
        self.sequence_length = sequence_length
        self.subsample_indices = None
        if subsample_indices_file is not None:
            self.subsample_indices = pickle.load(open(subsample_indices_file, "rb"))[
                (dataset_name, subset_name)
            ]
        super().__init__(
            config=config,
            data_dir=data_dir,
            num_prefix=num_prefix,
            transformer_model=transformer_model,
            **kwargs,
        )

    def setup(self, stage: Optional[str] = None):
        super().setup(stage=stage)
        if self.subsample_indices is not None:
            indices, checksum = self.subsamplme_indices
            dataset = self.dataset_dict[self.train_split].select(indices)
            assert md5("".join(str(sorted(ex.items())) for ex in dataset)) == checksum
            self.dataset_dict[self.train_split] = dataset

    @property
    def hash_fields(self) -> list[Any]:
        return super().hash_fields + [self.task_name]

    @property
    def dev_splits(self) -> list[str]:
        # Story Cloze doesn't have a training split, so we use the dev split for training
        return super().dev_splits if self.dataset_name != "story_cloze" else []

    @property
    def metric_names(self) -> list[str]:
        raise NotImplementedError  # TODO(akshitab): probably reuse the metrics info in self.seqio_task?

    def instantiate_metric(self, metric_name: str, split: str) -> Metric:
        raise NotImplementedError  # TODO(akshitab): ditto

    @property
    def metric_watch_mode(self) -> str:
        return "max"  # TODO(akshitab): verify

    @property
    def sort_key(self) -> str:
        return "inputs"

    def load(self) -> DatasetDict:
        data_path = self.t0_data_cache / self.task_name
        assert data_path.is_dir()

        dataset_dict = datasets.load_from_disk(data_path)

        if self.dataset_name == "story_cloze":
            # Story Cloze doesn't have a training split, so we use the validation split for training
            dataset_dict[self.train_split] = dataset_dict["validation"]
            del dataset_dict["validation"]

        return dataset_dict
