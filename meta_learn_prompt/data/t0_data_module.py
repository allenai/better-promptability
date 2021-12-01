from __future__ import annotations
import csv
import importlib
import json
import os
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
        sequence_length: Optional[Mapping[str, int]] = None,
        subsample_indices_file: Optional[str] = None,
        hf_cache_dir: Optional[PathOrStr] = None,
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
                sequence_length=sequence_length,
                subsample_indices_file=subsample_indices_file,
                hf_cache_dir=hf_cache_dir,
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
        sequence_length: Optional[Mapping[str, int]] = None,
        subsample_indices_file: Optional[str] = None,
        hf_cache_dir: Optional[PathOrStr] = None,
        **kwargs,
    ):
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.template_name = template_name
        self.sequence_length = sequence_length
        self.subsample_indices = None
        self.hf_cache_dir = hf_cache_dir
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
        if self.dataset_name == "story_cloze":
            data_dir = os.path.join(os.environ["STORY_CLOZE_PATH"], self.task_name)

            # Hack to add story cloze to the config in the P3 dataset builder -- import it first
            # and change relevant data structures
            dataset_module = datasets.load.dataset_module_factory(
                "bigscience/P3",
                revision=None,
                download_config=None,
                download_mode=None,
                data_files=None,
            )
            p3_module = importlib.import_module(dataset_module.module_path)

            # Mostly following https://huggingface.co/datasets/bigscience/P3/blob/main/P3.py
            task_splits_and_features = p3_module._TASK_SPLITS_AND_FEATURES_DICT  # type: ignore
            assert self.task_name not in task_splits_and_features
            for split_name in ("validation", "test"):  # story cloze has no training set
                split_info = json.load(open(os.path.join(data_dir, f"info.{split_name}.json")))
                features_dict = split_info["features"]
                assert split_info["num_shards"] == 1

                if self.task_name not in task_splits_and_features:
                    task_splits_and_features[self.task_name] = {
                        "splits": [],
                        "features_dict": features_dict,
                    }
                task_splits_and_features[self.task_name]["splits"].append(split_name)
                assert features_dict == task_splits_and_features[self.task_name]["features_dict"]
            splits_and_features_dict = task_splits_and_features[self.task_name]

            assert self.task_name not in p3_module._URLs  # type: ignore
            p3_module._URLs[self.task_name] = {  # type: ignore
                split_name: {"tfrecord": f"{data_dir}/{split_name}.tfrecord-00000-of-00001"}
                for split_name in splits_and_features_dict["splits"]
            }

            p3_module.P3.BUILDER_CONFIGS.append(  # type: ignore
                p3_module.P3Config(  # type: ignore
                    name=self.task_name,
                    splits=splits_and_features_dict["splits"],
                    features_dict=splits_and_features_dict["features_dict"],
                    score_eval=self.task_name.endswith("score_eval"),
                )
            )
            p3_module.P3.builder_configs = {  # type: ignore
                config.name: config for config in p3_module.P3.BUILDER_CONFIGS  # type: ignore
            }

        dataset_dict = datasets.load_dataset(
            "bigscience/P3", self.task_name, cache_dir=self.hf_cache_dir
        )

        if self.dataset_name == "story_cloze":
            # Story Cloze doesn't have a training split, so we use the validation split for training
            dataset_dict[self.train_split] = dataset_dict["validation"]
            del dataset_dict["validation"]

        return dataset_dict
