from __future__ import annotations
import importlib
import json
import os
import pickle
from typing import Any, Callable, Mapping, Optional

import t5
import datasets
from datasets import DatasetDict
import seqio

from .data_utils import md5
from .prompt_data_module import PromptDataModule


GREEN_DATASETS = [  # no big bench
    ("anli", "r1"),
    ("anli", "r2"),
    ("anli", "r3"),
    ("hellaswag", None),
    ("story_cloze", "2016"),
    ("super_glue", "cb"),
    ("super_glue", "copa"),
    ("super_glue", "rte"),
    ("super_glue", "wic"),
    ("super_glue", "wsc.fixed"),
    ("winogrande", "winogrande_xl"),
]


def get_task_name(dataset_name, subset_name, template_name):
    # UGH: Since it's impossible to work on this codebase given the dead-slow promptsource registry.
    if dataset_name == "unittest":
        return "unittest_unittest_unittest"

    # This import also populates seqio.MixtureRegistry. This is also the reason that it is put
    # here, since otherwise it causes very long start up time for everything.
    from promptsource.seqio_tasks import utils as ps_utils

    if dataset_name == "anli":
        assert subset_name in {"r1", "r2", "r3"}
        anli_round = subset_name
        subset_name = None
    task_name = ps_utils.get_task_name(dataset_name, subset_name, template_name)
    if dataset_name == "anli":
        task_name = task_name + "_" + anli_round
    return task_name


class T0Mixture:
    """
    This class is used to intiialize a T0DataModule or a collection of them. Supports three modes:
    1. Include all tasks in a mixture such as "d4_train" by providing `mixture_name`;
    2. Include all templates for a given dataset by providing `dataset_name` and `subset_name`;
    3. Include one specific dataset template by providing `dataset_name`, `subset_name`, and
        `template_name`.
    """

    def __init__(
        self,
        mixture_name: Optional[str] = None,  # most of the time either "d4_train" or "green"
        dataset_name: Optional[str] = None,
        subset_name: Optional[str] = None,
        template_name: Optional[str] = None,
        sequence_length: Optional[Mapping[str, int]] = None,
        subsample_indices_file: Optional[str] = None,
        *args,
        **kwargs,
    ):
        # This import also populates seqio.MixtureRegistry. This is also the reason that it is put
        # here, since otherwise it causes very long start up time for everything.
        from promptsource.seqio_tasks.tasks import all_templates

        self.use_green_datasets = mixture_name == "green"
        # green datasets are all within `d4_score_eval` but `d4_score_eval` has some extra ones
        self.mixture_name = "d4_score_eval" if self.use_green_datasets else mixture_name
        # There are local vars below with the same names, so saving these as fields & deleting them
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.template_name = template_name
        del mixture_name, dataset_name, subset_name, template_name
        assert (self.mixture_name is not None) != (self.dataset_name is not None)

        self.task_name_to_info: dict[str, tuple[str, Optional[str], str]] = {}
        for dataset_name, subset_name in all_templates.keys:
            dataset = all_templates.get_dataset(dataset_name, subset_name)
            for template_name in dataset.all_template_names:
                if dataset_name == "anli":
                    assert subset_name is None
                    for round in ("r1", "r2", "r3"):
                        task_name = get_task_name(dataset_name, round, template_name)
                        self.task_name_to_info[task_name] = (dataset_name, round, template_name)  # type: ignore
                        self.task_name_to_info[task_name + "_score_eval"] = (
                            dataset_name,
                            round,
                            template_name + "_score_eval",  # type: ignore
                        )
                else:
                    task_name = get_task_name(dataset_name, subset_name, template_name)
                    self.task_name_to_info[task_name] = (dataset_name, subset_name, template_name)  # type: ignore
                    self.task_name_to_info[task_name + "_score_eval"] = (
                        dataset_name,
                        subset_name,
                        template_name + "_score_eval",  # type: ignore
                    )

        tasks = None
        if self.mixture_name is not None:
            mixture = seqio.MixtureRegistry.get(self.mixture_name)
            tasks = mixture.tasks
        else:
            task_names = [
                task_name
                for task_name, (
                    dataset_name,
                    subset_name,
                    template_name,
                ) in self.task_name_to_info.items()
                if dataset_name == self.dataset_name
                and subset_name == self.subset_name
                and (
                    template_name == self.template_name if self.template_name is not None else True
                )
            ]
            tasks = [seqio.TaskRegistry.get(task_name) for task_name in task_names]

        self.data_modules: dict[str, T0DataModule] = {}
        for task in tasks:
            dataset_name, subset_name, template_name = self.task_name_to_info[task.name]
            if self.use_green_datasets and (dataset_name, subset_name) not in GREEN_DATASETS:
                continue

            assert task.name not in self.data_modules
            self.data_modules[task.name] = T0DataModule(
                dataset_name,
                subset_name,
                template_name,
                sequence_length,
                subsample_indices_file,
                *args,
                **kwargs,
            )
        assert len(self.data_modules) > 0


@PromptDataModule.register("t0")
class T0DataModule(PromptDataModule):
    """
    Represents a single dataset AND template, but all the splits.
    """

    def __init__(
        self,
        dataset_name: str,
        template_name: str,
        subset_name: Optional[str] = None,
        sequence_length: Optional[Mapping[str, int]] = None,
        subsample_indices_file: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.template_name = template_name
        self.task_name = get_task_name(self.dataset_name, self.subset_name, self.template_name)
        self.sequence_length = sequence_length
        self.subsample_indices = None
        if subsample_indices_file is not None:
            self.subsample_indices = pickle.load(open(subsample_indices_file, "rb"))[
                (dataset_name, subset_name)
            ]
        super().__init__(*args, **kwargs)

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

    @classmethod
    def get_metric_postprocessor(cls) -> Callable:
        # Keeping t5 as a dependency for now; may want to evaluate on other tasks.
        return t5.data.postprocessors.rank_classification

    @property
    def metric_names(self) -> list[str]:
        # For all the green (i.e., d4_score_eval) datasets, all tasks have rank classification as the metric.
        return ["rank_classification"]

    def instantiate_metric(self, metric_name: str, split: str) -> Callable:
        return t5.evaluation.metrics.rank_classification

    @property
    def metric_watch_mode(self) -> str:
        return "max"

    @property
    def sort_key(self) -> str:
        return "inputs"

    def load(self) -> DatasetDict:
        # Not the cleanest code, refactor later.
        if self.dataset_name == "unittest":
            return datasets.load_from_disk("test_fixtures/data/unittest")

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

        dataset_dict = datasets.load_dataset("bigscience/P3", self.task_name)

        if self.dataset_name == "story_cloze":
            # Story Cloze doesn't have a training split, so we use the validation split for training
            dataset_dict[self.train_split] = dataset_dict["validation"]
            del dataset_dict["validation"]

        return dataset_dict
