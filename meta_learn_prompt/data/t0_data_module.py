from __future__ import annotations
import pickle
from typing import Mapping, Optional

from allennlp.training.metrics import Metric
from datasets import Dataset, DatasetDict
import seqio
import tensorflow_datasets as tfds

from .data_utils import md5
from .prompt_data_module import PromptDataModule


def get_task_name(dataset_name, subset_name, template_name):
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


def tf_dataset_to_hf_dataset(tf_dataset):
    return Dataset.from_pandas(tfds.as_dataframe(tf_dataset))


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
        mixture_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        subset_name: Optional[str] = None,
        template_name: Optional[str] = None,
        sequence_length: Optional[Mapping[str, int]] = None,
        subsample_indices_file: Optional[str] = None,
        *args,
        **kwargs
    ):
        # This import also populates seqio.MixtureRegistry. This is also the reason that it is put
        # here, since otherwise it causes very long start up time for everything.
        from promptsource.seqio_tasks.tasks import all_templates

        # There are local vars below with the same names, so saving these as fields & deleting them
        self.mixture_name = mixture_name
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.template_name = template_name
        del mixture_name, dataset_name, subset_name, template_name
        assert (self.mixture_name is not None) != (self.dataset_name is not None)

        self.task_name_to_info = {}
        for dataset_name, subset_name in all_templates.keys:
            dataset = all_templates.get_dataset(dataset_name, subset_name)
            for template_name in dataset.all_template_names:
                if dataset_name == "anli":
                    assert subset_name is None
                    for round in ("r1", "r2", "r3"):
                        task_name = get_task_name(dataset_name, round, template_name)
                        self.task_name_to_info[task_name] = (dataset_name, round, template_name)
                else:
                    task_name = get_task_name(dataset_name, subset_name, template_name)
                    self.task_name_to_info[task_name] = (dataset_name, subset_name, template_name)

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

        dataset_to_subsample_indices = None
        if subsample_indices_file is not None:
            dataset_to_subsample_indices = pickle.load(open(subsample_indices_file, "rb"))

        self.data_modules = {}
        for task in tasks:
            dataset_name, subset_name, template_name = self.task_name_to_info[task.name]
            assert task.name not in self.data_modules
            self.data_modules[task.name] = T0DataModule(
                dataset_name,
                subset_name,
                template_name,
                task,
                sequence_length,
                dataset_to_subsample_indices[(dataset_name, subset_name)],
                *args,
                **kwargs
            )
        assert len(self.data_modules) > 0


class T0DataModule(PromptDataModule):
    """
    Represents a single dataset AND template, but all the splits.
    """

    def __init__(
        self,
        dataset_name: str,
        subset_name: Optional[str],
        template_name: str,
        seqio_task: seqio.Task,
        sequence_length: Optional[Mapping[str, int]],
        subsample_indices: Optional[tuple[list[int], str]],
        *args,
        **kwargs
    ):
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.template_name = template_name
        self.seqio_task = seqio_task
        self.sequence_length = sequence_length
        self.subsamplme_indices = subsample_indices
        super().__init__(*args, **kwargs)

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
        dataset_dict = DatasetDict(
            {
                split: tf_dataset_to_hf_dataset(
                    self.seqio_task.get_dataset(self.sequence_length, split=split, shuffle=False)
                )
                for split in self.seqio_task.splits
            }
        )
        if self.subsamplme_indices is not None:
            indices, checksum = self.subsamplme_indices
            dataset = dataset_dict[tfds.Split.TRAIN].select(indices)
            assert md5("".join(str(sorted(ex.items())) for ex in dataset)) == checksum
            dataset_dict[tfds.Split.TRAIN] = dataset
        return dataset_dict
