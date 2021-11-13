from __future__ import annotations
from typing import Mapping, Optional

from allennlp.training.metrics import Metric
from datasets import Dataset, DatasetDict
from promptsource.seqio_tasks import (  # noqa: E501; intended unused import for promptsource to populate seqio.MixtureRegistryß
    tasks as ps_tasks,
)
import seqio
import tensorflow_datasets as tfds

from .prompt_data_module import PromptDataModule


POSSIBLE_MIXTURES = {"d4_train", "d4_eval"}


def tf_dataset_to_hf_dataset(tf_dataset):
    return Dataset.from_pandas(tfds.as_dataframe(tf_dataset))


class T0Mixture:
    def __init__(
        self, mixture_name: str, sequence_length: Optional[Mapping[str, int]], *args, **kwargs
    ):
        # assert mixture_name in POSSIBLE_MIXTURES
        mixture = seqio.MixtureRegistry.get(mixture_name)
        self.data_modules = {
            task.name: T0DataModule(task, sequence_length, *args, **kwargs)
            for task in mixture.tasks
        }


@PromptDataModule.register("t0")
class T0DataModule(PromptDataModule):
    """
    Represents a single dataset AND template, but all the splits.
    """

    def __init__(
        self, seqio_task: seqio.Task, sequence_length: Optional[Mapping[str, int]], *args, **kwargs
    ):
        self.seqio_task = seqio_task
        self.sequence_length = sequence_length
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
        return DatasetDict(
            {
                split: tf_dataset_to_hf_dataset(
                    self.seqio_task.get_dataset(self.sequence_length, split=split, shuffle=False)
                )
                for split in self.seqio_task.splits
            }
        )
