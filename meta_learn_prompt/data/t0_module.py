from __future__ import annotations
from typing import Any, Mapping, Optional

from tango.common.aliases import PathOrStr

from pathlib import Path
import pickle

from allennlp.training.metrics import Metric
import datasets
from tango.common import DatasetDict

from .data_utils import md5
from .prompt_data_module import PromptDataModule
from .config import Config


@PromptDataModule.register("t0", exist_ok=True)
class T0Module(PromptDataModule):
    """
    Represents a single dataset AND template, but all the splits.
    """

    def __init__(
        self,
        config: Config,
        data_dir: PathOrStr,
        num_prefix: int,
        transformer_model: PathOrStr,
        mixture_name: str,
        task_name: str,
        dataset_name: str,
        subset_name: Optional[str],
        template_name: str,
        t0_data_cache: PathOrStr = "/net/nfs2.allennlp/petew/meta-learn-prompt/t0/cache",
        sequence_length: Optional[Mapping[str, int]] = None,
        subsample_indices_file: Optional[str] = None,
        **kwargs,
    ):

        super().__init__(config, data_dir, num_prefix, transformer_model, mixture_name, **kwargs)

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

    @property
    def hash_fields(self) -> list[Any]:
        return super().hash_fields + [self.task_name]

    def setup(self, stage: Optional[str] = None):
        super().setup(stage=stage)
        if self.subsample_indices is not None:
            indices, checksum = self.subsamplme_indices
            dataset = self.dataset_dict[self.train_split].select(indices)
            assert md5("".join(str(sorted(ex.items())) for ex in dataset)) == checksum
            self.dataset_dict[self.train_split] = dataset

    @property
    def dev_splits(self) -> list[str]:
        # Story Cloze doesn't have a training split, so we use the dev split for training
        return super().dev_splits if self.dataset_name != "story_cloze" else []

    @property
    def metric_names(self) -> list[str]:
        # For all the green (i.e., d4_score_eval) datasets, all tasks have accuracy as the metric.
        return ["categorical_accuracy"]

    def instantiate_metric(self, metric_name: str, split: str) -> Metric:
        # return t5.evaluation.metrics.rank_classification
        return Metric.by_name(metric_name)()

    @property
    def metric_watch_mode(self) -> str:
        return "max"

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
