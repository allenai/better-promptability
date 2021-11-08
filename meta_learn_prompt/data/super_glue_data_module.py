from typing import Any, Optional, Union

from allennlp.training.metrics import Metric
from datasets import DatasetDict, load_dataset
from datasets.arrow_dataset import Batch
import torch

from .noisy_channel_data_module import NoisyChannelDataModule


SUPER_GLUE_DATASETS = {
    "axb",  # broadcoverage diagnostics
    "axg",  # winogender schema diagnostics
    "cb",  # commitment  bank
    "copa",  # choise of plausible alternatives
    "multirc",  # multi-sentence reading comprehension
    "rte",  # recognizing textual entailment
    "wic",  # words in context
    "wsc",  # winograd schema challenge
    "boolq",  # BoolQ
    "record",  # reading comprehension with commonsense reasoning
}
DATA_TRANFORMATION_PREFIX = "aug_"


class BaseSuperGlueDataModule(NoisyChannelDataModule):
    dataset = None  # name of the dataset

    def __init__(self, n_shot: Optional[int] = None, *args, **kwargs):
        self.n_shot = n_shot
        super().__init__(*args, **kwargs)

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        if self.n_shot is not None:
            self.dataset_dict[self.train_split] = (
                self.dataset_dict[self.train_split].shuffle().select(range(self.n_shot))
            )

    @property
    def metric_names(self) -> list[str]:
        return ["categorical_accuracy"]

    @property
    def metric_watch_mode(self) -> str:
        return "max"

    def load(self) -> DatasetDict:
        return load_dataset("super_glue", name=self.dataset)


@NoisyChannelDataModule.register("boolq")
class BoolqDataModule(BaseSuperGlueDataModule):
    dataset = "boolq"

    @property
    def sort_key(self) -> str:
        return "passage"


@NoisyChannelDataModule.register("cb")
class CbDataModule(BaseSuperGlueDataModule):
    dataset = "cb"

    @property
    def metric_names(self) -> list[str]:
        return ["categorical_accuracy", "f1"]

    @property
    def metric_to_watch(self) -> str:
        return "categorical_accuracy"

    def instantiate_metric(self, metric_name: str, split: str) -> Metric:
        if metric_name == "f1":
            return Metric.by_name("fbeta")(beta=1, average="macro")
        else:
            return super().instantiate_metric(metric_name, split)

    @property
    def sort_key(self) -> str:
        return "premise"


@NoisyChannelDataModule.register("rte")
class RteDataModule(BaseSuperGlueDataModule):
    dataset = "rte"

    @property
    def sort_key(self) -> str:
        return "premise"


@NoisyChannelDataModule.register("copa")
class CopaDataModule(BaseSuperGlueDataModule):
    dataset = "copa"

    @property
    def sort_key(self) -> str:
        return "premise"

    def preprocess(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = super().preprocess(dataset_dict)

        dataset_dict = DatasetDict(  # reimplementing DatasetDict.map to provide `split`
            {
                split: dataset.map(
                    lambda examples: self._data_aug(split, examples),
                    batched=True,
                    remove_columns=dataset.column_names,
                )  # these are needed to add rows; see https://huggingface.co/docs/datasets/process.html
                for split, dataset in dataset_dict.items()
            }
        )
        for dataset in dataset_dict.values():
            for name in dataset.column_names:
                assert name.startswith(DATA_TRANFORMATION_PREFIX)
                dataset.rename_column_(name, name[len(DATA_TRANFORMATION_PREFIX) :])

        return dataset_dict

    def _data_aug(self, split: str, examples: Batch) -> dict[str, list[Any]]:
        if split != self.train_split:
            return {DATA_TRANFORMATION_PREFIX + k: v for k, v in examples.items()}

        # data augmentation: mirror choice1 and choice2
        new_examples = {
            DATA_TRANFORMATION_PREFIX + "choice1": examples["choice1"] + examples["choice2"],
            DATA_TRANFORMATION_PREFIX + "choice2": examples["choice2"] + examples["choice1"],
        }
        for field in set(examples.data.keys()) - {"choice1", "choice2"}:
            new_examples[DATA_TRANFORMATION_PREFIX + field] = examples[field] + examples[field]

        return new_examples


@NoisyChannelDataModule.register("wic")
class WicDataModule(BaseSuperGlueDataModule):
    dataset = "wic"

    @property
    def sort_key(self) -> str:
        return "sentence1"

    def preprocess(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = super().preprocess(dataset_dict)

        dataset_dict = DatasetDict(  # reimplementing DatasetDict.map to provide `split`
            {
                split: dataset.map(
                    lambda examples: self._data_aug(split, examples),
                    batched=True,
                    remove_columns=dataset.column_names,
                )  # these are needed to add rows; see https://huggingface.co/docs/datasets/process.html
                for split, dataset in dataset_dict.items()
            }
        )
        for dataset in dataset_dict.values():
            for name in dataset.column_names:
                assert name.startswith(DATA_TRANFORMATION_PREFIX)
                dataset.rename_column_(name, name[len(DATA_TRANFORMATION_PREFIX) :])

        return dataset_dict

    def _data_aug(self, split: str, examples: Batch) -> dict[str, list[Any]]:
        if split != self.train_split:
            return {DATA_TRANFORMATION_PREFIX + k: v for k, v in examples.items()}

        # data augmentation: mirror sentence1 and sentence2
        new_examples = {
            DATA_TRANFORMATION_PREFIX + "sentence1": examples["sentence1"] + examples["sentence2"],
            DATA_TRANFORMATION_PREFIX + "sentence2": examples["sentence2"] + examples["sentence1"],
        }
        for field in set(examples.data.keys()) - {"sentence1", "sentence2"}:
            new_examples[DATA_TRANFORMATION_PREFIX + field] = examples[field] + examples[field]

        return new_examples


@NoisyChannelDataModule.register("multirc")
class MultircDataModule(BaseSuperGlueDataModule):
    dataset = "multirc"

    @property
    def metric_names(self) -> list[str]:
        return ["f1"]  # TODO: em

    def instantiate_metric(self, metric_name: str, split: str) -> Metric:
        if metric_name == "f1":
            return Metric.by_name("f1")(positive_label=1)
        else:
            return super().instantiate_metric(metric_name, split)

    def postprocess_metric(self, metric_name: str, metric: Any) -> Union[int, float]:
        if metric_name == "f1":
            return metric["f1"]
        else:
            return super().postprocess_metric(metric_name, metric)

    @property
    def sort_key(self) -> str:
        return "paragraph"


@NoisyChannelDataModule.register("record")
class RecordDataModule(BaseSuperGlueDataModule):
    dataset = "record"

    @property
    def metric_names(self) -> list[str]:
        return ["em"]  # TODO: f1

    def compute_example_indices(self, examples: dict[str, list[Any]]) -> list[list[int]]:
        example_indices = []
        next = 0
        for answers in examples["answers"]:
            example_indices.append([next + i for i in range(len(answers))])
            next += len(answers)
        return example_indices

    def instantiate_metric(self, metric_name: str, split: str) -> Metric:
        if metric_name == "em":
            return RecordEM(self.all_example_indices[split])
        else:
            return super().instantiate_metric(metric_name, split)

    @property
    def sort_key(self) -> str:
        return "passage"

    def preprocess(self, dataset_dict: DatasetDict) -> DatasetDict:
        # for metric instantiation
        self.all_example_indices = {
            split: self.compute_example_indices(examples)
            for split, examples in dataset_dict.items()
        }

        dataset_dict = DatasetDict(  # reimplementing DatasetDict.map to provide `split`
            {
                split: dataset.map(
                    lambda examples: self._reformat(split, examples),
                    batched=True,
                    remove_columns=dataset.column_names,
                )  # these are needed to add rows; see https://huggingface.co/docs/datasets/process.html
                for split, dataset in dataset_dict.items()
            }
        )
        for k, dataset in dataset_dict.items():
            for name in dataset.column_names:
                assert name.startswith(DATA_TRANFORMATION_PREFIX)
                dataset.rename_column_(name, name[len(DATA_TRANFORMATION_PREFIX) :])

        dataset_dict = super().preprocess(dataset_dict)

        return dataset_dict

    def _reformat(self, split: str, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # we follow the GPT-3 paper wrt @highlight annotations
        examples["passage"] = [p.replace("@highlight\n", "- ") for p in examples["passage"]]

        if split != "test":
            # create one example per answer
            new_examples: dict[str, list] = {
                k if k != "answers" else self.label_key: [] for k in examples.keys()
            }
            for i, answers in enumerate(examples["answers"]):
                for answer in answers:
                    new_examples[self.label_key].append(answer)
                    for field in set(examples.keys()) - {"answers"}:
                        new_examples[field].append(examples[field][i])
            examples = new_examples
        else:
            raise NotImplementedError  # pivoting project and abandoning this dataset

        return {DATA_TRANFORMATION_PREFIX + k: v for k, v in examples.items()}


@Metric.register("record_em")
class RecordEM(Metric):
    def __init__(self, example_indices: list[list[int]]) -> None:
        """
        example_indices: mapping between the original examples and the split sequence indices.
            E.g. [[0, 1, 2], [3, 4], [5, 6, 7, 8], ...]
        """
        assert sum(len(indices) for indices in example_indices) == example_indices[-1][-1] + 1
        self.example_indices = example_indices
        self.n_total = example_indices[-1][-1] + 1
        self.reset()

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        """
        breakpoint()
        predictions, gold_labels = self.detach_tensors(predictions, gold_labels)
        self.predictions.extend(predictions)
        self.gold_labels.extend(gold_labels)

    def get_metric(self, reset: bool):
        assert len(self.predictions) == len(self.gold_labels) == self.n_total
        n_correct = sum(
            int(any(self.predictions[i] == self.gold_labels[i] for i in indices))
            for indices in self.example_indices
        )
        accuracy = n_correct / self.n_total

        if reset:
            self.reset()

        return accuracy

    def reset(self) -> None:
        self.predictions = []
        self.gold_labels = []
