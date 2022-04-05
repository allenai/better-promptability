from typing import Union, Dict, Any, Set, Optional

import catwalk.task
import catwalk.tasks
import datasets
from datasets import DatasetDict
from tango import Step
from tango.common import Tqdm
from tango.common.sequences import MappedSequence
from tango.integrations.datasets import DatasetsFormat


@Step.register("catwalk::seq2seq_training_data")
class Seq2SeqTrainingDataStep(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    VERSION = "003"

    FORMAT = DatasetsFormat

    def massage_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(kwargs["task"], str):
            kwargs["task"] = catwalk.tasks.TASKS[kwargs["task"]]
        if kwargs["splits"] is None:
            kwargs["splits"] = {"train", "test", "validation"}
        return kwargs

    def run(
        self,
        task: Union[str, catwalk.task.Task],
        splits: Optional[Set[str]] = None,
    ) -> DatasetDict:
        if isinstance(task, str):
            task = catwalk.tasks.TASKS[task]
        if splits is None:
            splits = {"train", "test", "validation"}

        datasets_features = datasets.Features(
            {"source": datasets.Value("string"), "target": datasets.Value("string")}
        )

        result = {}
        for split in splits:
            instances = task.get_split(split)
            instances = MappedSequence(
                task.instance_conversions[catwalk.task.InstanceFormat.T5_PROMPT], instances
            )
            new_instances = datasets.Dataset.from_dict(
                {"source": [], "target": []}, features=datasets_features, split=split
            )
            for instance in Tqdm.tqdm(instances, f"Processing split {split}"):
                new_instances = new_instances.add_item(
                    {"source": instance[0], "target": instance[1]}
                )
            result[split] = new_instances

        return DatasetDict(result)
