from typing import Any

import datasets
from tango import Step


@Step.register("preprocess_super_glue")
class PreprocessSuperGlue(Step):
    """
    A step that processes SuperGLUE datasets into a format that is amenable to templatizing.
    """

    DETERMINISTIC = True
    CACHEABLE = False  # we'll use 'datasets' caching mechanism, not Tango's.

    def run(self, dataset_dict: datasets.DatasetDict) -> datasets.DatasetDict:
        for name, dataset in dataset_dict.items():
            task, split = name.split("_")
            dataset_dict[name] = dataset.map(
                lambda examples: self.map(task, split, examples),
                batched=True,
                remove_columns=dataset.column_names,
            )  # these are needed to add rows; see https://huggingface.co/docs/datasets/process.html
        return dataset_dict

    def map(self, task: str, split: str, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if task == "copa" and split == "train":
            # data augmentation: mirror choice1 and choice2
            for field in set(examples.keys()) - {"choice1", "choice2"}:
                examples[field].extend([x for x in examples[field]])
            examples["choice1"].extend([x for x in examples["choice2"]])
            examples["choice2"].extend([x for x in examples["choice1"]])
        elif task == "wic" and split == "train":
            # data augmentation: mirror sentence1 and sentence2
            for field in set(examples.keys()) - {"sentence1", "sentence2"}:
                examples[field].extend([x for x in examples[field]])
            examples["sentence1"].extend([x for x in examples["sentence2"]])
            examples["sentence2"].extend([x for x in examples["sentence1"]])
        elif task == "record":
            # we follow the GPT-3 paper wrt @highlight annotations
            examples["passage"] = [p.replace("@highlight\n", "- ") for p in examples["passage"]]
            if split == "train":
                # create one example per answer
                new_examples = {k if k != "answers" else "answer": [] for k in examples.keys()}
                for i, answers in examples["answers"]:
                    for answer in answers:
                        new_examples["answer"].append(answer)
                        for field in set(examples.keys()) - {"answers"}:
                            new_examples[field].append(examples[field][i])
                examples = new_examples
        else:
            return examples
