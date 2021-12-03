import json
import importlib
import sys
from pathlib import Path

import datasets


def main(input_dir: str, cache_dir: str):
    input_dir = Path(input_dir)
    for path in input_dir.iterdir():
        path = path.resolve()
        if path.is_dir():
            task_name = path.name

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
            assert task_name not in task_splits_and_features
            for split_name in ("validation", "test"):  # story cloze has no training set
                split_info = json.load(open(path / f"info.{split_name}.json"))
                features_dict = split_info["features"]
                assert split_info["num_shards"] == 1

                if task_name not in task_splits_and_features:
                    task_splits_and_features[task_name] = {
                        "splits": [],
                        "features_dict": features_dict,
                    }
                task_splits_and_features[task_name]["splits"].append(split_name)
                assert features_dict == task_splits_and_features[task_name]["features_dict"]
            splits_and_features_dict = task_splits_and_features[task_name]

            assert task_name not in p3_module._URLs  # type: ignore
            p3_module._URLs[task_name] = {  # type: ignore
                split_name: {"tfrecord": path / f"{split_name}.tfrecord-00000-of-00001"}
                for split_name in splits_and_features_dict["splits"]
            }

            p3_module.P3.BUILDER_CONFIGS.append(  # type: ignore
                p3_module.P3Config(  # type: ignore
                    name=task_name,
                    splits=splits_and_features_dict["splits"],
                    features_dict=splits_and_features_dict["features_dict"],
                    score_eval=task_name.endswith("score_eval"),
                )
            )
            p3_module.P3.builder_configs = {  # type: ignore
                config.name: config for config in p3_module.P3.BUILDER_CONFIGS  # type: ignore
            }

            print(f"Loading {task_name}")
            d = datasets.load_dataset("bigscience/P3", task_name)
            print(f"Caching {task_name}")
            d.save_to_disk(Path(cache_dir) / task_name)


if __name__ == "__main__":
    main(*sys.argv[1:])
