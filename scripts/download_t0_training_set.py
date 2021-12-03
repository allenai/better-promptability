"""
Download all of the data from the [bigscience/P3](https://huggingface.co/datasets/bigscience/P3)
corresponding to a particular mixture. This script should only be run from the root of this repository.
"""

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import datasets

STORY_CLOZE_PATH = Path("/net/nfs2.allennlp/zhaofengw/story_cloze_dir")


async def track(tasks: List[str], queue):
    completed = 0
    while completed < len(tasks):
        task_name, returncode, stdout, stderr = await queue.get()
        completed += 1
        if returncode != 0:
            print(f"Failed to download '{task_name}':\n[stdout]\n{stdout}\n[stderr]\n{stderr}")
        else:
            print(f"[{completed}/{len(tasks)}] download for '{task_name}' complete")
        queue.task_done()


async def run(task_name: str, cmd: str, queue) -> int:
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    assert proc.returncode is not None
    queue.put_nowait(
        (
            task_name,
            proc.returncode,
            stdout.decode() if stdout else "",
            stderr.decode() if stderr else "",
        )
    )
    return proc.returncode


async def main(mixture_name: str, cache_dir: str, task: Optional[str] = None):
    cache_dir = Path(cache_dir)

    def download_task_dataset(task_name: str):
        local_path = cache_dir / task_name  # type: ignore
        if not os.path.isdir(local_path) or not os.listdir(local_path):
            if task_name.startswith("story_cloze_"):
                data_dir = STORY_CLOZE_PATH / task_name
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
                    split_info = json.load(open(data_dir / f"info.{split_name}.json"))
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
                    split_name: {"tfrecord": data_dir / f"{split_name}.tfrecord-00000-of-00001"}
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

            dataset = datasets.load_dataset(
                "bigscience/P3", task_name, cache_dir=cache_dir / ".hf_cache"  # type: ignore
            )
            dataset.save_to_disk(local_path)

    tasks = [line.strip() for line in open(f"data/{mixture_name}_tasks.txt")]

    exitcode = 0

    if task is not None:
        assert task in tasks
        download_task_dataset(task)
    else:
        queue: asyncio.Queue = asyncio.Queue()
        results = await asyncio.gather(
            track(tasks, queue),
            *[
                run(
                    task_name,
                    f"python scripts/download_t0_training_set.py '{mixture_name}' '{cache_dir}' '{task_name}'",
                    queue,
                )
                for task_name in tasks
            ],
        )
        failures = 0
        for returncode in results[1:]:
            if returncode != 0:
                failures += 1
                exitcode = 1

    if failures:
        print(f"There were {failures} task datasets that failed to download")

    sys.exit(exitcode)


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
