"""
Download all of the data from the [bigscience/P3](https://huggingface.co/datasets/bigscience/P3)
corresponding to a particular mixture. This script should only be run from the root of this repository.
"""

import asyncio
import os
import sys
from typing import Optional, Tuple

import datasets


async def run(task_name: str, cmd: str) -> Tuple[int, str, str]:
    print(f"Starting download for {task_name}")
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await proc.communicate()
    print(f"Finished download for {task_name}")
    assert proc.returncode is not None
    return proc.returncode, stdout.decode() if stdout else "", stderr.decode() if stderr else ""


async def main(mixture_name: str, cache_dir: str, task: Optional[str] = None):
    def download_task_dataset(task_name: str):
        local_path = os.path.join(cache_dir, task_name)
        if not os.path.isdir(local_path) or not os.listdir(local_path):
            dataset = datasets.load_dataset("bigscience/P3", task_name, cache_dir=cache_dir)
            dataset.save_to_disk(local_path)

    tasks = [
        line.strip()
        for line in open(f"data/{mixture_name}_tasks.txt")
        if not line.startswith("story_cloze_")  # these are handled separately
    ]

    exitcode = 0

    if task is not None:
        assert task in tasks
        download_task_dataset(task)
    else:
        results = await asyncio.gather(
            *[
                run(
                    task_name,
                    f"python scripts/download_t0_training_set.py '{mixture_name}' '{cache_dir}' '{task_name}'",
                )
                for task_name in tasks
            ]
        )
        for task_name, (returncode, stdout, stderr) in zip(tasks, results):
            if returncode != 0:
                print(f"Failed to download '{task_name}':\n[stdout]\n{stdout}\n[stderr]\n{stderr}")
                exitcode = 1

    sys.exit(exitcode)


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
