"""
Download all of the data from the [bigscience/P3](https://huggingface.co/datasets/bigscience/P3)
corresponding to a particular mixture. This script should only be run from the root of this repository.
"""

import asyncio
import os
import sys
from typing import Optional, List

import datasets


async def track(tasks: List[str], queue):
    completed = 0
    while completed < len(tasks):
        task_name, returncode, stdout, stderr = await queue.get()
        completed += 1
        if returncode != 0:
            print(f"Failed to download '{task_name}':\n[stdout]\n{stdout}\n[stderr]\n{stderr}")
        else:
            print(f"[{completed}/len(tasks)] download for '{task_name}' complete")
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
        for returncode in results[1:]:
            if returncode != 0:
                exitcode = 1

    sys.exit(exitcode)


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
