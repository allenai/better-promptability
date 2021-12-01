from multiprocessing import Pool
import sys
from typing import NamedTuple

import datasets
from tqdm import tqdm


class LoadInput(NamedTuple):
    task_name: str
    cache_dir: str


def load(input: LoadInput) -> str:
    datasets.load_dataset("bigscience/P3", input.task_name, cache_dir=input.cache_dir)
    return input.task_name


def main(mixture_name: str, cache_dir: str):
    tasks = [line.strip() for line in open(f"data/{mixture_name}_tasks.txt")]
    pool = Pool()
    with tqdm(
        pool.imap_unordered(
            load,
            [
                LoadInput(task_name=task, cache_dir=cache_dir)
                for task in tasks
                if task != "story_cloze"
            ],
        )
    ) as result_iter:
        for task_name in result_iter:
            result_iter.set_postfix({"finished": task_name})


if __name__ == "__main__":
    main(*sys.argv[1:])
