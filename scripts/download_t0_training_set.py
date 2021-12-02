import os
import sys

import datasets
from tqdm import tqdm


def main(mixture_name: str, cache_dir: str):
    tasks = [line.strip() for line in open(f"data/{mixture_name}_tasks.txt")]
    # I tried to parallelize this but the loading script for this dataset uses multiprocessing
    # itself, which causes errors when we try to use a multiprocessing Pool. There is probably
    # a way around this, but it didn't seem worth anymore development time at the moment since
    # we probably won't have to run this script more than once.
    with tqdm(tasks) as task_iter:
        for task_name in task_iter:
            task_iter.set_postfix({"downloading": task_name})
            dataset = datasets.load_dataset("bigscience/P3", task_name, cache_dir=cache_dir)
            dataset.save_to_disk(os.path.join(cache_dir, task_name))


if __name__ == "__main__":
    main(*sys.argv[1:])
