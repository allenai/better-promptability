import logging
import os
import sys

from tango.common import Params
from tqdm import tqdm

from better_promptability.steps.process_dataset import ProcessDataset
from better_promptability.steps.process_story_cloze import ProcessStoryCloze


logging.basicConfig(level=logging.INFO)


def process_green_datasets(old_base_path, new_base_path):
    datasets = Params.from_file("configs/t0_mixtures.jsonnet")["green"]
    for dataset in tqdm(datasets):
        dataset = dataset.strip()
        if "story_cloze" not in dataset:
            step = ProcessDataset()
        else:
            step = ProcessStoryCloze()
        try:
            step.run(
                old_data_path=os.path.join(old_base_path, dataset),
                new_data_path=os.path.join(new_base_path, dataset),
            )
        except KeyError:
            print(f"error in {dataset}")


if __name__ == "__main__":
    process_green_datasets(sys.argv[1], sys.argv[2])
