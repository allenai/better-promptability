import os
from meta_learn_prompt.steps.process_dataset import ProcessDataset


def process_green_datasets(old_base_path, new_base_path):

    with open("data/green_tasks.txt") as f:
        datasets = f.readlines()

    for dataset in datasets:
        step = ProcessDataset()
        result = step.run(
            old_data_path=os.path.join(old_base_path, dataset),
            new_data_path=os.path.join(new_base_path, dataset),
        )


if __name__ == "__main__":
    old_base_path = "/net/nfs2.allennlp/petew/meta-learn-prompt/t0/cache/"
    new_base_path = "/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache/"

    process_green_datasets(old_base_path, new_base_path)
