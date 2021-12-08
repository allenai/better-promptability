
import sys
import datasets
import os

def check_multiple_answers(dataset):
    # Only checking the validation set.
    total_actual_instances = dataset['validation']['idx'][-1][0] + 1
    total_correct_answers = sum(dataset['validation']['is_correct'])

    assert total_actual_instances == total_correct_answers

def check_all_datasets(folder_path):
    multiple_answers = []
    for dirname in os.listdir(folder_path):
        if dirname.endswith("score_eval") and "story_cloze" not in dirname:
            print("Checking", dirname)
            dataset = datasets.load_from_disk(os.path.join(folder_path, dirname))
            try:
                check_multiple_answers(dataset)
            except AssertionError:
                multiple_answers.append(dirname)
    print("Datasets with multiple right answers: ", multiple_answers)

if __name__ == '__main__':
    check_all_datasets(sys.argv[1])
