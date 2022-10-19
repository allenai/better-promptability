"""
Subsamples the training set for each dataset (i.e., for all tepmlates).
Ideally we want to sample the same examples across templates for a given dataset, but unfortunately
this is impossible since the P3 dataset cache does not guarantee the same example order across
templates. Check out, for example, hellaswag_complete_first_then_score_eval[29372] and
hellaswag_Predict_ending_with_hint_score_eval[29372].
"""

from pathlib import Path
import pickle
import sys
import random

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from better_promptability.data.config import Config  # noqa: E402
from better_promptability.data.data_utils import md5  # noqa: E402
from better_promptability.data.t0_mixture import T0Mixture  # noqa: E402


def main(mixture_name, n_shot, seed, output_file):
    n_shot = int(n_shot)
    seed = int(seed)
    random.seed(seed)

    # All arguments apart from the first two are dummy
    mixture = T0Mixture(
        mixture_name=mixture_name,
        t0_data_cache="/net/nfs2.allennlp/akshitab/better-promptability/t0/processed_cache",
        config=Config(),
        data_dir="tmp",
        num_prefix=20,
        transformer_model="t5-base",
    )
    taskname_to_indices = {}
    for data_module in tqdm(mixture.data_modules.values()):
        task_name = data_module.task_name
        dataset_dict = data_module.load()
        train_split = dataset_dict[data_module.train_split]
        total_len = len(train_split)
        print(f"Sampling {n_shot} examples from {total_len} for {task_name} with seed {seed}")
        indices = random.sample(range(total_len), n_shot)
        checksum = md5(
            "".join(str(train_split[i]["inputs"] + train_split[i]["targets"]) for i in indices)
        )
        taskname_to_indices[task_name] = (indices, checksum)

    pickle.dump(taskname_to_indices, open(output_file, "wb"))


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
