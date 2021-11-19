"""
Subsamples the training set for each dataset (i.e., for all tepmlates).
"""

from pathlib import Path
import pickle
import sys
import random

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from meta_learn_prompt.data.config import Config  # noqa: E402
from meta_learn_prompt.data.data_utils import md5  # noqa: E402
from meta_learn_prompt.data.t0_data_module import T0Mixture  # noqa: E402


def main(n_shot, seed, output_file):
    n_shot = int(n_shot)
    seed = int(seed)
    random.seed(seed)

    # All arguments apart from the first are dummy
    mixture = T0Mixture(
        mixture_name="green",
        num_prefix=20,
        transformer_model="t5-base",
        config=Config(),
        data_dir="tmp",
    )
    dataset_to_indices = {}
    for data_module in tqdm(mixture.data_modules.values()):
        dataset_id = (data_module.dataset_name, data_module.subset_name)
        if dataset_id in dataset_to_indices:
            continue  # already sampled

        dataset_dict = data_module.load()
        train_split = dataset_dict[data_module.train_split]
        total_len = len(train_split)
        print(f"Sampling {n_shot} examples from {total_len} for {dataset_id} with seed {seed}")
        indices = random.sample(range(total_len), n_shot)
        checksum = md5("".join(str(sorted(train_split[i].items())) for i in indices))
        dataset_to_indices[dataset_id] = (indices, checksum)

    pickle.dump(dataset_to_indices, open(output_file, "wb"))


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
