import sys

import numpy as np
from tqdm import tqdm

from meta_learn_prompt.data.config import Config  # noqa: E402
from meta_learn_prompt.data.t0_mixture import T0Mixture  # noqa: E402


def main(mixture_name):
    # All arguments apart from the first two are dummy
    mixture = T0Mixture(
        mixture_name=mixture_name,
        t0_data_cache="/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache",
        config=Config(),
        data_dir="tmp",
        num_prefix=20,
        transformer_model="t5-base",
    )
    agg = {}
    for data_module in tqdm(mixture.data_modules.values()):
        data_module.setup()
        acc_total = 0
        for ex in data_module["dev"]:
            acc_total += 1 / sum(ex["is_correct_mask"])

        agg_key = (data_module.dataset_name, data_module.subset_name)
        if agg_key not in agg:
            agg[agg_key] = []
        agg[agg_key].append(acc_total / len(data_module["dev"]))

    for k, v in agg.items():
        print(k, np.mean(v), np.std(v))


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
