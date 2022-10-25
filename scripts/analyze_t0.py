"""

"""

from pathlib import Path
import random
import sys

from datasets import Dataset as HFDataset
from datasets.utils import set_progress_bar_enabled
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from meta_learn_prompt.data.config import Config  # noqa: E402
from meta_learn_prompt.data.mixer_dataset import _UndersampledDataset
from meta_learn_prompt.data.t0_mixture import T0Mixture  # noqa: E402
from meta_learn_prompt.data.t0_multitask_data_module import T0MultiTaskDataModule  # noqa: E402

N_SAMPLE = 100_000
# N_SAMPLE = 10_000
MAX_INPUTS_LEN = 768
MAX_TARGETS_LEN = 192

set_progress_bar_enabled(False)


def main(max_inputs_len=MAX_INPUTS_LEN, max_targets_len=MAX_TARGETS_LEN):
    max_inputs_len = int(max_inputs_len)
    max_targets_len = int(max_targets_len)

    # All arguments apart from the first two are dummy
    # mixture = T0Mixture(
    #     mixture_name="d4_train",
    #     t0_data_cache="/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache",
    #     config=Config(),
    #     data_dir="tmp",
    #     num_prefix=20,
    #     transformer_model="t5-base",
    # )
    # datasets = [
    #     (name, data_module.load()["train"])
    #     for name, data_module in tqdm(mixture.data_modules.items())
    #        # if "validation" in data_module.load()
    # ]
    mtl_module = T0MultiTaskDataModule(
        mixture_name="d4_train",
        config=Config(),
        num_prefix=20,
        transformer_model="t5-base",
        t0_data_cache="/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache",
        data_dir="tmp",
    )
    # datasets = [
    #     (name, data_module.load()["validation"])
    #     for name, data_module in tqdm(mixture.data_modules.items())
    #     if "validation" in data_module.load()
    # ]

    # curr_inputs_len = 0
    # curr_targets_len = 0
    # n_sequences = 1  # for the final one
    # for _ in tqdm(range(N_SAMPLE)):
    #     _, dataset = random.choice(datasets)
    #     e = random.choice(dataset)
    #     curr_inputs_len += len(e["inputs"])
    #     curr_targets_len += len(e["targets"])
    #     if curr_inputs_len > MAX_INPUTS_LEN or curr_targets_len > MAX_TARGETS_LEN:
    #         curr_inputs_len = len(e["inputs"])
    #         curr_targets_len = len(e["targets"])
    #         n_sequences += 1
    # print(N_SAMPLE, n_sequences)

    # ---------

    # total_too_long = 0
    # total = 0
    # n_exceeded = 0
    # n_exceeded_1p = 0
    # for name, dataset in tqdm(datasets):
    #     # breakpoint()
    #     too_long = dataset.filter(
    #         lambda e: len(e["inputs"]) > max_inputs_len or len(e["targets"]) > max_targets_len,
    #         num_proc=8,
    #     )
    #     if len(too_long) > 0:
    #         print(
    #             f"\n{name} {len(too_long)}/{len(dataset)}={len(too_long)/len(dataset):.2f} too long"
    #         )
    #         n_exceeded += 1
    #     if len(too_long) / len(dataset) >= 0.01:
    #         n_exceeded_1p += 1

    #     total_too_long += len(too_long)
    #     total += len(dataset)

    # print(f"\nAt {max_inputs_len} X {max_targets_len}")
    # print(f"Out of {len(datasets)}, {n_exceeded} datasets have long examples, {n_exceeded_1p} has more than 1%")
    # print(f"Tottal: {total_too_long}/{total}={total_too_long/total:.2f} too long")

    # ---------

    # lengths = []
    # for name, dataset in tqdm(datasets):
    #     dataset = dataset.map(
    #         lambda e: {"len": min(len(e["inputs"]), MAX_INPUTS_LEN)},
    #         batched=False,
    #         num_proc=16,
    #         remove_columns=dataset.column_names
    #     )
    #     lens = dataset["len"]
    #     lengths.extend(lens)
    #     print(f"{name}: {np.mean(lens)} +- {np.std(lens)}")
    #     # for e in dataset:
    #     #     dataset = dataset.map(
    #     #         lambda e: {"len": min(len(e["inputs"]), MAX_INPUTS_LEN)},
    #     #         batched=False,
    #     #         num_proc=8,
    #     #     )
    #     #     lengths.extend(dataset["len"])
    # for i in range(10, 100, 5):
    #     print(np.percentile(lengths, i))

    dataset_dict = mtl_module.load()
    src_lens = []  # dataset_dict["train"].get_all_example_lens()
    tgt_lens = []
    for dataset in tqdm(dataset_dict["train"]._datasets, desc="Getting lengths for sampler"):
        if isinstance(dataset, _UndersampledDataset):
            dataset = dataset._dataset.select(dataset._indices[: dataset._sampling_cap])
        dataset = dataset.map(
            lambda e: {
                "src_len": min(len(e["inputs"]), MAX_INPUTS_LEN),
                "tgt_len": min(len(e["inputs"]), MAX_TARGETS_LEN),
            },
            batched=False,
            num_proc=16,
            remove_columns=dataset.column_names,
        )
        src_lens.extend(dataset["src_len"])
        tgt_lens.extend(dataset["tgt_len"])
    print(max(src_lens), max(tgt_lens), sum(src_lens), sum(tgt_lens))
    breakpoint()


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
