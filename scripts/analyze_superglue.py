from collections import Counter
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from tqdm import tqdm

from meta_learn_prompt.data.config import Config
from meta_learn_prompt.data.prompt_data_module import PromptDataModule
import meta_learn_prompt.data.super_glue_data_module
from meta_learn_prompt.data.templates import templatize, get_possible_labels


def main(dataset, template_idx=0, label_idx=0):
    dataset_class = PromptDataModule.by_name(dataset)
    data_module = dataset_class(
        num_prefix=20,
        template_idx=template_idx,
        soft_only=False,
        direct_model=False,
        transformer_model="gpt2",
        config=Config(),
        data_dir="tmp_analysis",
    )
    dataset_dict = data_module.load()
    tokenizer = data_module.setup_tokenizer()
    counters = {k: Counter() for k in dataset_dict.keys()}
    for split, examples in dataset_dict.items():
        for example in tqdm(examples):
            label = get_possible_labels(dataset, example)[label_idx]
            prefix, input = templatize(dataset, template_idx, example, label)
            tokenized_input = tokenizer(input)["input_ids"]
            length = len(tokenized_input)
            counters[split][length] += 1

    for split, counter in counters.items():
        print(split)
        print(sorted(counter.items()))
        for threshold in (256, 512, 768):
            n_exceeded = sum(v for k, v in counter.items() if k > threshold)
            n_total = sum(v for v in counter.values())
            print(f"At {threshold}, {n_exceeded}/{n_total}={n_exceeded/n_total}")


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
