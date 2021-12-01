import sys

from tqdm import tqdm

from meta_learn_prompt.data.config import Config
from meta_learn_prompt.data.t0_data_module import T0Mixture


def main(mixture_name: str, cache_dir: str):
    assert mixture_name in {"green", "d4_train"}
    mixture = T0Mixture(
        mixture_name=mixture_name,
        cache_dir=cache_dir,
        num_prefix=20,
        transformer_model="t5-base",
        config=Config(),
        data_dir="tmp",
    )
    with tqdm(mixture.data_modules.items()) as dm_iter:
        for name, data_module in dm_iter:
            dm_iter.set_postfix({"module": name})
            data_module.load()


if __name__ == "__main__":
    main(*sys.argv[1:])
