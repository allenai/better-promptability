from typing import Optional

from tango.common import Tqdm, DatasetDict, PathOrStr

from .config import Config
from .mixer_dataset import MixerDataset
from .prompt_data_module import PromptDataModule
from .t0_mixture import T0Mixture


@PromptDataModule.register("t0_multitask")
class T0MultiTaskDataModule(PromptDataModule):
    def __init__(
        self,
        mixture_name: str,  # should be 'd4_train', 'd4_dev', or 'green'.
        config: Config,
        num_prefix: int,
        transformer_model: PathOrStr,
        sampling_cap: Optional[int] = 500000,
        t0_data_cache: PathOrStr = "/net/nfs2.allennlp/petew/meta-learn-prompt/t0/cache",
        **kwargs,
    ):
        super().__init__(config, num_prefix, transformer_model, preprocess_and_save=False, **kwargs)
        self.t0_mixture = T0Mixture(
            mixture_name,
            config,
            num_prefix,
            transformer_model,
            t0_data_cache=t0_data_cache,
            **kwargs,
        )
        self.sampling_cap = sampling_cap

    def load(self) -> DatasetDict:
        with Tqdm.tqdm(self.t0_mixture.data_modules.items(), "Loading T0 datasets") as dm_iter:
            for name, data_module in dm_iter:
                dm_iter.set_postfix({"module": name if len(name) < 30 else (name[:27] + "...")})
                data_module.tokenizer = self.tokenizer
                data_module.task_token_ids = self.task_token_ids
                data_module.setup()

        return DatasetDict(
            splits={
                "dev": MixerDataset(
                    [
                        dm[dev_split]
                        for dm in self.t0_mixture.data_modules.values()
                        for dev_split in dm.dev_splits
                    ],
                ),
                "train": MixerDataset(
                    [dm[dm.train_split] for dm in self.t0_mixture.data_modules.values()],
                    sampling_cap=self.sampling_cap,
                ),
            }
        )
