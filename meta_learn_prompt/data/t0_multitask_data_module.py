from typing import Optional

from tango.common import Tqdm, DatasetDict

from .mixer_dataset import MixerDataset
from .prompt_data_module import PromptDataModule
from .t0_data_module import T0Mixture


@PromptDataModule.register("t0_multitask")
class T0MultiTaskDataModule(PromptDataModule):
    def __init__(
        self,
        mixture_name: str,  # should be 'd4_train', 'd4_dev', or 'green'.
        *args,
        sampling_cap: Optional[int] = 500000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.t0_mixture = T0Mixture(mixture_name, *args, **kwargs)
        self.sampling_cap = sampling_cap

    def setup(self, stage: Optional[str] = None):
        with Tqdm.tqdm(
            self.t0_mixture.data_modules.items(), "Preprocessing T0 datasets"
        ) as dm_iter:
            for name, data_module in dm_iter:
                dm_iter.set_postfix({"module": name})
                data_module.setup()
        super().setup(stage)

    def load(self) -> DatasetDict:
        return DatasetDict(
            splits={
                "train": MixerDataset(
                    [dm[dm.train_split] for dm in self.t0_mixture.data_modules.values()],
                    sampling_cap=self.sampling_cap,
                ),
                "dev": MixerDataset(
                    [
                        dm[dev_split]
                        for dm in self.t0_mixture.data_modules.values()
                        for dev_split in dm.dev_splits
                    ],
                ),
            }
        )
