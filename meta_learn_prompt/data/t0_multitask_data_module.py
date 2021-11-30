from typing import Optional, Union

from datasets import DatasetDict, IterableDatasetDict
from tango.common import Tqdm

from .data_utils import MixerStreamDataset
from .prompt_data_module import PromptDataModule
from .t0_data_module import T0Mixture


@PromptDataModule.register("t0_multitask")
class T0MultiTaskDataModule(PromptDataModule):
    def __init__(
        self,
        mixture_name: str,  # should be 'd4_train' or 'green' most of the time
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.t0_mixture = T0Mixture(mixture_name, *args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        with Tqdm.tqdm(
            self.t0_mixture.data_modules.items(), "Preprocessing T0 datasets"
        ) as dm_iter:
            for name, data_module in dm_iter:
                dm_iter.set_postfix({"module": name})
                data_module.setup()
        super().setup(stage)

    def load(self) -> Union[DatasetDict, IterableDatasetDict]:
        return IterableDatasetDict(
            {
                "train": MixerStreamDataset(
                    [dm[dm.train_split] for dm in self.t0_mixture.data_modules.values()]
                ),
                "dev": MixerStreamDataset(
                    [
                        dm[dev_split]
                        for dm in self.t0_mixture.data_modules.values()
                        for dev_split in dm.dev_splits
                    ]
                ),
            }
        )

    def preprocess(self, dataset_dict: DatasetDict) -> DatasetDict:
        return dataset_dict
