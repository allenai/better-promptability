from __future__ import annotations
from typing import Optional

from tango.common import PathOrStr
from torch.utils.data.dataloader import DataLoader
from transformers.trainer_pt_utils import LengthGroupedSampler, DistributedLengthGroupedSampler

from .config import Config
from .data_utils import collate_fn
from .mixer_dataloader import MixerDataLoader
from .prompt_data_module import PromptDataModule
from .t0_multitask_data_module import T0MultiTaskDataModule


@PromptDataModule.register("t0_meta_learning")
class T0MetaLearningDataModule(T0MultiTaskDataModule):
    def __init__(
        self,
        meta_batch_size: int,
        mixture_name: str,  # should be 'd4_train', 'd4_dev', or 'green'.
        config: Config,
        num_prefix: int,
        transformer_model: PathOrStr,
        sampling_cap: Optional[int] = 500000,
        **kwargs
    ):
        self.meta_batch_size = meta_batch_size
        super().__init__(
            mixture_name, config, num_prefix, transformer_model, sampling_cap=sampling_cap, **kwargs
        )

    def dataloader(self, split: str, batch_size: int, shuffle=False) -> DataLoader:
        dataset_split = self.dataset_dict[split]
        pad_token_map = self.pad_token_map(split)
        assert all(pad is not None for pad in pad_token_map.values())

        dataloaders = []
        for dataset in dataset_split._datasets:
            # zhaofeng: I don't particularly like this design because of the redundancy with
            # DataModule. But this is necessary at least to accomodate _UndersampledDataset at the
            # moment, unless we can somehow turn it into a DataModule too.
            if shuffle:
                # LengthGroupedSampler sorts from longest to shortest; we want the reverse
                lens = [-len(ids) for ids in dataset[self.sort_key]]
                if self.config.gpus is None or self.config.gpus <= 1:
                    sampler = LengthGroupedSampler(batch_size, lengths=lens)
                else:
                    sampler = DistributedLengthGroupedSampler(batch_size, lengths=lens)
            else:
                sampler = None
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                samlper=sampler,
                collate_fn=lambda batch: collate_fn(
                    batch, pad_token_map, self.tokenizer.padding_side
                ),
                pin_memory=True,
            )
            dataloaders.append(dataloader)

        return MixerDataLoader(dataloaders, self.meta_batch_size)
