from __future__ import annotations
import random
from typing import Optional

from datasets import Dataset as HFDataset
from tango.common import PathOrStr, Tqdm
from torch.utils.data.dataloader import DataLoader
from transformers.trainer_pt_utils import LengthGroupedSampler

from .config import Config
from .data_utils import collate_fn
from .mixer_dataloader import MixerDataLoader
from .mixer_dataset import _UndersampledDataset
from .prompt_data_module import PromptDataModule
from .t0_multitask_data_module import T0MultiTaskDataModule


def split_batch(meta_batch: list, support_batch_size: int) -> list:
    # Because each batch is internally sorted by length, the naive split will cause a distributional
    # difference.
    processed_meta_batch = []
    for batch in meta_batch:
        batch_size = len(list(batch.values())[0])
        assert all(len(v) == batch_size for v in batch.values())
        support_indices = random.sample(range(batch_size), support_batch_size)
        support_indices_set = set(support_indices)
        query_indices = [i for i in range(batch_size) if i not in support_indices_set]

        support_batch = {k: v[support_indices] for k, v in batch.items()}
        query_batch = {k: v[query_indices] for k, v in batch.items()}
        processed_meta_batch.append((support_batch, query_batch))
    return processed_meta_batch


@PromptDataModule.register("t0_meta_learning")
class T0MetaLearningDataModule(T0MultiTaskDataModule):
    def __init__(
        self,
        meta_batch_size: int,
        support_batch_size: int,
        mixture_name: str,  # should be 'd4_train', 'd4_dev', or 'green'.
        config: Config,
        num_prefix: int,
        transformer_model: PathOrStr,
        sampling_cap: Optional[int] = 500000,
        **kwargs
    ):
        self.meta_batch_size = meta_batch_size
        self.support_batch_size = support_batch_size
        super().__init__(
            mixture_name, config, num_prefix, transformer_model, sampling_cap=sampling_cap, **kwargs
        )

    def dataloader(self, split: str, batch_size: int, shuffle=False) -> DataLoader:
        if split != "train":
            return super().dataloader(split, batch_size, shuffle=shuffle)

        dataset_split = self.dataset_dict[split]
        pad_token_map = self.pad_token_map(split)
        assert all(pad is not None for pad in pad_token_map.values())

        dataloaders = []
        for dataset in Tqdm.tqdm(dataset_split._datasets, desc="Creating dataloaders"):
            # zhaofeng: I don't particularly like this design because of the redundancy with
            # DataModule. But this is necessary at least to accomodate _UndersampledDataset at the
            # moment, unless we can somehow turn it into a DataModule too.
            assert shuffle
            if isinstance(dataset, HFDataset):
                lens = dataset["sort_key_len"]
            elif isinstance(dataset, _UndersampledDataset):
                lens = dataset.get_active_example_lens()
            else:
                assert False
            # LengthGroupedSampler sorts from longest to shortest; we want the reverse
            lens = [-l for l in lens]  # noqa: E741
            # It's important we don't used the distributed sampler here since distributed logic
            # is handled in MixerDataloader
            sampler = LengthGroupedSampler(batch_size, lengths=lens)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=0,  # otherwise too many open files are opened
                collate_fn=lambda batch: collate_fn(
                    batch, pad_token_map, self.tokenizer.padding_side
                ),
                pin_memory=True,
                drop_last=True,  # division into support/query is unclear with incomplete batches
            )
            dataloaders.append(dataloader)

        return MixerDataLoader(
            dataloaders,
            self.meta_batch_size,
            batch_postprocessor=lambda b: split_batch(b, self.support_batch_size),
        )
