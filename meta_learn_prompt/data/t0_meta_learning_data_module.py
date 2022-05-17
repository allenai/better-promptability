from __future__ import annotations
import random
from typing import Optional, Mapping

from datasets import Dataset as HFDataset
from tango.common import PathOrStr, Tqdm
import torch
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from transformers.trainer_pt_utils import LengthGroupedSampler

from .config import Config
from .data_utils import collate_fn as default_collate_fn, PAD_TYPE
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
        instance_level_mixing: bool = False,
        **kwargs
    ):
        self.meta_batch_size = meta_batch_size
        self._meta_batch_size_per_device = self.meta_batch_size // (
            dist.get_world_size() if dist.is_initialized() else 1
        )
        self.support_batch_size = support_batch_size
        self.instance_level_mixing = instance_level_mixing
        if self.instance_level_mixing:
            self.real_batch_size = kwargs["batch_size"]
            kwargs["batch_size"] *= self._meta_batch_size_per_device
            kwargs["num_workers"] = 0  # avoid too many open files error
        super().__init__(
            mixture_name, config, num_prefix, transformer_model, sampling_cap=sampling_cap, **kwargs
        )

    def collate_fn(
        self, batch: list[dict[str, list]], pad_token_map: Mapping[str, PAD_TYPE], padding_side: str
    ) -> list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        batch = [
            collate_fn(batch[i : i + self.real_batch_size], pad_token_map, padding_side)
            for i in range(0, len(batch), self.real_batch_size)
        ]
        if len(batch[-1]["input_ids"]) < self.real_batch_size:
            batch = batch[:-1]
        return split_batch(batch, self.support_batch_size)

    def dataloader(
        self, split: str, batch_size: int, shuffle=False, collate_fn=default_collate_fn
    ) -> DataLoader:
        if split != "train":
            return super().dataloader(split, batch_size, shuffle=shuffle)
        if self.instance_level_mixing:
            assert collate_fn is None
            return super().dataloader(
                split, batch_size, shuffle=shuffle, collate_fn=self.collate_fn
            )

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
                num_workers=0,  # avoid too many open files error
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
