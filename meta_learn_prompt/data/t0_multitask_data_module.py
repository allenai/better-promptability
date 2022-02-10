from __future__ import annotations
from typing import Optional, Mapping, Any

from tango.common import Tqdm, DatasetDict, PathOrStr

from .data_utils import PAD_TYPE
from .config import Config
from .mixer_dataset import MixerDataset, _UndersampledDataset
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
        self.mixture_name = mixture_name
        self.t0_mixture = T0Mixture(
            mixture_name,
            config,
            num_prefix,
            transformer_model,
            t0_data_cache=t0_data_cache,
            **kwargs,
        )
        self.sampling_cap = sampling_cap

    @property
    def hash_fields(self) -> list[Any]:
        return super().hash_fields + [
            self.mixture_name,
            self.sampling_cap,
        ]

    @property
    def dev_splits(self) -> list[str]:
        return ["dev"]

    @property
    def test_splits(self) -> list[str]:
        # We don't need the test sets. The test set labels of some datasets are hidden
        # (e.g., superglue), and T0 only evaluated on the dev sets.
        return []

    @property
    def metric_names(self) -> list[str]:
        return ["categorical_accuracy"]

    @property
    def metric_watch_mode(self) -> str:
        return "max"

    @property
    def sort_key(self) -> str:
        return "inputs"

    def pad_token_map(self, split: str) -> Mapping[str, PAD_TYPE]:  # type: ignore
        pad_token_map_ = {
            "input_ids": 0,
            "input_mask": False,
            "target_ids": 0,
            "target_mask": False,
        }

        if self.mixture_name in {"d4_dev", "debug_dev", "green"} and split != self.train_split:
            pad_token_map_["is_correct"] = False
            pad_token_map_["is_correct_mask"] = False
        return pad_token_map_

    def load(self) -> DatasetDict:
        with Tqdm.tqdm(self.t0_mixture.data_modules.items(), "Loading T0 datasets") as dm_iter:
            for name, data_module in dm_iter:
                dm_iter.set_postfix({"module": name if len(name) < 30 else (name[:27] + "...")})
                data_module.tokenizer = self.tokenizer
                data_module.task_token_ids = self.task_token_ids
                data_module.setup()

        return DatasetDict(
            splits={
                "train": MixerDataset(
                    [dm[dm.train_split] for dm in self.t0_mixture.data_modules.values()],
                    sampling_cap=self.sampling_cap,
                ),
                "dev": MixerDataset(
                    [
                        dm[dm.dev_splits[0]]
                        for dm in self.t0_mixture.data_modules.values()
                        if len(dm.dev_splits) > 0
                    ],
                    sampling_cap=self.sampling_cap,
                ),
            }
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        epochs_elapsed = checkpoint["epoch"]  # verified that this is 1-based, so we're good
        assert self.dataset_dict is not None  # loaded already
        for mixer_dataset in self.dataset_dict.values():
            assert isinstance(mixer_dataset, MixerDataset)
            for dataset in mixer_dataset._datasets:
                if isinstance(dataset, _UndersampledDataset):
                    dataset.fast_forward(epochs_elapsed)

        super().on_load_checkpoint(checkpoint)
