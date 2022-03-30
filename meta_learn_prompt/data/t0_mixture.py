from __future__ import annotations
from typing import Mapping, Optional
import pickle

from tango.common import PathOrStr, Params

from .config import Config
from .t0_module import T0Module


class T0Mixture:
    """
    This class is used to initialize a collection of T0DataModule.
    """

    def __init__(
        self,
        mixture_name: str,  # should be "d4_train", "debug_train", "d4_dev", "debug_dev", or "green"
        config: Config,
        num_prefix: int,
        transformer_model: PathOrStr,
        t0_data_cache: PathOrStr = "/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache/",
        sequence_length: Optional[Mapping[str, int]] = None,
        subsample_indices_file: Optional[str] = None,
        train_module: bool = False,
        **data_module_kwargs,
    ):
        assert mixture_name in {"d4_train", "d4_dev", "green", "debug_train", "debug_dev"}
        self.mixture_name = mixture_name
        self.data_modules: dict[str, T0Module] = {}
        task_indices = None
        if subsample_indices_file:
            task_indices = pickle.load(open(subsample_indices_file, "rb"))
        for task_name in Params.from_file("configs/t0_mixtures.jsonnet")[mixture_name]:
            if task_indices is not None and (task_name not in task_indices or len(task_indices[task_name]) == 0):
                print(f'task {task_name} not in indices and is being left out of mixture.')
                continue
            self.data_modules[task_name] = T0Module(
                config=config,
                num_prefix=num_prefix,
                transformer_model=transformer_model,
                mixture_name=self.mixture_name,
                task_name=task_name,
                t0_data_cache=t0_data_cache,
                sequence_length=sequence_length,
                subsample_indices_file=subsample_indices_file,
                train_module=train_module,
                **data_module_kwargs,
            )
        assert len(self.data_modules) > 0
