from __future__ import annotations
from typing import Mapping, Optional

from tango.common import PathOrStr, Params

from .config import Config
from .t0_module import T0Module


class T0Mixture:
    """
    This class is used to initialize a collection of T0Module.
    """

    def __init__(
        self,
        mixture_name: str,  # should be "d4_train", "d4_dev", or "green"
        config: Config,
        num_prefix: int,
        transformer_model: PathOrStr,
        t0_data_cache: PathOrStr,
        subsample_indices_file: Optional[str] = None,
        **data_module_kwargs,
    ):
        assert mixture_name in {"d4_train", "d4_dev", "green"}
        self.mixture_name = mixture_name
        self.data_modules: dict[str, T0Module] = {}
        for task_name in Params.from_file("configs/t0_mixtures.jsonnet")[mixture_name]:
            self.data_modules[task_name] = T0Module(
                config=config,
                num_prefix=num_prefix,
                transformer_model=transformer_model,
                mixture_name=self.mixture_name,
                task_name=task_name,
                t0_data_cache=t0_data_cache,
                subsample_indices_file=subsample_indices_file,
                **data_module_kwargs,
            )
        assert len(self.data_modules) > 0
