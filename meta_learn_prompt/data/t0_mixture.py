from __future__ import annotations
import csv
from typing import Mapping, Optional

from tango.common import PathOrStr

from .config import Config
from .t0_module import T0Module


# TODO: make this a singleton or something, if it's slow
def read_task_info() -> dict[str, tuple[str, Optional[str], str]]:
    task_name_to_info: dict[str, tuple[str, Optional[str], str]] = {}
    with open("data/t0_task_info.tsv", newline="") as task_info_file:
        reader = csv.DictReader(task_info_file, delimiter="\t")
        for row in reader:
            if len(row["subset_name"]) == 0:
                row["subset_name"] = None  # type: ignore
            task_name_to_info[row["task_name"]] = (
                row["dataset_name"],
                row["subset_name"],
                row["template_name"],
            )
    return task_name_to_info


class T0Mixture:
    """
    This class is used to initialize a collection of T0DataModule.
    """

    def __init__(
        self,
        mixture_name: str,  # should be "d4_train", "d4_dev", or "green"
        config: Config,
        num_prefix: int,
        transformer_model: PathOrStr,
        t0_data_cache: PathOrStr = "/net/nfs2.allennlp/akshitab/meta-learn-prompt/t0/processed_cache/",
        sequence_length: Optional[Mapping[str, int]] = None,
        subsample_indices_file: Optional[str] = None,
        **data_module_kwargs,
    ):
        assert mixture_name in {"d4_train", "d4_dev", "green"}
        self.mixture_name = mixture_name
        self.data_modules: dict[str, T0Module] = {}
        for task_name in (line.strip() for line in open(f"data/{self.mixture_name}_tasks.txt")):
            self.data_modules[task_name] = T0Module(
                config=config,
                num_prefix=num_prefix,
                transformer_model=transformer_model,
                mixture_name=self.mixture_name,
                task_name=task_name,
                t0_data_cache=t0_data_cache,
                sequence_length=sequence_length,
                subsample_indices_file=subsample_indices_file,
                **data_module_kwargs,
            )
        assert len(self.data_modules) > 0
