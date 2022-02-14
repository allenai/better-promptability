from typing import Optional, Union

from tango.common.aliases import PathOrStr
from tango.common.registrable import Registrable


class Config(Registrable):
    def __init__(
        self,
        seed: int = 42,
        gpus: int = 1,
        precision: Union[int, str] = 32,
        output_dir: Optional[PathOrStr] = None,
        auto_select_gpus: bool = True,
    ):
        self.seed = seed
        self.precision = precision
        self.gpus = gpus  # TODO: do stuff with visible devices.
        self.output_dir = output_dir
        self.auto_select_gpus = auto_select_gpus


Config.register("default")(Config)
