from typing import Optional

from tango.common.aliases import PathOrStr
from tango.common.registrable import Registrable
from tango.step import Step


class Config(Registrable):
    def __init__(
        self,
        seed: int = 42,
        gpus: int = 1,
        fp16: bool = False,
        output_dir: Optional[PathOrStr] = None,
    ):
        self.seed = seed
        self.fp16 = fp16
        self.gpus = gpus  # TODO: do stuff with visible devices.
        self.output_dir = output_dir


Config.register("default")(Config)


@Step.register("create_config")
class CreateConfig(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(self, **kwargs) -> Config:
        # Is this ridiculous? Yes.
        return Config(**kwargs)


class ConfigStep(Step):
    def run(self, config: Config, *args, **kwargs):
        return NotImplementedError
