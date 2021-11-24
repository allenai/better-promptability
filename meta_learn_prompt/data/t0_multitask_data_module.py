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
        mixture = T0Mixture(mixture_name=mixture_name)
