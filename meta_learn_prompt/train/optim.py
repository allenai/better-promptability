from transformers.optimization import (
    AdamW,
    Adafactor as HFAdafactor,
    get_linear_schedule_with_warmup,
)
from tango.integrations.torch.optim import Optimizer, LRScheduler


# Register optimizers from HF as `Optimizer`s so we can use them in the train step.
Optimizer.register("transformers_adamw")(AdamW)


@Optimizer.register("adafactor")
class Adafactor(HFAdafactor):
    """See https://github.com/huggingface/transformers/issues/14830"""

    @staticmethod
    def _get_options(param_group, param_shape, min_dim_size_to_factor=128):
        factored, use_first_moment = HFAdafactor._get_options(param_group, param_shape)
        if all(d < min_dim_size_to_factor for d in param_shape):
            factored = False
        return factored, use_first_moment


# We also want to use `get_linear_schedule_with_warmup()` from HF, but we need a class
# to work with, so we just create this dummy class with a classmethod that will call
# `get_linear_schedule_with_warmup()`.
@LRScheduler.register("linear_with_warmup", constructor="linear_with_warmup")
class TransformersLambdaLR(LRScheduler):
    @classmethod
    def linear_with_warmup(cls, optimizer: Optimizer, **kwargs) -> LRScheduler:
        return get_linear_schedule_with_warmup(optimizer, **kwargs)
