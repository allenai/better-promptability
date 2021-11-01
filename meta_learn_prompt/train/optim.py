from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tango.integrations.torch.optim import Optimizer, LRScheduler


# Register the AdamW optimizer from HF as an `Optimizer` so we can use it in the train step.
Optimizer.register("transformers_adamw")(AdamW)


# We also want to use `get_linear_schedule_with_warmup()` from HF, but we need a class
# to work with, so we just create this dummy class with a classmethod that will call
# `get_linear_schedule_with_warmup()`.
@LRScheduler.register("linear_with_warmup", constructor="linear_with_warmup")
class TransformersLambdaLR(LRScheduler):
    @classmethod
    def linear_with_warmup(cls, optimizer: Optimizer, **kwargs) -> LRScheduler:
        return get_linear_schedule_with_warmup(optimizer, **kwargs)
