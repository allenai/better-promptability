from __future__ import annotations
import re
from typing import Any, Union

import numpy as np
import torch
from transformers.optimization import (
    AdamW,
    Adafactor as HFAdafactor,
    get_linear_schedule_with_warmup,
)
from tango.integrations.torch.optim import Optimizer, LRScheduler

from ..modules.with_prefix_embedding import WithPrefixEmbedding


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


def resolve_optimizer_conf(
    opt_conf: Union[list[Optimizer], tuple[list[Optimizer], list[dict]]]
) -> Optimizer:
    """
    Get the optimizer from the lightning's configure_optimizers() output.
    """
    if (
        isinstance(opt_conf, (list, tuple))
        and len(opt_conf) == 2
        and isinstance(opt_conf[0][0], Optimizer)
    ):
        # optimizers + schedulers
        optimizers = opt_conf[0]
    else:
        optimizers = opt_conf
    assert len(optimizers) == 1
    return optimizers[0]
