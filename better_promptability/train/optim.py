from __future__ import annotations
from typing import Union

from transformers.optimization import Adafactor as HFAdafactor
from tango.integrations.torch.optim import Optimizer


@Optimizer.register("adafactor")
class Adafactor(HFAdafactor):
    """See https://github.com/huggingface/transformers/issues/14830

    Nevertheless, this is only here for backward compatibility, and I suspect technically
    you can just use transformers::adafactor in your config.
    """

    @staticmethod
    def _get_options(param_group, param_shape, min_dim_size_to_factor=128):
        factored, use_first_moment = HFAdafactor._get_options(param_group, param_shape)
        if all(d < min_dim_size_to_factor for d in param_shape):
            factored = False
        return factored, use_first_moment


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
