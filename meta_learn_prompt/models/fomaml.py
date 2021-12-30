from re import A
from typing import Optional

import learn2learn as l2l
from tango.integrations.torch.optim import Optimizer

from .prefix_transformer import PrefixTransformer
from ..train.optim import resolve_optimizer_conf


class FOMAML(l2l.algorithms.MAML):
    """
    l2l.algorithms.MAML only supports SGD as the inner loop optimizer. This class allows more
    general optimizers, but only supports the first order case.
    """

    def __init__(
        self,
        model: PrefixTransformer,
        lr: float,
        inner_optimizer_state: dict,
        allow_unused: Optional[bool] = None,
        allow_nograd: bool = False,
    ):
        super().__init__(
            model, lr, first_order=True, allow_unused=allow_unused, allow_nograd=allow_nograd
        )
        self.inner_optimizer: Optional[Optimizer] = None  # only populated when cloned
        self.inner_optimizer_state = inner_optimizer_state

    def adapt(self, loss):
        self.inner_optimizer.zero_grad()
        loss.backward()
        self.inner_optimizer.step()
        self.inner_optimizer_state = self.inner_optimizer.state_dict()

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd

        copy = self.module.meta_learning_copy()
        inner_optimizer_copy = resolve_optimizer_conf(
            copy.configure_optimizers(load_opt_states=False)
        )
        clone = FOMAML(
            copy,
            self.lr,
            inner_optimizer_copy.state_dict(),
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
        )
        clone.inner_optimizer = inner_optimizer_copy
        return clone
