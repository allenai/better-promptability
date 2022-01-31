from __future__ import annotations
import logging
import os
import pickle
from typing import Any, Dict, Union

from allennlp.training.metrics import Metric
from learn2learn.utils import clone_module
from tango.common.lazy import Lazy
import torch
from tango.common.params import logger as tango_logger
from tango.integrations.torch.optim import Optimizer

from .model import Model
from .prefix_transformer import PrefixTransformer
from ..modules.with_prefix_embedding import logger as wpe_logger
from ..train.optim import load_adafactor_state, resolve_optimizer_conf

logger = logging.getLogger(__name__)


@Model.register("meta_learner")
class MetaLearner(Model):
    def __init__(
        self,
        model: PrefixTransformer,
        adaptation_steps: int,
        algorithm: str,
        meta_optimizer: Lazy[Optimizer],
        load_opt_states: bool = True,
        meta_sgd: bool = False,
    ):
        # TODO: anneal meta LR?
        assert algorithm in {"fomaml", "reptile"}
        # Meta-SGD not well defined for REPTILE.
        assert not algorithm == "fomaml" and meta_sgd

        super().__init__(model.config, model.dataset, optimizer=meta_optimizer, epochs=model.epochs)

        self.model = model
        self.algorithm = algorithm
        self.adaptation_steps = adaptation_steps
        self.load_opt_states = load_opt_states
        self.meta_sgd = meta_sgd

        inner_optimizer = resolve_optimizer_conf(self.model.configure_optimizers())
        self.inner_optimizer_state = inner_optimizer.state_dict()

        if meta_sgd:
            initial_lr = inner_optimizer.defaults['lr']
            learning_rates = {}
            for name, parameter in self.model.named_parameters():
                # NOTE(rloganiv): Setting learning rates to 1.0 under
                # assumption of further multiplication by static learning rates
                # provided to optimizer.
                if parameter.requires_grad:
                    learning_rate = torch.nn.Parameter(
                        data=torch.tensor(
                            initial_lr,
                            device=parameter.device,
                            dtype=parameter.dtype,
                        ),
                        requires_grad=True,
                    )
                    learning_rates[name] = learning_rate
            self.learning_rates = torch.nn.ParameterDict(learning_rates)
        else:
            self.learning_rates = None

        if algorithm == "reptile" and self.adaptation_steps == 1:
            logger.warning("Reptile with 1 adaptation step is equivalent to MTL.")

    def setup(self, stage: str = None):
        pass

    def setup_metrics(self) -> Dict[str, Dict[str, Metric]]:
        return {}

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def get_predictions(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, meta_batch: list[tuple[dict, dict]]) -> dict[str, torch.Tensor]:
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p.data)

        # These are for logging only
        support_loss = 0.0
        query_loss = torch.zeros([], device=p.device)  # a dummy query loss for reptile

        for support_batch, query_batch in meta_batch:
            # Disable <ERROR logging from model recreation which would otherwise pollute stdout
            # TODO: this is ugly, but I don't know how to globally change logging level. A better
            # solution may be something like warn_once.
            wpe_logger_level = wpe_logger.level
            wpe_logger.setLevel(logging.ERROR)
            tango_logger_level = tango_logger.level
            tango_logger.setLevel(logging.ERROR)

            learner = clone_module(self.model)
            detach_module(learner, keep_requires_grad=True)
            learner.train()  # for meta-evaluation, though we don't have it right now
            inner_optimizer = resolve_optimizer_conf(
                learner.configure_optimizers(load_opt_states=False)
            )
            inner_optimizer.load_state_dict(self.inner_optimizer_state)

            if self.meta_sgd:
                inner_optimizer.defaults['lr'] = 1.0
                for param_group in inner_optimizer.param_groups:
                    if 'lr' in param_group:
                        param_group['lr'] = 1.0
                    
            wpe_logger.setLevel(wpe_logger_level)
            tango_logger.setLevel(tango_logger_level)

            # NOTE(rloganiv): In Meta-SGD the first order approximation of the
            # gradient w.r.t. the learning rates requires storing the grad
            # w.r.t. the inner loop loss.
            gradient_cache: Dict[str, torch.Tensor] = {}

            # Inner loop
            for _ in range(self.adaptation_steps):
                output = learner(support_batch)
                loss = self.model.compute_loss(
                    output["logits"], support_batch["target_ids"], support_batch.get("target_mask")
                )
                inner_optimizer.zero_grad()
                loss.backward()
                if self.meta_sgd:
                    for name, parameter in self.model.named_parameters():
                        if name in self.learning_rates:
                            # Store grad before rescaling.
                            gradient_cache[name] = parameter.grad.detach().clone()
                            parameter.grad *= self.learning_rates[name]
                inner_optimizer.step()
            self.inner_optimizer_state = inner_optimizer.state_dict()
            support_loss += loss.detach().cpu()

            # Outer loop
            if self.algorithm == "fomaml":
                # In the inner loop we only tune the prompt embeddings, and in the outer loop we
                # unfreeze the model to tune it in its entirety.
                learner.unfreeze()
                query_output = learner(query_batch)
                loss = self.model.compute_loss(
                    query_output["logits"],
                    query_batch["target_ids"],
                    query_batch.get("target_mask"),
                )
                inner_optimizer.zero_grad()
                loss.backward()
                for (n, p), l in zip(self.model.named_parameters(), learner.parameters()):
                    # FOMAML
                    p.grad.data.add_(l.grad.data)
                    # Meta-SGD
                    if self.meta_sgd:
                        if n in self.learning_rates:
                            grad = (p * gradient_cache[n]).sum()
                            self.learning_rates[n].grad.copy_(grad)
                query_loss += loss.detach().cpu()

            elif self.algorithm == "reptile":
                for p, l in zip(self.model.parameters(), learner.parameters()):
                    p.grad.data.add_(-1.0, l.data)
            else:
                assert False

        for p in self.model.parameters():
            if self.algorithm == "fomaml":
                p.grad.data.div_(len(meta_batch))
            elif self.algorithm == "reptile":
                p.grad.data.div_(len(meta_batch)).add_(p.data)

        support_loss /= len(meta_batch)
        query_loss /= len(meta_batch)

        return {"support_loss": support_loss, "query_loss": query_loss}

    def backward(self, *args, **kwargs):
        # Gradients are manually populated
        pass

    def optimizer_zero_grad(self, *args, **kwargs):
        # Gradients are manually populated, and we don't want them to be zeroed
        pass

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        output = self(batch)
        for k, v in output.items():
            self.log(k, v)
        if len(self.trainer.lr_schedulers) > 0:
            self.log(
                "lr", self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1], prog_bar=True
            )
        return {"loss": output["query_loss"]}

    def configure_optimizers(self) -> Union[list[Optimizer], tuple[list[Optimizer], list[dict]]]:
        opt_conf = super().configure_optimizers()

        if self._optimizer._params["type"] == "adafactor" and self.load_opt_states:  # type: ignore
            assert self.model.optstates_dir is not None
            optstates_path = os.path.join(
                self.model.optstates_dir, self.model.transformer_name.split("/")[-1]
            )
            optstates = pickle.load(open(optstates_path, "rb"))
            optimizer = resolve_optimizer_conf(opt_conf)
            load_adafactor_state(self.model.transformer.model, optimizer, optstates)

        return opt_conf

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        """
        PyTorch's native optimizer state checkpoint logic is very fragile, so we also do it on our
        own. See https://github.com/pytorch/pytorch/issues/1489
        """
        optimizer_states = self.optimizers(use_pl_optimizer=False).state
        param_to_name = {p: n for n, p in self.named_parameters()}
        states = {param_to_name[p]: states for p, states in optimizer_states.items()}
        checkpoint["custom_optimizer_states"] = states


def detach_module(module, keep_requires_grad=False):
    """
    Adapted from learn2learn.utils.detach_module to add the `keep_requires_grad` flag.
    This will no longer be necessary once https://github.com/learnables/learn2learn/pull/294 is
    merged.
    """
    if not isinstance(module, torch.nn.Module):
        return
    # First, re-write all parameters
    for param_key in module._parameters:
        if module._parameters[param_key] is not None:
            requires_grad = module._parameters[param_key].requires_grad
            detached = module._parameters[param_key].detach_()  # noqa: F841; consistency w/ orig.
            if keep_requires_grad and requires_grad:
                module._parameters[param_key].requires_grad_()

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        if module._buffers[buffer_key] is not None and module._buffers[buffer_key].requires_grad:
            module._buffers[buffer_key] = module._buffers[buffer_key].detach_()
            if keep_requires_grad:  # requires_grad checked above
                module._buffers[buffer_key].requires_grad_()

    # Then, recurse for each submodule
    for module_key in module._modules:
        detach_module(module._modules[module_key], keep_requires_grad=keep_requires_grad)
