from __future__ import annotations
import logging
import os
import pickle
from typing import Any, Dict, Union

from allennlp.training.metrics import Metric
from learn2learn.utils import clone_module
from tango.common.lazy import Lazy
import torch
import torch.distributed as dist
from tango.common.params import logger as tango_logger
from tango.integrations.torch.optim import Optimizer

from .model import Model
from .prefix_transformer import PrefixTransformer
from ..modules.with_prefix_embedding import logger as wpe_logger
from ..train.optim import load_adafactor_state, resolve_optimizer_conf

logger = logging.getLogger(__name__)


def split_batch(batch, split_size):
    bsz = batch["input_ids"].shape[0]
    assert bsz % split_size == 0
    assert all(v.shape[0] == bsz for v in batch.values())
    splits = None
    for k, v in batch.items():
        v_splits = v.split(split_size)
        if splits is None:
            splits = [{} for _ in v_splits]
        for i, v_split in enumerate(v_splits):
            splits[i][k] = v_split
    return splits


@Model.register("meta_learner")
class MetaLearner(Model):
    def __init__(
        self,
        model: PrefixTransformer,
        adaptation_steps: int,
        algorithm: str,
        meta_optimizer: Lazy[Optimizer],
        different_inner_loop_batches: bool = False,
        load_opt_states: bool = True,
        meta_accumulate_grad_batches: int = 1,
        reuse_inner_opt_state: bool = True,
    ):
        # TODO: anneal meta LR?
        assert algorithm in {"fomaml", "reptile"}

        super().__init__(model.config, model.dataset, optimizer=meta_optimizer, epochs=model.epochs)

        self.model = model
        self.algorithm = algorithm
        self.adaptation_steps = adaptation_steps
        self.different_inner_loop_batches = different_inner_loop_batches
        self.load_opt_states = load_opt_states
        self.meta_accumulate_grad_batches = meta_accumulate_grad_batches
        self.reuse_inner_opt_state = reuse_inner_opt_state

        inner_optimizer = resolve_optimizer_conf(self.model.configure_optimizers())
        if self.reuse_inner_opt_state:
            self.inner_optimizer_state = inner_optimizer.state_dict()

        if algorithm == "reptile" and self.adaptation_steps == 1:
            logger.warning("Reptile with 1 adaptation step is equivalent to MTL.")

        self.model.metrics = self.model.setup_metrics()
        self.metrics = self.model.metrics

        # ShardedDataParallel uses .requires_grad for sharding, and yet we use this property in
        # quite complicated ways for meta learning. We need to make sure that this property
        # correctly reflects the learnablity of each parameter after initialization. We restore
        # it for our purposes in the first forward pass.
        self.orig_requires_grad = self.model.unfreeze()
        self.restored_requires_grad = False

    def setup(self, stage: str = None):
        pass

    def setup_metrics(self) -> Dict[str, Dict[str, Metric]]:
        return {}

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def get_predictions(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, meta_batch: list[tuple[dict, dict]]) -> dict[str, torch.Tensor]:
        if not self.restored_requires_grad:
            for p in self.model.parameters():
                p.requires_grad = self.orig_requires_grad[p]
            self.restored_requires_grad = True

        for p in self.model.parameters():
            p.grad = torch.zeros_like(p.data)

        # These are for logging only
        support_loss = 0.0
        query_loss = torch.zeros([])  # a dummy query loss for reptile

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
            if self.reuse_inner_opt_state:
                inner_optimizer.load_state_dict(self.inner_optimizer_state)

            wpe_logger.setLevel(wpe_logger_level)
            tango_logger.setLevel(tango_logger_level)

            support_batch_size = support_batch["input_ids"].shape[0]
            if self.different_inner_loop_batches:
                support_batch_size = support_batch_size // self.adaptation_steps
                support_batches = split_batch(support_batch, support_batch_size)

            support_split_size = support_batch_size // self.meta_accumulate_grad_batches
            query_split_size = (
                query_batch["input_ids"].shape[0] // self.meta_accumulate_grad_batches
            )
            for i, adaptation_step in enumerate(range(self.adaptation_steps)):
                inner_optimizer.zero_grad()
                curr_support_batch = (
                    support_batches[i] if self.different_inner_loop_batches else support_batch
                )
                for support_batch_split in split_batch(curr_support_batch, support_split_size):
                    output = learner(support_batch_split)
                    loss = self.model.compute_loss(
                        output["logits"],
                        support_batch_split["target_ids"],
                        support_batch_split.get("target_mask"),
                    )
                    # Don't worry, this backward doesn't trigger unwanted gradient sync in
                    # distributed training, because self.model is a torch module, not a
                    # distributed wrapper.
                    loss.backward()
                    if adaptation_step == self.adaptation_steps - 1:
                        support_loss += loss.detach().cpu()
                inner_optimizer.step()

            # In the inner loop we only tune the prompt embeddings, and in the outer loop we
            # unfreeze the model to tune it in its entirety.
            learner.unfreeze()
            inner_optimizer.zero_grad()
            for query_batch_split in split_batch(query_batch, query_split_size):
                query_output = learner(query_batch_split)
                loss = self.model.compute_loss(
                    query_output["logits"],
                    query_batch_split["target_ids"],
                    query_batch_split.get("target_mask"),
                )
                loss.backward()
                query_loss += loss.detach().cpu()

            if self.algorithm == "fomaml":
                for p, l in zip(self.model.parameters(), learner.parameters()):
                    p.grad.data.add_(l.grad.data)
            elif self.algorithm == "reptile":
                inner_optimizer.step()
                for p, l in zip(self.model.parameters(), learner.parameters()):
                    p.grad.data.add_(-1.0, l.data)
            else:
                assert False

            if self.reuse_inner_opt_state:
                self.inner_optimizer_state = inner_optimizer.state_dict()

        for p in self.model.parameters():
            # In distributed training, these averages are in most cases exact. The only exception
            # is at the end of an epoch where different GPUs might have different-sized data.
            # But since that happens VERY infrequently, we can live with this rather than
            # implementating custom ddp comm hooks.
            if self.algorithm == "fomaml":
                p.grad.data.div_(len(meta_batch))
            elif self.algorithm == "reptile":
                p.grad.data.div_(len(meta_batch)).add_(p.data)

        support_loss /= len(meta_batch)
        query_loss /= len(meta_batch)

        if dist.is_initialized():
            # Gradient sync is normally performed in backward(), but we don't call backward for meta
            # learning since we modify .grad directly. So we need to manually sync gradients.
            # self.trainer.model is the distributed wrapper.
            self.trainer.model.reduce()
            # reduce uses SUM, but we want averages
            for p in self.model.parameters():
                if p.grad is not None:  # in sharded ddp, each worker only gets some gradients
                    p.grad.data.div_(dist.get_world_size())

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

    def eval_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx=0, compute_loss=True
    ) -> dict[str, Any]:
        return self.model.eval_step(batch, batch_idx, dataloader_idx, compute_loss)

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
