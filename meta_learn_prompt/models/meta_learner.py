from __future__ import annotations
import logging
import os
import pickle
from typing import Any, Dict, Optional, Union

from allennlp.training.metrics import Metric
import learn2learn as l2l
from tango.common.lazy import Lazy
import torch
from tango.common.params import logger as tango_logger
from tango.integrations.torch.optim import Optimizer

from .fomaml import FOMAML
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
        first_order: bool,
        meta_optimizer: Lazy[Optimizer],
        load_opt_states: bool = True,
    ):
        # TODO: anneal meta LR?
        assert algorithm in {"maml", "fomaml", "meta-sgd", "reptile"}
        if algorithm in {"reptile", "fomaml"} and not first_order:
            raise NotImplementedError("Only first-order fomaml and reptile are supported.")

        super().__init__(model.config, model.dataset, optimizer=meta_optimizer, epochs=model.epochs)

        self.model = model
        self.algorithm = algorithm
        self.adaptation_steps = adaptation_steps
        self.load_opt_states = load_opt_states

        inner_optimizer = resolve_optimizer_conf(self.model.configure_optimizers())
        self.inner_optimizer_state = inner_optimizer.state_dict()
        adaptation_lr = inner_optimizer.defaults["lr"]

        if algorithm == "maml":
            self.meta_model = l2l.algorithms.MAML(model, adaptation_lr, first_order=first_order)
        elif algorithm == "fomaml":
            self.meta_model = FOMAML(model, adaptation_lr, self.inner_optimizer_state)
        elif algorithm == "meta-sgd":
            self.meta_model = l2l.algorithms.MetaSGD(
                model, lr=adaptation_lr, first_order=first_order
            )
        elif algorithm == "reptile":
            if self.adaptation_steps == 1:
                logger.warning("Reptile with 1 adaptation step is equivalent to MTL.")
            model.train_full_model = True
        else:  # TODO: per-layer LR, or even per-layer per-step https://www.bayeswatch.com/2018/11/30/HTYM/
            raise NotImplementedError

    def setup(self, stage: str = None):
        pass

    def setup_metrics(self) -> Dict[str, Dict[str, Metric]]:
        return {}

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def get_predictions(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, meta_batch: list[tuple[dict, dict]]) -> dict[str, torch.Tensor]:
        if self.algorithm == "reptile":
            return self.reptile_foward(meta_batch)

        for p in self.model.parameters():
            p.grad = torch.zeros_like(p.data)

        support_loss = 0
        query_loss = 0
        for support_batch, query_batch in meta_batch:
            # Disable <ERROR logging from model recreation which would otherwise pollute stdout
            # TODO: this is ugly, but I don't know how to globally change logging level. A better
            # solution may be something like warn_once.
            wpe_logger_level = wpe_logger.level
            wpe_logger.setLevel(logging.ERROR)
            tango_logger_level = tango_logger.level
            tango_logger.setLevel(logging.ERROR)

            learner = self.meta_model.clone()
            learner.train()

            wpe_logger.setLevel(wpe_logger_level)
            tango_logger.setLevel(tango_logger_level)

            for _ in range(self.adaptation_steps):
                output = learner(support_batch)
                loss = self.model.compute_loss(
                    output["logits"], support_batch["target_ids"], support_batch.get("target_mask")
                )
                learner.adapt(loss)
            support_loss += loss.detach().cpu()

            learner.unfreeze()
            learner.inner_optimizer.zero_grad()
            query_output = learner(query_batch)
            loss = self.model.compute_loss(
                query_output["logits"], query_batch["target_ids"], query_batch.get("target_mask")
            )
            loss.backward()
            for p, l in zip(self.model.parameters(), learner.parameters()):
                p.grad.data.add_(l.grad.data)
            query_loss += loss.detach().cpu()

        support_loss /= len(meta_batch)
        query_loss /= len(meta_batch)

        return {"support_loss": support_loss, "query_loss": query_loss}

    def reptile_foward(self, meta_batch: list[tuple[dict, dict]]) -> dict[str, torch.Tensor]:
        # TODO: dedup with above
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p.data)

        support_loss = 0
        for support_batch, _ in meta_batch:  # reptile needs no query
            # Disable <ERROR logging from model recreation which would otherwise pollute stdout
            # TODO: this is ugly, but I don't know how to globally change logging level. A better
            # solution may be something like warn_once.
            wpe_logger_level = wpe_logger.level
            wpe_logger.setLevel(logging.ERROR)
            tango_logger_level = tango_logger.level
            tango_logger.setLevel(logging.ERROR)

            learner: PrefixTransformer = self.model.meta_learning_copy()
            inner_optimizer = resolve_optimizer_conf(
                learner.configure_optimizers(load_opt_states=False)
            )
            inner_optimizer.load_state_dict(self.inner_optimizer_state)

            wpe_logger.setLevel(wpe_logger_level)
            tango_logger.setLevel(tango_logger_level)

            for _ in range(self.adaptation_steps):
                inner_optimizer.zero_grad()
                output = learner(support_batch)
                loss = learner.compute_loss(
                    output["logits"], support_batch["target_ids"], support_batch.get("target_mask")
                )
                loss.backward()
                inner_optimizer.step()
            support_loss += loss.detach().cpu()
            self.inner_optimizer_state = inner_optimizer.state_dict()

            for p, l in zip(self.model.parameters(), learner.parameters()):
                p.grad.data.add_(-1.0, l.data)

        for p in self.model.parameters():
            p.grad.data.mul_(1.0 / len(meta_batch)).add_(p.data)

        support_loss /= len(meta_batch) * self.adaptation_steps
        return {"support_loss": support_loss}

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
        # Lightning requires a "loss" key, so when we don't have it (e.g., reptile), we use a dummy
        return {"loss": output.get("query_loss", torch.FloatTensor([0.0]))}

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
