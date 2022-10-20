from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from allennlp.training.metrics import Metric
import torch
import torch.nn.functional as F

from tango.common.lazy import Lazy
from tango.integrations.pytorch_lightning.model import LightningModule
from tango.integrations.torch.optim import Optimizer

from ..data.config import Config
from ..data.data_module import DataModule


class Model(LightningModule):
    def __init__(
        self,
        config: Config,
        dataset: DataModule,
        optimizer: Optional[Lazy[Optimizer]] = None,
        epochs: int = 3,
        weight_decay: float = 0.0,
        accumulate_grad_batches: int = 1,
        warmup_steps: int = 0,
    ):
        super().__init__()

        self.config = config
        self.dataset = dataset
        self._optimizer = optimizer
        if self._optimizer is not None:
            assert isinstance(self._optimizer, Lazy)

        self.epochs = epochs
        self.optimizer_kwargs = {
            "weight_decay": weight_decay,
            "accumulate_grad_batches": accumulate_grad_batches,
            "warmup_steps": warmup_steps,
        }

        self.metrics = self.setup_metrics()

    def setup(self, stage: str = None):
        """To set up self.dataset_size"""
        if stage != "fit":
            return
        self.dataset_size = len(self.dataset.dataset_dict[self.dataset.train_split])

    def setup_metrics(self) -> Dict[str, Dict[str, Metric]]:
        return {
            split: {
                name: self.dataset.instantiate_metric(name, split)
                for name in self.dataset.metric_names
            }
            for split in self.dataset.dev_splits + self.dataset.test_splits
        }

    def configure_optimizers(self) -> Union[List[Optimizer], Tuple[List[Optimizer], List[Dict]]]:
        """Prepare optimizer and schedule (linear warmup and decay)"""
        assert self._optimizer is not None

        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.optimizer_kwargs["weight_decay"],
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = self._optimizer.construct(params=optimizer_grouped_parameters)  # type: ignore

        return [optimizer]

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int
    ):
        """See https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"""
        optimizer.zero_grad()

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduce=True,
    ) -> torch.Tensor:
        assert mask is not None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none")
        loss = loss.view_as(labels) * mask
        if reduce:
            assert mask.any(dim=-1).all()
            loss = loss.sum() / mask.sum()  # type: ignore
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        loss = self.compute_loss(
            self(batch)["logits"], batch["target_ids"], batch.get("target_mask")
        )
        self.log("train_loss", loss)
        return {"loss": loss}

    def get_predictions(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def eval_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx=0,
        compute_loss=True,
    ) -> dict[str, Any]:
        logits = self(batch)["logits"]
        preds = self.get_predictions(logits, batch).masked_fill(
            ~batch["is_correct_mask"], torch.finfo(logits.dtype).min
        )
        targets = batch["target_ids"]  # target sequences.

        if "is_correct" in batch:
            labels = (batch["is_correct"] & batch["is_correct_mask"]).byte().argmax(dim=-1)

            split = self.dataset.dev_splits[dataloader_idx]
            for metric in self.metrics[split].values():
                metric(*metric.detach_tensors(preds, labels))

        return (
            {"loss": self.compute_loss(logits, targets, batch.get("targets_mask")).detach().cpu()}
            if compute_loss
            else {}
        )

    def eval_epoch_end(self, outputs: Union[list[list[dict[str, Any]]], list[dict[str, Any]]]):
        # pytorch-lightning "conveniently" unwraps the list when there's only one dataloader,
        # so we need a check here.
        num_splits = 1 if isinstance(outputs[0], dict) else len(outputs)

        # We gather individual metrics from each dataloader and compute the average if there is
        # more than one
        if num_splits > 1:
            sums: defaultdict = defaultdict(int)
        for i in range(num_splits):
            split = self.dataset.dev_splits[i]
            assert split != "avg"  # reserved keyword for below
            metrics = self.get_metrics(split, reset=True)
            for k, v in metrics.items():
                if num_splits > 1:
                    self.log(f"{k}_{split}", v)
                    sums[k] += v
                else:
                    self.log(k, v)
        if num_splits > 1:
            for k, v in sums.items():
                self.log(f"{k}_avg", v / num_splits)

    def get_metrics(self, split: str, reset=False) -> dict[str, Any]:
        metrics = {name: metric.get_metric() for name, metric in self.metrics[split].items()}
        if reset:
            for metric in self.metrics[split].values():
                metric.reset()
        return metrics

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx=0
    ) -> dict[str, Any]:
        return self.eval_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def validation_epoch_end(self, outputs: list[dict[str, Any]]):
        return self.eval_epoch_end(outputs)

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx=0
    ) -> dict[str, Any]:
        return self.eval_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def test_epoch_end(self, outputs: list[dict[str, Any]]):
        return self.eval_epoch_end(outputs)
