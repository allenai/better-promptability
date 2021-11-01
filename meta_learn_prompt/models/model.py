from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from allennlp.training.metrics import Metric
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW

from tango.common.lazy import Lazy
from tango.integrations.pytorch_lightning.model import LightningModule
from tango.integrations.torch.optim import Optimizer, LRScheduler

from ..data.config import Config
from ..data.data_module import DataModule


class Model(LightningModule):
    def __init__(
        self,
        config: Config,
        dataset: DataModule,
        epochs: int = 3,
        optimizer: Optional[Lazy[Optimizer]] = None,
        weight_decay: float = 0.0,
        accumulate_grad_batches: int = 1,
        warmup_steps: int = 0,
        lr_scheduler_total_steps: Optional[int] = None,
    ):
        super().__init__()

        self.config = config
        self.dataset = dataset
        self._optimizer = optimizer

        self.epochs = epochs
        self.optimizer_kwargs = {
            "weight_decay": weight_decay,
            "accumulate_grad_batches": accumulate_grad_batches,
            "warmup_steps": warmup_steps,
            "lr_scheduler_total_steps": lr_scheduler_total_steps,
        }

        self.metrics = self.setup_metrics()

    def setup(self, stage: str = None):
        """To set up self.dataset_size"""
        if stage != "fit":
            return

        self._train_dataloader = self.dataset.dataloader(
            "train", self.dataset.batch_size, shuffle=True
        )
        self.dataset_size = len(self._train_dataloader.dataset)

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    # # def val_dataloader(self, shuffle=False) -> list[DataLoader]:
    # #     return [
    # #         self.dataset.dataloader(split, self.eval_batch_size, shuffle=shuffle)
    # #         for split in self.dataset.dev_splits
    # #     ]

    # # def test_dataloader(self, shuffle=False) -> list[DataLoader]:
    # #     return [
    # #         self.dataset.dataloader(split, self.eval_batch_size, shuffle=shuffle)
    # #         for split in self.dataset.test_splits
    # #     ]

    def setup_metrics(self) -> Dict[str, Dict[str, Metric]]:
        return {
            split: {name: Metric.by_name(name)() for name in self.dataset.metric_names}
            for split in self.dataset.dev_splits + self.dataset.test_splits
        }

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict]]:
        "Prepare optimizer and schedule (linear warmup and decay)"

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
                "weight_decay": self.optimizer_kwargs["weight_decay"],
            },
        ]
        # optimizer = AdamW(
        #     optimizer_grouped_parameters,
        #     lr=self.optimizer_kwargs["lr"],
        #     eps=self.optimizer_kwargs["adam_epsilon"],
        # )

        optimizer = self._optimizer.construct(params=optimizer_grouped_parameters)
        scheduler = self.get_lr_scheduler(optimizer)

        return [optimizer], [scheduler]

    def get_lr_scheduler(self, optimizer: Optimizer) -> dict:
        num_devices = max(1, self.config.gpus)  # TODO: consider num_tpu_cores
        # TODO: use world_size.
        if self.optimizer_kwargs["lr_scheduler_total_steps"] is not None:
            total_steps = self.optimizer_kwargs["lr_scheduler_total_steps"]
        else:
            effective_batch_size = (
                self.dataset.batch_size
                * self.optimizer_kwargs["accumulate_grad_batches"]
                * num_devices
            )
            # Sometimes dataset_size could be smaller than the effective_batch_size
            # TODO: do something about dataset_size
            total_steps = max(self.dataset_size / effective_batch_size, 1) * self.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.optimizer_kwargs["warmup_steps"],
            num_training_steps=total_steps,
        )
        return {"scheduler": scheduler, "interval": "step", "frequency": 1}

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int
    ):
        """See https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"""
        optimizer.zero_grad(set_to_none=True)

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor = None, reduce=True
    ) -> torch.Tensor:
        if self.dataset.output_mode == "classification":
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        elif self.dataset.output_mode == "token_classification":
            assert mask.any(dim=-1).all()
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
            )
            loss = loss.view_as(labels) * mask
            if reduce:
                loss = loss.sum() / mask.sum()
        elif self.dataset.output_mode == "regression":
            loss = F.mse_loss(logits.view(-1), labels.view(-1))
        else:
            raise KeyError(f"Output mode not supported: {self.dataset.output_mode}")
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        loss = self.compute_loss(
            self(batch)["logits"], batch[self.dataset.label_key], batch.get("label_mask")
        )
        self.log("train_loss", loss)
        self.log("lr", self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[-1], prog_bar=True)
        return {"loss": loss}

    def get_predictions(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.dataset.output_mode in ("classification", "token_classification"):
            return logits.argmax(dim=-1)
        elif self.dataset.output_mode == "regression":
            return logits.squeeze(dim=-1)
        else:
            raise KeyError(f"Output mode not supported: {self.dataset.output_mode}")

    def eval_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        mode: str,
        dataloader_idx=0,
        compute_loss=True,
    ) -> dict[str, Any]:
        assert mode in {"dev", "test"}

        logits = self(batch)["logits"]
        preds = self.get_predictions(logits, batch)
        labels = batch[self.dataset.label_key]

        splits = self.dataset.dev_splits if mode == "dev" else self.dataset.test_splits
        split = splits[dataloader_idx]
        for metric in self.metrics[split].values():
            metric(*metric.detach_tensors(preds, labels))

        return (
            {"loss": self.compute_loss(logits, labels, batch.get("label_mask")).detach().cpu()}
            if compute_loss
            else {}
        )

    def eval_epoch_end(
        self, outputs: Union[list[list[dict[str, Any]]], list[dict[str, Any]]], mode: str
    ):
        assert mode in {"dev", "test"}

        # pytorch-lightning "conveniently" unwraps the list when there's only one dataloader,
        # so we need a check here.
        num_splits = 1 if isinstance(outputs[0], dict) else len(outputs)

        # We gather individual metrics from each dataloader and compute the average if there is
        # more than one
        if num_splits > 1:
            sums = defaultdict(int)
        for i in range(num_splits):
            split = (self.dataset.dev_splits if mode == "dev" else self.dataset.test_splits)[i]
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
        return self.eval_step(batch, batch_idx, "dev", dataloader_idx=dataloader_idx)

    def validation_epoch_end(self, outputs: list[dict[str, Any]]):
        return self.eval_epoch_end(outputs, "dev")

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx=0
    ) -> dict[str, Any]:
        return self.eval_step(batch, batch_idx, "test", dataloader_idx=dataloader_idx)

    def test_epoch_end(self, outputs: list[dict[str, Any]]):
        return self.eval_epoch_end(outputs, "test")
