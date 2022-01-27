import logging
import os
from typing import Dict, List, Tuple

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from tango.common.lazy import Lazy
from tango.integrations.pytorch_lightning import (
    LightningCallback,
    LightningModule,
    LightningTrainer,
)
from tango.format import JsonFormat
from tango.step import Step

from ..data.config import Config
from ..data.prompt_data_module import PromptDataModule
from ..data.t0_multitask_data_module import T0MultiTaskDataModule
from ..models.model import Model

logger = logging.getLogger(__name__)


@LightningCallback.register("my_logger")
class LoggingCallback(LightningCallback):
    def __init__(self):
        self.best_epoch = None
        self.best_dev_metric = None
        self.best_dev_metrics = None
        self.metrics_history = []

    @rank_zero_only
    def on_validation_end(self, trainer: LightningTrainer, pl_module: LightningModule):
        logger.info("")
        logger.info(f"***** Validation results at epoch {trainer.current_epoch} *****")

        assert pl_module.dataset.metric_watch_mode in {"max", "min"}
        self.metrics_history.append({})

        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}".format(key, str(metrics[key])))
                self.metrics_history[-1][key] = metrics[key]

            if key == pl_module.dataset.metric_to_watch and not trainer.sanity_checking:
                curr_metric = pl_module.dataset.postprocess_metric(key, metrics[key])
                if (
                    self.best_dev_metric is None
                    or (
                        pl_module.dataset.metric_watch_mode == "max"
                        and curr_metric > self.best_dev_metric
                    )
                    or (
                        pl_module.dataset.metric_watch_mode == "min"
                        and curr_metric < self.best_dev_metric
                    )
                ):
                    self.best_epoch = trainer.current_epoch
                    self.best_dev_metric = curr_metric
                    self.best_dev_metrics = {
                        k: v
                        for k, v in metrics.items()
                        if k not in {"log", "progress_bar", "loss", "val_loss", "lr", "epoch"}
                    }

        if not trainer.sanity_checking:
            logger.info(f"best_epoch = {self.best_epoch}")
            self.metrics_history[-1]["best_epoch"] = self.best_epoch
            for key, value in sorted(self.best_dev_metrics.items()):
                logger.info(f"best_{key} = {value}")
                self.metrics_history[-1][f"best_{key}"] = value


@LightningCallback.register("t0_multitask")
class T0MultiTaskCallback(LightningCallback):
    """
    A Lightning callback for resampling the ``MixerDataset`` at the end of each epoch.
    """

    def on_epoch_end(self, trainer: LightningTrainer, pl_module: LightningModule):
        assert isinstance(pl_module.dataset, T0MultiTaskDataModule)
        for dataset in pl_module.dataset.dataset_dict.values():
            dataset.resample()


@Step.register("train_step")
class TrainStep(Step):

    DETERMINISTIC: bool = True
    CACHEABLE = True
    FORMAT = JsonFormat()

    def run(  # type: ignore[override]
        self,
        config: Config,
        trainer: Lazy[LightningTrainer],
        model: Lazy[Model],
        datamodule: Lazy[PromptDataModule],
        # optimizer: Lazy[Optimizer],
        # lr_schedule: Lazy[LRScheduler],
    ) -> Tuple[str, List[Dict]]:

        pl.seed_everything(config.seed)

        datamodule = datamodule.construct(config=config)

        datamodule.prepare_data()
        datamodule.setup()

        trainer: LightningTrainer = trainer.construct(
            work_dir=self.work_dir,
            gpus=config.gpus,
            precision=16 if config.fp16 else 32,
            accelerator="gpu" if config.gpus else "cpu",
            auto_select_gpus=config.auto_select_gpus,
            # Need to reload the dataloaders each epoch when using the T0MultiTaskDataModule.
            reload_dataloaders_every_n_epochs=1
            if isinstance(datamodule, T0MultiTaskDataModule)
            else 0,
        )

        # Make sure we're using the `T0MultiTaskCallback` if using the `T0MultiTaskDataModule`
        if isinstance(datamodule, T0MultiTaskDataModule):
            for callback in trainer.callbacks:
                if isinstance(callback, T0MultiTaskCallback):
                    break
            else:
                raise RuntimeError("T0MultiTaskCallback required when using T0MultiTaskDataModule")

        epochs = trainer.max_epochs

        model = model.construct(
            config=config,
            dataset=datamodule,
            epochs=epochs,
            accumulate_grad_batches=trainer.accumulate_grad_batches,
        )

        assert model.epochs == epochs

        # Find the checkpoint callback and make sure it uses the right directory.
        # Also find the logging callback.
        checkpoint_callback: pl.callbacks.model_checkpoint.ModelCheckpoint
        logging_callback: LoggingCallback
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.model_checkpoint.ModelCheckpoint):
                callback.dirpath = self.work_dir
                checkpoint_callback = callback
            if isinstance(callback, LoggingCallback):
                logging_callback = callback

        resume_from_checkpoint = None
        if "last.ckpt" in os.listdir(self.work_dir):
            resume_from_checkpoint = os.path.join(self.work_dir, "last.ckpt")
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume_from_checkpoint)

        return (checkpoint_callback.best_model_path, logging_callback.metrics_history)
