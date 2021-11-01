import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from tango.integrations.pytorch_lightning import (
    LightningCallback,
    LightningTrainer,
    LightningModule,
)
from tango.common.lazy import Lazy
from tango.step import Step
from tango.integrations.torch.format import TorchFormat

from ..data.config import Config
from ..data.data_module import DataModule, FewShotDataset

from ..models.prefix_transformer import Model, PrefixTransformer

logger = logging.getLogger(__name__)


@LightningCallback.register("my_logger")
class LoggingCallback(LightningCallback):
    def __init__(self):
        self.best_epoch = None
        self.best_dev_metric = None
        self.best_dev_metrics = None

    @rank_zero_only
    def on_validation_end(self, trainer: LightningTrainer, pl_module: LightningModule):
        logger.info("")
        logger.info(f"***** Validation results at epoch {trainer.current_epoch} *****")

        assert pl_module.dataset.metric_watch_mode in {"max", "min"}

        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}".format(key, str(metrics[key])))

            if key == pl_module.dataset.metric_to_watch and not trainer.sanity_checking:
                if (
                    self.best_dev_metric is None
                    or (
                        pl_module.dataset.metric_watch_mode == "max"
                        and metrics[key] > self.best_dev_metric
                    )
                    or (
                        pl_module.dataset.metric_watch_mode == "min"
                        and metrics[key] < self.best_dev_metric
                    )
                ):
                    self.best_epoch = trainer.current_epoch
                    self.best_dev_metric = metrics[key]
                    self.best_dev_metrics = {
                        k: v
                        for k, v in metrics.items()
                        if k not in {"log", "progress_bar", "loss", "val_loss", "lr", "epoch"}
                    }

        if not trainer.sanity_checking:
            logger.info(f"best_epoch = {self.best_epoch}")
            for key, value in sorted(self.best_dev_metrics.items()):
                logger.info(f"best_{key} = {value}")


@Step.register("train_step")
class TrainStep(Step):

    DETERMINISTIC: bool = True
    CACHEABLE = True
    FORMAT = TorchFormat()

    def run(  # type: ignore[override]
        self,
        config: Config,
        trainer: Lazy[LightningTrainer],
        model: Lazy[PrefixTransformer],
        datamodule: Lazy[FewShotDataset],
        # optimizer: Lazy[Optimizer],
        # lr_schedule: Lazy[LRScheduler],
    ):  # -> torch.nn.Module:

        pl.seed_everything(config.seed)

        datamodule = datamodule.construct(config=config)

        datamodule.prepare_data()
        datamodule.setup()

        trainer: LightningTrainer = trainer.construct(work_dir=self.work_dir)

        model = model.construct(config=config, dataset=datamodule)

        # model.setup()
        # model.configure_optimizers()

        # train_dataloader = dataloader(dataset, dataset.train_split, shuffle=True, batch_size=model.batch_size, gpus=config.gpus)
        # val_dataloader = dataloader(dataset, "dev", shuffle=False, batch_size=model.eval_batch_size, gpus=config.gpus)

        # model._train_dataloader = train_dataloader
        # model.dataset_size = len(model._train_dataloader.dataset)

        # trainer: LightningTrainer = trainer.construct(work_dir=self.work_dir)

        # Find the checkpoint callback and make sure it uses the right directory.
        checkpoint_callback: pl.callbacks.model_checkpoint.ModelCheckpoint
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.model_checkpoint.ModelCheckpoint):
                callback.dirpath = self.work_dir
                checkpoint_callback = callback

        trainer.fit(model, datamodule=datamodule)  # train_dataloader, val_dataloader)

        best_model = torch.load(checkpoint_callback.best_model_path)
        return best_model