from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
from tango.common.lazy import Lazy
from tango.integrations.pytorch_lightning import LightningTrainer
from tango.format import JsonFormat
from tango.step import Step

from ..data.config import Config
from ..data.prompt_data_module import PromptDataModule
from ..models.model import Model


@Step.register("eval_step")
class EvalStep(Step):

    DETERMINISTIC: bool = True
    CACHEABLE = True
    FORMAT = JsonFormat()

    def run(  # type: ignore[override]
        self,
        config: Config,
        trainer: Lazy[LightningTrainer],
        model: Lazy[Model],
        datamodule: Lazy[PromptDataModule],
    ) -> Tuple[Optional[str], List[Dict[str, float]]]:
        pl.seed_everything(config.seed)

        datamodule = datamodule.construct(config=config)

        datamodule.prepare_data()
        datamodule.setup()

        trainer: LightningTrainer = trainer.construct(
            work_dir=self.work_dir,
            gpus=config.gpus,
            accelerator="gpu" if config.gpus else "cpu",
            auto_select_gpus=True,
        )

        model = model.construct(config=config, dataset=datamodule)

        output = trainer.test(model, dataloaders=datamodule.val_dataloader())

        # Make output the same format as TrainStep for results aggregation.
        # Maybe it's cleaner to make the aggregation more flexible instead.
        assert len(output) == 1
        output = [{"best_" + k: v for k, v in output[0].items()}]

        return None, output
