from typing import Dict, List

import pytorch_lightning as pl
from tango.common.lazy import Lazy
from tango.integrations.pytorch_lightning import LightningTrainer
from tango.format import JsonFormat
from tango.step import Step

from ..data.config import Config
from ..data.prompt_data_module import PromptDataModule
from ..models.prefix_transformer import PrefixTransformer
from meta_learn_prompt.models.model import Model


@Step.register("eval_step")
class EvalStep(Step):

    DETERMINISTIC: bool = True
    CACHEABLE = False
    FORMAT = JsonFormat()

    def run(  # type: ignore[override]
        self,
        config: Config,
        trainer: Lazy[LightningTrainer],
        model: Lazy[Model],
        datamodule: Lazy[PromptDataModule],
    ) -> List[Dict[str, float]]:
        pl.seed_everything(config.seed)

        datamodule = datamodule.construct(config=config)

        datamodule.prepare_data()
        datamodule.setup()

        trainer: LightningTrainer = trainer.construct(
            work_dir=self.work_dir,
            gpus=config.gpus,
            accelerator="gpu" if config.gpus else "cpu",
            auto_select_gpus=True,
            limit_test_batches=500,
        )

        model = model.construct(config=config, dataset=datamodule)

        output = trainer.test(model, dataloaders=datamodule.val_dataloader())

        return output
