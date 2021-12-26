from typing import Dict, List, Optional, Tuple

from tango import Step
from tango.common.lazy import Lazy
from tango.format import JsonFormat
from tango.integrations.pytorch_lightning.train import LightningTrainer

from ..data.config import Config
from ..data.t0_dataset import T0Dataset
from ..models.prefix_transformer import PrefixTransformer
from .train import TrainStep


@Step.register("train_one_dataset")
class TrainOneDatasetStep(Step):
    """
    A step that trains one T0 dataset, aggregating over all templates.
    """

    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = JsonFormat()

    def run(  # type: ignore[override]
        self,
        config: Config,
        trainer: Lazy[LightningTrainer],
        model: Lazy[PrefixTransformer],
        t0_dataset: T0Dataset,
        # optimizer: Lazy[Optimizer],
        # lr_schedule: Lazy[LRScheduler],
    ) -> Tuple[str, List[Dict]]:
        # TODO: we probably need different log directories for each run
        single_step = TrainStep()
        best_model_paths = []
        all_metrics = {}
        for task_name, datamodule in t0_dataset.data_modules.items():
            best_model_path, metrics = single_step.run(config, trainer, model, datamodule)
            best_model_paths.append(best_model_path)
            all_metrics[task_name] = metrics
        agg = self.aggregate(all_metrics)
        return agg, best_model_paths, all_metrics

    def aggregate(self, all_metrics: Dict[str, List]):
        pass  # TODO
