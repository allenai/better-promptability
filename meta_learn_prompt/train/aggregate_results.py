from typing import Any, Dict, List, Tuple

from tango import Format, JsonFormat, Step


@Step.register("aggregate_results")
class AggregateResults(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT: Format = JsonFormat()
    VERSION = "002"

    def run(self, results: List[Tuple[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
        accuracy_total = 0.0
        for _, epoch_metrics in results:
            accuracy_total += epoch_metrics[-1]["best_categorical_accuracy"]
        return {
            "categorical_accuracy": accuracy_total / len(results),
        }
