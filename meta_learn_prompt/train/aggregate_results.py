from typing import Any, Dict, List

from tango import Format, JsonFormat, Step


@Step.register("aggregate_results")
class AggregateResults(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT: Format = JsonFormat()
    VERSION = "001"

    def run(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # TODO
        return {
            "accuracy": 0.0,
        }
