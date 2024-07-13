from typing import Any, Optional, Union, List, Dict

import pandas as pd

from alpaca_eval import constants, metrics, utils


def evaluate_alpacaeval(
    annotations: List[Dict],
    fn_metric: Union[str, callable] = "get_length_controlled_winrate" if constants.IS_ALPACA_EVAL_2 else "get_winrate",
    metric_kwargs: Optional[dict[str, Any]] = None,
):
    # annotations = pd.read_json(output_path / "annotations.json")
    annotations = pd.DataFrame(annotations)

    if isinstance(fn_metric, str):
        fn_metric_ = getattr(metrics, fn_metric)
    else:
        fn_metric_ = fn_metric

    res = fn_metric_(annotations, **(metric_kwargs or {}))

    # print(res)

    return res
