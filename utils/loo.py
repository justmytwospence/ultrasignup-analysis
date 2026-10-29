"""PSIS-LOO helpers shared by the notebooks."""

from __future__ import annotations

import arviz as az
import pandas as pd


def compare_table(loos: dict) -> pd.DataFrame:
    """az.compare's table (rank, elpd_loo, p_loo, elpd_diff, se, dse) for two or more models,
    and the same columns for a single model (az.compare needs at least two)."""
    if len(loos) > 1:
        return az.compare(loos)[["rank", "elpd_loo", "p_loo", "elpd_diff", "se", "dse"]]
    (name, loo), = loos.items()
    return pd.DataFrame({"rank": [0], "elpd_loo": [float(loo.elpd_loo)], "p_loo": [float(loo.p_loo)],
                         "elpd_diff": [0.0], "se": [float(loo.se)], "dse": [0.0]}, index=[name])
