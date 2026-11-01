"""Shared reporting model from notebook 2.1, with unchanged priors and likelihood."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from scipy.special import expit

from utils.data_processing import is_running_race


def load_reporting_editions(con) -> pd.DataFrame:
    """Load public running editions from the caller's analyst-schema connection."""
    editions = con.execute("""
        SELECT did, course_id, series_id, series_name, race_name, year, distance_km,
               n_starters, n_finishers, coalesce(n_dnf, 0) AS n_dnf, coalesce(n_dns, 0) AS n_dns,
               reports_dnfs
        FROM main.races
        WHERE results_state = 'public' AND race_format = 'distance'
          AND NOT virtual AND NOT cancelled
          AND distance_km BETWEEN 42 AND 165
          AND year IS NOT NULL AND n_starters > 0
    """).df()
    editions = editions[editions["race_name"].map(is_running_race)].reset_index(drop=True)
    editions["kind"] = np.select(
        [editions["n_dnf"] > 0, editions["n_dns"] > 0],
        ["reports", "reports, zero DNFs"], default="unknown",
    )
    editions["known_reporting"] = editions["kind"] != "unknown"
    editions["log_distance"] = np.log(editions["distance_km"] / 50)
    return editions


def _validate_editions(editions):
    if editions["series_id"].isna().any():
        raise ValueError("Missing series identifiers are not supported")
    if editions["year"].isna().any():
        raise ValueError("Missing year coordinates are not supported")
    if editions["did"].isna().any() or editions["did"].duplicated().any():
        raise ValueError("Edition did identifiers must be present and unique")
    for column in ("log_distance", "n_starters"):
        if not np.isfinite(editions[column].to_numpy()).all():
            raise ValueError(f"{column} must be finite")
    if (editions["n_starters"] <= 0).any():
        raise ValueError("n_starters must be positive")


def build_reporting_model(
    editions: pd.DataFrame, *, series_effect: bool = True, distance_on_reporting: bool = True,
) -> pm.Model:
    """Build 2.1's model; flags remove terms for its existing comparisons.

    Series retain first-appearance order. The 2005 coordinate includes all earlier
    years. Priors and the three observation branches are the original model's.
    """
    _validate_editions(editions)
    if editions.empty:
        raise ValueError("At least one reporting-model edition is required")
    series_idx, series_ids = pd.factorize(editions["series_id"])
    years = np.arange(2005, max(2005, editions["year"].max()) + 1)
    year_idx = (editions["year"].clip(lower=2005) - years[0]).to_numpy()
    log_distance = editions["log_distance"].to_numpy()
    log_field = np.log(editions["n_starters"].to_numpy() / 50)
    _n = editions["n_starters"].to_numpy()
    _k = editions["n_dnf"].to_numpy()
    _f = editions["n_finishers"].to_numpy()
    _lists_dns = (editions["n_dns"] > 0).to_numpy()
    _kind = editions["kind"].to_numpy()
    if not np.isin(_kind, ["reports", "reports, zero DNFs", "unknown"]).all():
        raise ValueError("Unrecognized reporting kind")
    _rep = np.flatnonzero(_kind == "reports")
    _zero = np.flatnonzero(_kind == "reports, zero DNFs")
    _unk = np.flatnonzero(_kind == "unknown")

    def _log_sigmoid(x):
        return -pt.softplus(-x)

    def _betaln(a, b):
        return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)

    with pm.Model(coords={"year": years, "series": series_ids}) as model:
        _year_effect = pm.Normal("year_effect", 0.0, 5.0, dims="year")
        _logit_r = _year_effect[year_idx]
        if series_effect:
            _series_sd = pm.HalfNormal("series_sd", 2.0)
            _series_raw = pm.Normal("series_raw", 0.0, 1.0, dims="series")
            _series_effect = pm.Deterministic("series_effect", _series_sd * _series_raw, dims="series")
            _logit_r = _logit_r + _series_effect[series_idx]
        if distance_on_reporting:
            _report_distance = pm.Normal("report_distance", 0.0, 1.0)
            _logit_r = _logit_r + _report_distance * log_distance

        _dnf_intercept = pm.Normal("dnf_intercept", -1.5, 1.0)
        _dnf_distance = pm.Normal("dnf_distance", 0.0, 1.0)
        _kappa = pm.LogNormal("kappa", np.log(10.0), 1.0)
        _p = pm.math.invlogit(_dnf_intercept + _dnf_distance * log_distance)
        _a, _b = _p * _kappa, (1 - _p) * _kappa
        _dns_intercept = pm.Normal("dns_intercept", 0.0, 1.5)
        _dns_field = pm.Normal("dns_field", 0.0, 1.0)
        _logit_d = _dns_intercept + _dns_field * log_field
        _log_r, _log_not_r = _log_sigmoid(_logit_r), _log_sigmoid(-_logit_r)
        _log_d, _log_not_d = _log_sigmoid(_logit_d), _log_sigmoid(-_logit_d)
        _log_q0 = _betaln(_a, _b + _f) - _betaln(_a, _b)
        _bb = pm.logp(pm.BetaBinomial.dist(alpha=_a[_rep], beta=_b[_rep], n=_n[_rep]), _k[_rep])
        pm.Potential("reports", (_log_r[_rep] + _bb + pt.switch(_lists_dns[_rep], _log_d[_rep], _log_not_d[_rep])).sum())
        pm.Potential("reports_zero", (_log_r[_zero] + _log_q0[_zero] + _log_d[_zero]).sum())
        pm.Potential("unknown", pm.math.logaddexp(_log_r[_unk] + _log_q0[_unk] + _log_not_d[_unk], _log_not_r[_unk]).sum())
    return model


def reporting_probabilities(posterior: xr.Dataset, editions: pd.DataFrame) -> pd.DataFrame:
    """Posterior means of *prior* reporting and conditional DNS probabilities.

    Uses every retained draw, in bounded chain/draw/row batches, and coordinate
    labels rather than training-row positions. No observed DNF/finisher counts
    enter this calculation. This preserves the upstream plug-in approximation.
    """
    _validate_editions(editions)
    years = editions["year"].clip(lower=2005).to_numpy()
    series = editions["series_id"].to_numpy()
    for dim, requested in (("year", years), ("series", series)):
        if dim not in posterior.coords or not posterior.indexes[dim].is_unique:
            raise ValueError(f"Missing or duplicate posterior {dim} coordinates")
        if not np.isin(requested, posterior[dim].values).all():
            raise ValueError(f"Unseen {dim} coordinates in requested editions")
    n_draws = posterior.sizes.get("chain", 0) * posterior.sizes.get("draw", 0)
    if not n_draws:
        raise ValueError("Posterior must contain retained chain/draw samples")
    totals = np.zeros((2, len(editions)), dtype=float)
    for chain in range(posterior.sizes["chain"]):
        for start in range(0, posterior.sizes["draw"], 128):
            post = posterior.isel(chain=chain, draw=slice(start, start + 128))
            for row in range(0, len(editions), 2048):
                stop = min(row + 2048, len(editions))
                rows = editions.iloc[row:stop]
                selector = xr.DataArray(years[row:stop], dims="edition")
                logit_r = post["year_effect"].sel(year=selector).transpose("draw", "edition").values
                if "series_effect" in post:
                    selector = xr.DataArray(series[row:stop], dims="edition")
                    logit_r = logit_r + post["series_effect"].sel(series=selector).transpose("draw", "edition").values
                if "report_distance" in post:
                    logit_r = logit_r + post["report_distance"].values[:, None] * rows["log_distance"].to_numpy()
                logit_d = (post["dns_intercept"].values[:, None]
                           + post["dns_field"].values[:, None] * np.log(rows["n_starters"].to_numpy() / 50))
                totals[0, row:stop] += expit(logit_r).sum(axis=0)
                totals[1, row:stop] += expit(logit_d).sum(axis=0)
    return pd.DataFrame({"did": editions["did"].to_numpy(),
                         "p_reports_prior": totals[0] / n_draws,
                         "p_lists_dns": totals[1] / n_draws})
