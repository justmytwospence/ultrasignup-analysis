"""Synthetic checks of the extracted 2.1 model (never import a notebook)."""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.special import betaln, expit, log_expit
from scipy.stats import betabinom

from _dnf_model_fixtures import reporting_editions
from utils.reporting_model import build_reporting_model, load_reporting_editions, reporting_probabilities

REFERENCE = json.loads((Path(__file__).parent / "fixtures/dnf_model_reference.json").read_text())["models"]["reporting"]


def test_reporting_logp_and_gradient_match_frozen_original_builder():
    model = build_reporting_model(reporting_editions())
    assert [v.name for v in model.free_RVs] == REFERENCE["free_rvs"]
    assert [v.name for v in model.value_vars] == REFERENCE["value_vars"]
    assert model.coords["series"] == (9, 2)  # first appearance, not sorted
    assert model.coords["year"] == tuple(range(2005, 2012))
    logp, gradient = model.compile_logp(), model.compile_dlogp()
    for ref in REFERENCE["points"]:
        point = {k: np.asarray(v) for k, v in ref["point"].items()}
        np.testing.assert_allclose(logp(point), ref["logp"], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(gradient(point), ref["gradient"], rtol=1e-10, atol=1e-10)


def test_three_likelihood_branches_match_independent_density():
    frame = reporting_editions()
    model = build_reporting_model(frame)
    # At the original initial point reporting and DNS probabilities are both 1/2.
    p, kappa = expit(-1.5), np.exp(np.log(10) + 0.5)
    a, b = p * kappa, (1 - p) * kappa
    q = betaln(a, b + frame["n_finishers"].to_numpy()) - betaln(a, b)
    expected = [
        log_expit(0) + betabinom.logpmf(2, 5, a, b) + log_expit(0),
        log_expit(0) + q[1] + log_expit(0),
        np.logaddexp(log_expit(0) + q[2:] + log_expit(0), log_expit(0)).sum(),
    ]
    actual = model.compile_logp(vars=model.potentials, sum=False)(model.initial_point())
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("series_effect,distance_on_reporting", [(False, False), (False, True), (True, False), (True, True)])
def test_comparison_flags_preserve_original_builder(series_effect, distance_on_reporting):
    # This executes only the frozen original builder AST body on tiny inputs.
    import pymc as pm
    import pytensor.tensor as pt

    env = dict(np=np, pd=pd, pm=pm, pt=pt, editions=reporting_editions())
    exec(REFERENCE["original_builder_source"], env)
    original = env["build_reporting_model"](series_effect, distance_on_reporting)
    extracted = build_reporting_model(reporting_editions(), series_effect=series_effect, distance_on_reporting=distance_on_reporting)
    assert ("series_raw" in extracted.named_vars) == series_effect
    assert ("report_distance" in extracted.named_vars) == distance_on_reporting
    assert [v.name for v in extracted.value_vars] == [v.name for v in original.value_vars]
    point = extracted.initial_point()
    np.testing.assert_allclose(extracted.compile_logp()(point), original.compile_logp()(point), rtol=1e-11)


def _posterior():
    # Reverse coordinate order, non-default sample labels and >1 batching block.
    rng = np.random.default_rng(sum(map(ord, "reporting-extraction-test")))
    return xr.Dataset(
        {
            "year_effect": (("chain", "draw", "year"), rng.normal(size=(2, 129, 4))),
            "series_effect": (("chain", "draw", "series"), rng.normal(size=(2, 129, 2))),
            "report_distance": (("chain", "draw"), rng.normal(size=(2, 129))),
            "dns_intercept": (("chain", "draw"), rng.normal(size=(2, 129))),
            "dns_field": (("chain", "draw"), rng.normal(size=(2, 129))),
        },
        coords={"chain": [3, 8], "draw": np.arange(129) * 2, "year": [2011, 2005, 2010, 2006], "series": [2, 9]},
    )


def test_prediction_is_label_aligned_uses_all_draws_and_no_outcome_counts():
    post, frame = _posterior(), reporting_editions().iloc[[3, 0, 2, 1]].copy()
    actual = reporting_probabilities(post, frame)
    assert actual["did"].tolist() == frame["did"].tolist()
    for i, row in enumerate(frame.itertuples()):
        logits = (post["year_effect"].sel(year=max(2005, row.year))
                  + post["series_effect"].sel(series=row.series_id)
                  + post["report_distance"] * row.log_distance)
        dns = post["dns_intercept"] + post["dns_field"] * np.log(row.n_starters / 50)
        np.testing.assert_allclose(actual.iloc[i]["p_reports_prior"], expit(logits).mean())
        np.testing.assert_allclose(actual.iloc[i]["p_lists_dns"], expit(dns).mean())
    frame["n_dnf"], frame["n_finishers"] = 100000, -99
    pd.testing.assert_frame_equal(actual, reporting_probabilities(post, frame))
    # No DNF model parameters or count columns are necessary for these probabilities.
    minimal = frame[["did", "year", "series_id", "log_distance", "n_starters"]]
    pd.testing.assert_frame_equal(actual, reporting_probabilities(post, minimal))


def test_prediction_handles_reporting_comparison_flags():
    post = _posterior().drop_vars(["series_effect", "report_distance"])
    actual = reporting_probabilities(post, reporting_editions())
    expected = expit(post["year_effect"].sel(year=2005)).mean().item()
    np.testing.assert_allclose(actual.iloc[0]["p_reports_prior"], expected)


@pytest.mark.parametrize("column,value", [("series_id", 999), ("year", 2020)])
def test_prediction_rejects_unseen_coordinates(column, value):
    frame = reporting_editions()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match="Unseen"):
        reporting_probabilities(_posterior(), frame)


def test_missing_series_is_explicit_error_and_early_year_bin_exists():
    frame = reporting_editions()
    frame.loc[0, "series_id"] = None
    with pytest.raises(ValueError, match="Missing series"):
        build_reporting_model(frame)
    frame = reporting_editions().iloc[:1]
    assert build_reporting_model(frame).coords["year"] == (2005,)


def test_heldout_true_starter_total_cannot_affect_training_or_probabilities():
    original, changed = reporting_editions(), reporting_editions()
    changed.loc[changed["did"] == 103, ["n_starters", "n_dnf"]] = [9999, 9995]
    train_a = original[original["did"] != 103]
    train_b = changed[changed["did"] != 103]
    pd.testing.assert_frame_equal(train_a, train_b)
    a, b = build_reporting_model(train_a), build_reporting_model(train_b)
    np.testing.assert_equal(a.compile_logp()(a.initial_point()), b.compile_logp()(b.initial_point()))
    pd.testing.assert_frame_equal(reporting_probabilities(_posterior(), train_a), reporting_probabilities(_posterior(), train_b))


def test_loader_preserves_original_sql_filters_and_classification():
    base = dict(did=1, course_id="course", series_id=7, series_name="Trail", race_name="Trail 50K",
                year=2020, distance_km=50.0, n_starters=10, n_finishers=8, n_dnf=2, n_dns=0,
                reports_dnfs=True, results_state="public", race_format="distance", virtual=False, cancelled=False)
    changes = [{}, {"n_dnf": None, "n_dns": 2}, {"n_dnf": None, "n_dns": None, "reports_dnfs": False},
               {"race_name": "Bike Race"}, {"results_state": "hidden"}, {"race_format": "fixed_time"},
               {"virtual": True}, {"cancelled": True}, {"distance_km": 41.9}, {"distance_km": 165.1},
               {"year": None}, {"n_starters": 0}, {"race_name": None}, {"distance_km": 42.0}, {"distance_km": 165.0}]
    rows = pd.DataFrame([{**base, **change, "did": i + 1} for i, change in enumerate(changes)])
    with duckdb.connect() as con:
        con.register("input_rows", rows)
        con.execute("CREATE TABLE races AS SELECT * FROM input_rows")
        actual = load_reporting_editions(con)
        # The helper must not close the caller's connection.
        assert con.execute("SELECT 1").fetchone() == (1,)
    assert actual["did"].tolist() == [1, 2, 3, 14, 15]
    assert actual["kind"].tolist() == ["reports", "reports, zero DNFs", "unknown", "reports", "reports"]
    assert actual["known_reporting"].tolist() == [True, True, False, True, True]
    np.testing.assert_allclose(actual["log_distance"], np.log(actual["distance_km"] / 50))
