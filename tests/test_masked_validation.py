"""Synthetic out-of-fit reconstruction tests: no database or production traces."""
from dataclasses import fields, replace
import json

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from utils import masked_validation as mv
from utils.dnf_model import build_unobserved_model


def processed_frame(*, courses=4, races=6, size=4):
    rows = []
    for c in range(courses):
        for race in range(races):
            did = 100 * c + race + 1
            finishers = size if np.isscalar(size) else size[race]
            for i in range(finishers + race % 3):
                finished = i < finishers
                rows.append(dict(event_distance_id=did, name=f"course-{c}", course_id=f"course-{c}",
                                 participant_id=f"runner-{i}" if finished else f"dnf-{did}-{i}",
                                 gender="M" if i % 2 == 0 else "F", age=30 + i,
                                 distance_km=[50., 80., 100., 160.][c % 4], year=2010 + race,
                                 finished=finished, reports_dnfs=True,
                                 time_ms=(20_000_000 + i * 100_000) if finished else np.nan,
                                 n_starters=9876, n_dnf=8765, irrelevant="source metadata"))
    return pd.DataFrame(rows)


@pytest.fixture
def frame():
    return processed_frame()


@pytest.fixture
def setup(frame):
    heldout = [2, 102, 202, 302]
    visible = mv.mask_results(frame, heldout)
    probabilities = pd.DataFrame(columns=["did", "p_reports_prior", "p_lists_dns"])
    data = mv.prepare_dnf_inputs(visible, probabilities, sorted(frame.name.unique()), known_nonreporting_ids=heldout)
    return frame, visible, data, heldout


def fake_posterior(data, *, draws=12, chains=2, level=-1.3, donor=0., tau=0.):
    shape = (chains, draws)
    scalars = {"tau_edition": tau, "dnf_gender_gap": 0., "dnf_age_linear": 0., "dnf_age_quadratic": 0.,
               "intercept": -1.5, "slope": .2, "dispersion": 10.}
    variables = {n: (("chain", "draw"), np.full(shape, v)) for n, v in scalars.items()}
    variables["course_dnf_level"] = (("chain", "draw", "course"), np.full((*shape, len(data.courses)), level))
    variables["course_effect"] = (("chain", "draw", "course"), np.zeros((*shape, len(data.courses))))
    variables["runner_dnf_effect"] = (("chain", "draw", "runner"), np.full((*shape, len(data.runners)), donor, dtype=np.float32))
    # These must NEVER enter prediction: an extreme fitted target race effect.
    variables["edition_dnf_effect"] = (("chain", "draw", "edition"), np.full((*shape, len(data.edition_ids)), 123.))
    return xr.Dataset(variables, coords={"chain": np.arange(chains) + 5, "draw": np.arange(draws) + 30,
                                         "course": data.courses, "runner": data.runners,
                                         "edition": data.edition_ids.astype(str)})


def test_manifest_seed_row_order_and_no_hidden_count_dependence(frame):
    courses = sorted(frame.name.unique())
    a = mv.make_race_manifest(frame, courses, seed=210)
    b = mv.make_race_manifest(frame.sample(frac=1, random_state=9), courses[::-1], seed=210)
    pd.testing.assert_frame_equal(a, b)
    changed = frame.copy()
    hidden = changed[~changed.finished].copy()
    hidden.participant_id = "hidden-" + hidden.participant_id
    changed = pd.concat([changed, hidden, hidden])
    changed.loc[~changed.finished, ["age", "n_dnf", "n_starters"]] = 99999
    pd.testing.assert_frame_equal(a, mv.make_race_manifest(changed, courses, seed=210))
    assert len(a[a.selected]) == round(.2 * len(a[a.eligible]))
    assert not {"n_dnf", "n_starters", "n_dns", "residual"} & set(a)
    assert a.attrs["actual_count"] == a.selected.sum()
    assert a.attrs["actual_fraction"] == a.actual_fraction.iloc[0]
    for _, group in a.groupby("course_id"):
        assert (~group.selected).sum() >= 2
    assert set(a.loc[a.did.isin([1, 101, 201, 301]), "eligible"]) == {True}  # genuine zeros


def test_manifest_quotas_edges_sparse_and_capacity():
    frame = processed_frame(races=9, size=[2, 2, 2, 4, 4, 4, 8, 8, 8])
    manifest = mv.make_race_manifest(frame, frame.name.unique(), fraction=.4, seed=211)
    edges = json.loads(manifest.finisher_edges_json.iloc[0])
    assert set(edges) == {"42-60", "60-90", "90-120", "120-165"}
    assert all(len(edge) == 4 for edge in edges.values())
    assert all(len(group.finisher_bin.unique()) == 3 for _, group in manifest.groupby("distance_bin"))
    allocation = json.loads(manifest.allocation_json.iloc[0])
    assert sum(v["quota"] for v in allocation.values()) == round(.4 * manifest.eligible.sum())
    assert all(v["actual"] == v["quota"] for v in allocation.values())
    # Capacity, not an implicit course/group holdout: two races always remain.
    small = processed_frame(courses=2, races=3)
    constrained = mv.make_race_manifest(small, small.name.unique(), fraction=.9, seed=212)
    assert constrained.selected.sum() == 2
    assert constrained.actual_fraction.iloc[0] == pytest.approx(1 / 3)
    assert any(v["deviation"] for v in constrained.attrs["allocation"].values())
    assert all(len(edges) == 1 for edges in constrained.attrs["finisher_edges"].values())
    # Sibling events (same year) are not forced into/out of a mask as a unit.
    assert any(0 < g.selected.sum() < len(g) for _, g in manifest.groupby("year"))


def test_manifest_filters_and_reallocation():
    frame = processed_frame(courses=2, races=6)
    # Course 0 has only two eligible reporting years; never selected.
    frame.loc[(frame.name == "course-0") & (frame.year >= 2012), "reports_dnfs"] = False
    manifest = mv.make_race_manifest(frame, ["course-0", "course-1"], fraction=.5, seed=213)
    assert not manifest.loc[manifest.course_id == "course-0", "eligible"].any()
    assert set(manifest.loc[manifest.selected, "course_id"]) == {"course-1"}
    assert len(manifest) == 8
    # One course has finite quota capacity; redistribute its unused quota elsewhere.
    frame = processed_frame(courses=2, races=9)
    frame = frame[~((frame.name == "course-0") & (frame.year >= 2013))]
    manifest = mv.make_race_manifest(frame, frame.name.unique(), fraction=.5, seed=214)
    assert manifest.selected.sum() == 6
    assert manifest.loc[manifest.course_id == "course-0", "selected"].sum() == 1
    assert manifest.loc[manifest.course_id == "course-1", "selected"].sum() == 5


@pytest.mark.parametrize("fraction", [0, 1, -1, np.nan])
def test_manifest_bad_fraction(frame, fraction):
    with pytest.raises(ValueError):
        mv.make_race_manifest(frame, frame.name.unique(), fraction=fraction, seed=1)


def test_manifest_impossible_inconsistent_distance_edges():
    tiny = processed_frame(courses=1, races=2)
    with pytest.raises(ValueError, match="No eligible"):
        mv.make_race_manifest(tiny, tiny.name.unique(), seed=1)
    frame = processed_frame()
    # Exact bin boundaries: 60 remains first, 90 second, 120 third, 165 fourth.
    for c, distance in enumerate([60., 90., 120., 165.]):
        frame.loc[frame.name == f"course-{c}", "distance_km"] = distance
    result = mv.make_race_manifest(frame, frame.name.unique(), seed=1)
    assert result.groupby("course_id").distance_bin.first().tolist() == ["42-60", "60-90", "90-120", "120-165"]
    frame.loc[0, "distance_km"] = 61
    with pytest.raises(ValueError, match="Inconsistent"):
        mv.make_race_manifest(frame, frame.name.unique(), seed=1)


def test_mask_and_truth_isolation(setup):
    frame, visible, data, heldout = setup
    original = frame.copy(deep=True)
    mv.mask_results(frame, heldout)
    pd.testing.assert_frame_equal(frame, original)
    assert len(visible) == len(frame) - len(frame[frame.event_distance_id.isin(heldout) & ~frame.finished])
    kept = visible[visible.event_distance_id.isin(heldout)]
    pd.testing.assert_series_equal(kept.time_ms.reset_index(drop=True),
                                   frame.loc[frame.event_distance_id.isin(heldout) & frame.finished, "time_ms"].reset_index(drop=True))
    assert not kept.reports_dnfs.any()
    assert kept.finished.all()
    assert not set(heldout) & set(data.starts.event_distance_id)
    assert not set(frame.loc[frame.event_distance_id.isin(heldout) & ~frame.finished, "participant_id"]) & set(data.runners)
    assert set(data.starts.columns) == set(mv.VISIBLE_FIELDS)
    assert "n_starters" not in data.editions
    assert data.editions.loc[heldout, "n_dnf"].eq(0).all()
    assert data.editions.loc[heldout, ["p_reports_prior", "p_lists_dns"]].isna().all().all()
    truth = mv.heldout_truth(frame, heldout[::-1])
    assert truth.did.tolist() == sorted(heldout)
    assert truth.n_dnf.tolist() == [1] * 4
    assert truth.n_finishers.tolist() == [4] * 4
    assert "n_dnf" not in visible
    assert "n_starters" not in visible


def assert_same_inputs(a, b):
    for field in fields(a):
        av, bv = getattr(a, field.name), getattr(b, field.name)
        if isinstance(av, pd.DataFrame):
            pd.testing.assert_frame_equal(av, bv)
        elif isinstance(av, pd.Index):
            pd.testing.assert_index_equal(av, bv)
        else:
            np.testing.assert_equal(av, bv)


def test_heldout_perturbation_entire_pipeline_invariance(setup):
    frame, visible, data, heldout = setup
    extra = frame.loc[frame.event_distance_id.isin(heldout) & ~frame.finished].copy()
    extra.participant_id = "totally-hidden-" + extra.participant_id
    extra.age = 94
    extra.gender = "F"
    altered = pd.concat([frame, extra], ignore_index=True)
    altered["n_dnf"] = 999_999
    altered["n_starters"] = 999_999
    masked = mv.mask_results(altered, heldout)
    # Raw source-only metadata is dropped/ignored, so inputs and counts are bitwise identical.
    pd.testing.assert_frame_equal(visible, masked)
    other = mv.prepare_dnf_inputs(masked, pd.DataFrame(columns=["did", "p_reports_prior", "p_lists_dns"]),
                                  data.courses, known_nonreporting_ids=heldout)
    assert_same_inputs(data, other)
    pd.testing.assert_frame_equal(mv.baseline_training_counts(visible, data.courses), mv.baseline_training_counts(masked, data.courses))
    post = fake_posterior(data)
    a = mv.predict_hidden_counts(post, data, heldout, seed=215, posterior_draws_per_chain=4, nodes=8)
    b = mv.predict_hidden_counts(post, other, heldout, seed=215, posterior_draws_per_chain=4, nodes=8)
    xr.testing.assert_identical(a, b)
    assert (mv.heldout_truth(altered, heldout).n_dnf > mv.heldout_truth(frame, heldout).n_dnf).all()


def test_prepare_coordinates_probabilities_and_shared_builder(frame):
    heldout = [2]
    frame.loc[frame.event_distance_id == 103, "reports_dnfs"] = False
    visible = mv.mask_results(frame, heldout)
    probabilities = pd.DataFrame({"did": [303, 103], "p_reports_prior": [.99, .12], "p_lists_dns": [.8, .34]})
    data = mv.prepare_dnf_inputs(visible, probabilities, frame.name.unique(), known_nonreporting_ids=heldout)
    assert data.editions.loc[103, "p_reports_prior"] == .12
    assert data.editions.loc[103, "p_lists_dns"] == .34
    assert data.editions.loc[303, ["p_reports_prior", "p_lists_dns"]].isna().all()
    assert not set(heldout) & set(data.starts.event_distance_id)
    assert np.array_equal(data.f_gender, data.runner_gender[data.f_runner])
    assert np.array_equal(data.s_gender, data.runner_gender[data.s_runner])
    model = build_unobserved_model(data)
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    assert "unreported" in model.named_vars
    # A reporting occurrence with no finishers belongs to start/race coordinates,
    # even though it does not appear among finisher-based edition_ids.
    zero = frame[(frame.event_distance_id == 1) & frame.finished].copy()
    zero.event_distance_id = 999
    zero.finished = False
    zero.time_ms = np.nan
    zero.participant_id = "zero-" + zero.participant_id
    extended = mv.prepare_dnf_inputs(pd.concat([visible, zero]), probabilities, data.courses, known_nonreporting_ids=heldout)
    assert 999 not in extended.edition_ids
    assert 999 in extended.starts.event_distance_id.values
    assert "999" in build_unobserved_model(extended).coords["edition"]


@pytest.mark.parametrize("case", ["missing_probability", "duplicate_probability", "nonfinite_probability", "bad_probability",
                                    "missing_finisher_course", "duplicate_start", "bad_gender", "no_training",
                                    "not_masked", "unknown_id", "bad_time"])
def test_prepare_rejects_invalid(frame, case):
    frame = mv.mask_results(frame, [2])
    probabilities = pd.DataFrame({"did": [103], "p_reports_prior": [.5], "p_lists_dns": [.4]})
    frame.loc[frame.event_distance_id == 103, "reports_dnfs"] = False
    courses = sorted(frame.name.unique())
    known = [2]
    if case == "missing_probability":
        probabilities = probabilities.iloc[:0]
    elif case == "duplicate_probability":
        probabilities = pd.concat([probabilities, probabilities])
    elif case == "nonfinite_probability":
        probabilities.loc[0, "p_lists_dns"] = np.nan
    elif case == "bad_probability":
        probabilities.loc[0, "p_reports_prior"] = 1.1
    elif case == "missing_finisher_course":
        frame = frame[frame.name != "course-0"]
    elif case == "duplicate_start":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif case == "bad_gender":
        frame.gender = "M"
    elif case == "no_training":
        frame.reports_dnfs = False
    elif case == "not_masked":
        frame.loc[frame.event_distance_id == 2, "reports_dnfs"] = True
    elif case == "unknown_id":
        known = [99999]
    elif case == "bad_time":
        frame.loc[0, "time_ms"] = np.nan
    with pytest.raises(ValueError):
        mv.prepare_dnf_inputs(frame, probabilities, courses, known_nonreporting_ids=known)


def test_varying_posted_gender_preserves_original_row_and_runner_semantics(frame):
    # Original 2.3 assigns runner gender at first appearance for centering, but
    # uses the posted row gender in the finish/DNF regressions. No row is dropped.
    frame.loc[frame.event_distance_id == 2, "gender"] = frame.loc[frame.event_distance_id == 2, "gender"].map({"M": "F", "F": "M"})
    empty = pd.DataFrame(columns=["did", "p_reports_prior", "p_lists_dns"])
    data = mv.prepare_dnf_inputs(frame, empty, frame.name.unique())
    runner = data.runners.get_loc("runner-0")
    assert data.runner_gender[runner] == 0
    finish_rows = (data.f_runner == runner) & (data.edition_ids[data.f_edition] == 2)
    start_rows = (data.s_runner == runner) & (data.starts.event_distance_id.to_numpy() == 2)
    assert data.f_gender[finish_rows].tolist() == [1]
    assert data.s_gender[start_rows].tolist() == [1]
    assert len(data.log_pace) == frame.finished.sum()
    assert len(data.starts) == len(frame)


def test_baseline_training_all_years_zeros_model_and_prior(setup):
    _, visible, data, heldout = setup
    visible.loc[visible.event_distance_id == 1, "year"] = 2001
    counts = mv.baseline_training_counts(visible, data.courses)
    assert set(heldout).isdisjoint(counts.did)
    assert 1 in counts.did.values
    assert counts.loc[counts.did == 1, "n_dnf"].item() == 0
    assert counts.loc[counts.did == 1, "year"].item() == 2001
    model = mv.build_count_baseline(counts)
    assert model.coords["course"] == tuple(data.courses)
    assert set(["intercept", "slope", "course_sd", "course_raw", "course_effect", "dispersion", "count"]) <= set(model.named_vars)
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    with model:
        prior = pm.sample_prior_predictive(draws=8, random_seed=216)
    assert prior.prior_predictive["count"].shape == (1, 8, len(counts))
    # Explicit fixed prior constants, not empirical centering.
    assert model["intercept"].owner.inputs[-2].eval() == -1.5


def test_baseline_nutpie_coordinate_smoke():
    # Execution/coordinate serialization check only, not scientific inference.
    counts = pd.DataFrame({"did": [101, 102, 201, 202], "course_id": ["a", "a", "b", "b"],
                           "distance_km": [50., 50., 100., 100.], "n_finishers": [5, 8, 4, 6],
                           "n_dnf": [0, 1, 2, 1]})
    model = mv.build_count_baseline(counts)
    assert model.coords["did"] == ("101", "102", "201", "202")
    assert all(type(value) is str for value in model.coords["did"])
    with model:
        trace = pm.sample(draws=5, tune=5, chains=2, cores=1, nuts_sampler="nutpie", random_seed=232,
                          progressbar=False, compute_convergence_checks=False)
    assert trace.posterior.sizes["chain"] == 2
    assert trace.posterior.sizes["draw"] == 5
    assert np.isfinite(trace.posterior.intercept).all()


def test_baseline_prediction_order_budget_reproducibility_and_moments(setup):
    frame, _, data, heldout = setup
    targets = mv.make_race_manifest(frame, data.courses, seed=217)
    targets = targets[targets.did.isin(heldout)]
    post = fake_posterior(data, draws=600)
    a = mv.predict_count_baseline(post, targets, seed=218, posterior_draws_per_chain=500)
    b = mv.predict_count_baseline(post, targets.iloc[::-1], seed=218, posterior_draws_per_chain=500)
    xr.testing.assert_identical(a, b)
    assert a.dims == ("chain", "draw", "did")
    assert np.issubdtype(a.dtype, np.integer)
    np.testing.assert_array_equal(a.draw, post.draw.values[np.linspace(0, 599, 500, dtype=int)])
    assert a.chain.values.tolist() == [5, 6]
    expected = 4 * np.exp(-1.5)
    assert float(a.sel(did=2).mean()) == pytest.approx(expected, abs=.12)
    assert float(a.sel(did=2).var()) == pytest.approx(expected + expected**2 / 10, abs=.2)
    with pytest.raises(ValueError, match="Unknown"):
        mv.predict_count_baseline(post.drop_sel(course=data.courses[0]), targets, seed=1, posterior_draws_per_chain=2)
    with pytest.raises(ValueError, match="exceed"):
        mv.predict_count_baseline(post, targets, seed=1, posterior_draws_per_chain=1000)
    bad = targets.copy()
    bad.loc[bad.index[0], "n_finishers"] = 0
    with pytest.raises(ValueError, match="F>0"):
        mv.predict_count_baseline(post, bad, seed=1, posterior_draws_per_chain=2)


def test_hidden_prediction_batch_order_and_fitted_race_invariance(setup):
    _, _, data, heldout = setup
    post = fake_posterior(data, tau=.5)
    a = mv.predict_hidden_counts(post, data, heldout, seed=219, posterior_draws_per_chain=5, nodes=24, race_batch=1)
    b = mv.predict_hidden_counts(post, data, heldout[::-1], seed=219, posterior_draws_per_chain=5, nodes=24, race_batch=3)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.target_probability, b.target_probability)
    assert a.did.values.tolist() == sorted(heldout)
    np.testing.assert_array_equal(a.draw, [30, 32, 35, 38, 41])
    post.edition_dnf_effect[:] = -123.
    c = mv.predict_hidden_counts(post, data, heldout, seed=219, posterior_draws_per_chain=5, nodes=24, race_batch=1)
    xr.testing.assert_identical(a, c)
    # Selecting only a subset of targets cannot change that target's substreams.
    subset = mv.predict_hidden_counts(post, data, [heldout[0]], seed=219, posterior_draws_per_chain=5, nodes=24)
    np.testing.assert_array_equal(a.sel(did=[heldout[0]]), subset)
    np.testing.assert_array_equal(a.sel(did=[heldout[0]]).target_probability, subset.target_probability)
    diagnostic = mv.target_probability_summary(a)
    assert diagnostic.dims == ("chain", "draw", "did")
    assert np.all((diagnostic.values > 0) & (diagnostic.values < 1))
    assert np.var(diagnostic.values) > .001  # fresh race effects actually used
    other = mv.predict_hidden_counts(post, data, heldout, seed=220, posterior_draws_per_chain=5, nodes=24)
    assert not np.array_equal(a, other)
    assert a.attrs["endpoint_clip_count"] == 0


def test_hidden_prediction_reads_runner_effects_one_draw_at_a_time(setup, monkeypatch):
    _, _, data, heldout = setup
    post = fake_posterior(data, draws=7, chains=2)
    original = xr.DataArray.values
    seen_shapes = []

    def inspect_values(array):
        if array.name == "runner_dnf_effect":
            assert array.dims == ("runner",)
            seen_shapes.append(array.shape)
        return original.fget(array)

    monkeypatch.setattr(xr.DataArray, "values", property(inspect_values, original.fset))
    mv.predict_hidden_counts(post, data, heldout, seed=230, posterior_draws_per_chain=3, nodes=2)
    assert seen_shapes == [(len(data.runners),)] * 6


def test_hidden_prediction_analytic_probability_count_moments_and_composition(setup):
    _, _, data, heldout = setup
    post = fake_posterior(data, draws=800, level=0., donor=0., tau=0.)
    pred = mv.predict_hidden_counts(post, data, [heldout[0]], seed=221, posterior_draws_per_chain=800, nodes=1)
    np.testing.assert_array_equal(pred.target_probability, .5)
    # NB(F=4, success=1-p=.5): E[K]=4, Var[K]=8.
    assert float(pred.mean()) == pytest.approx(4, abs=.25)
    assert float(pred.var()) == pytest.approx(8, abs=1.)
    post = fake_posterior(data, level=0.)
    post.dnf_gender_gap[:] = 2.
    post.dnf_age_linear[:] = .4
    post.dnf_age_quadratic[:] = -.3
    pred = mv.predict_hidden_counts(post, data, [2], seed=222, posterior_draws_per_chain=3, nodes=1)
    rows = np.flatnonzero(data.edition_ids[data.f_edition] == 2)
    from scipy.special import expit
    expected = expit((data.f_gender[rows] - .5) * 2 + data.f_age[rows] * .4 - data.f_age2[rows] * .3).mean()
    np.testing.assert_allclose(pred.target_probability, expected)


def test_hidden_prediction_start_frequency_donors_not_finisher_or_all_runner_pool(setup):
    _, _, data, _ = setup
    post = fake_posterior(data, level=0.)
    # Every finisher-only runner has a huge positive effect. Reporting starts are
    # intentionally 99% runner 0: sampling unique runners would be visibly wrong.
    post.runner_dnf_effect[:] = 15
    post.runner_dnf_effect.loc[dict(runner=data.runners[0])] = -4
    post.runner_dnf_effect.loc[dict(runner=data.runners[1])] = 4
    weighted = replace(data, s_runner=np.array([0] * 99 + [1]))
    prediction = mv.predict_hidden_counts(post, weighted, [2], seed=223, posterior_draws_per_chain=4, nodes=1000)
    from scipy.special import expit
    expected = .99 * expit(-4) + .01 * expit(4)
    assert float(prediction.target_probability.mean()) == pytest.approx(expected, abs=.005)
    # In particular, the target's fitted own runner effects are not used directly.
    assert float(prediction.target_probability.mean()) < .04


@pytest.mark.parametrize("case", ["unknown_target", "zero_finishers", "zero_nodes", "zero_batch", "no_donors", "nan_effect",
                                    "nan_scalar", "bad_tau", "bad_coordinate", "bad_donor"])
def test_hidden_prediction_invalid_cases(setup, case):
    _, _, data, _ = setup
    post = fake_posterior(data)
    ids, kwargs = [2], dict(seed=224, posterior_draws_per_chain=2, nodes=4)
    if case == "unknown_target":
        ids = [1]
    elif case == "zero_finishers":
        data = replace(data, f_edition=np.zeros_like(data.f_edition))
    elif case == "zero_nodes":
        kwargs["nodes"] = 0
    elif case == "zero_batch":
        kwargs["race_batch"] = 0
    elif case == "no_donors":
        data = replace(data, s_runner=np.array([], dtype=int))
    elif case == "nan_effect":
        post.runner_dnf_effect[:] = np.nan
    elif case == "nan_scalar":
        post.dnf_gender_gap[:] = np.nan
    elif case == "bad_tau":
        post.tau_edition[:] = -1
    elif case == "bad_coordinate":
        post = post.isel(runner=slice(None, None, -1))
    elif case == "bad_donor":
        data = replace(data, s_runner=np.array([-1]))
    with pytest.raises(ValueError):
        mv.predict_hidden_counts(post, data, ids, **kwargs)


def test_hidden_endpoint_clip_and_overflow_explicit(setup):
    _, _, data, _ = setup
    post = fake_posterior(data, level=-1000.)
    prediction = mv.predict_hidden_counts(post, data, [2], seed=225, posterior_draws_per_chain=3, nodes=1)
    assert prediction.attrs["endpoint_clip_count"] == 6
    np.testing.assert_array_equal(prediction.target_probability, np.finfo(float).eps)
    assert not prediction.values.any()
    # A saturated probability near one can overflow NB sampling at large F.
    # Failure must be explicit, not integer wraparound or silent capping.
    post.course_dnf_level[:] = 1000.
    prediction = mv.predict_hidden_counts(post, data, [2], seed=226, posterior_draws_per_chain=2, nodes=1)
    assert prediction.attrs["endpoint_clip_count"] == 4
    assert (prediction.values >= 0).all()


def test_integrated_target_moments_constant_probability_analytic_means(setup):
    _, _, data, heldout = setup
    post = fake_posterior(data, level=-.7, donor=.2)
    result = mv.integrated_target_moments(post, data, heldout, seed=240, posterior_draws_per_chain=5)
    from scipy.special import expit
    p = expit(-.7 + float(np.float32(.2)))
    np.testing.assert_allclose(result.mean_probability, p, rtol=1e-14)
    np.testing.assert_allclose(result.mean_count, 4 * p / (1 - p), rtol=1e-14)
    assert set(result.data_vars) == {"mean_probability", "mean_count"}
    assert result.mean_probability.dims == result.mean_count.dims == ("chain", "draw", "did")
    np.testing.assert_array_equal(result.chain, [5, 6])
    np.testing.assert_array_equal(result.draw, [30, 32, 35, 38, 41])
    np.testing.assert_array_equal(result.did, sorted(heldout))
    assert result.attrs["nodes"] == 64
    assert result.attrs["edition_nodes"] == 8
    assert "mixing diagnostics ONLY" in result.attrs["purpose"]
    assert "not scored predictions" in result.attrs["purpose"]


def test_integrated_target_moments_nonzero_edition_against_scipy_fixed_oracle(setup):
    from scipy.integrate import fixed_quad
    from scipy.special import expit

    _, _, data, _ = setup
    post = fake_posterior(data, level=-1., donor=.25, tau=.8)
    post.dnf_gender_gap[:] = 1.2
    post.dnf_age_linear[:] = .4
    post.dnf_age_quadratic[:] = -.3
    result = mv.integrated_target_moments(post, data, [2], seed=241, posterior_draws_per_chain=2,
                                          edition_nodes=32)
    rows = np.flatnonzero(data.edition_ids[data.f_edition] == 2)
    base = -1. + .25 + (data.f_gender[rows] - .5) * 1.2 + .4 * data.f_age[rows] - .3 * data.f_age2[rows]

    def integrand(z, count=False):
        p = expit(base[:, None] + .8 * z[None, :]).mean(axis=0)
        moment = len(rows) * p / (1 - p) if count else p
        return moment * np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)

    # Independent fixed-order integration of the standard-normal density on
    # [-12,12], not Hermite nodes/weights or the implementation's probability.
    expected_p = fixed_quad(integrand, -12, 12, n=256)[0]
    expected_count = fixed_quad(lambda z: integrand(z, count=True), -12, 12, n=256)[0]
    np.testing.assert_allclose(result.mean_probability, expected_p, rtol=1e-10)
    np.testing.assert_allclose(result.mean_count, expected_count, rtol=1e-10)
    assert expected_count > 1.1 * len(rows) * expected_p / (1 - expected_p)
    post.edition_dnf_effect[:] = -123.
    xr.testing.assert_identical(result, mv.integrated_target_moments(
        post, data, [2], seed=241, posterior_draws_per_chain=2, edition_nodes=32))


def test_integrated_target_moments_reproducible_common_donors_and_batch_chain_order(setup):
    _, _, data, heldout = setup
    post = fake_posterior(data, tau=.5)
    effects = np.linspace(-3, 3, len(data.runners), dtype=np.float32)
    post.runner_dnf_effect[:] = effects
    kwargs = dict(seed=242, posterior_draws_per_chain=5, nodes=13)
    result = mv.integrated_target_moments(post, data, heldout, **kwargs)
    xr.testing.assert_identical(result, mv.integrated_target_moments(post, data, heldout[::-1], **kwargs))
    # Constant posterior parameters give identical moments in EVERY chain/draw:
    # no residual donor resampling or fresh-effect simulation noise.
    for name in result.data_vars:
        np.testing.assert_array_equal(result[name], np.broadcast_to(result[name].values[:1, :1], result[name].shape))
    other = mv.integrated_target_moments(post, data, heldout, **{**kwargs, "seed": 243})
    assert not np.array_equal(result.mean_probability, other.mean_probability)
    # Batching targets into separate calls retains exactly the same donor pool.
    batches = [mv.integrated_target_moments(post, data, heldout[i:i + 2], **kwargs) for i in (2, 0)]
    xr.testing.assert_identical(result, xr.concat(batches, dim="did").sortby("did"))
    # Nonidentical posterior draws make accidental chain-position dependence visible.
    post.course_dnf_level.values[:] += np.arange(24).reshape(2, 12, 1) / 10
    result = mv.integrated_target_moments(post, data, heldout, **kwargs)
    reversed_chains = mv.integrated_target_moments(post.isel(chain=[1, 0]), data, heldout, **kwargs)
    xr.testing.assert_identical(result, reversed_chains.sel(chain=post.chain))
    # Different draw selectors use the same common donors, too.
    full = mv.integrated_target_moments(post, data, heldout, **{**kwargs, "posterior_draws_per_chain": 12})
    xr.testing.assert_identical(result, full.sel(draw=result.draw))


def test_integrated_target_moments_start_frequency_donors(setup):
    from scipy.special import expit

    _, _, data, _ = setup
    post = fake_posterior(data, level=0.)
    post.runner_dnf_effect[:] = 15
    post.runner_dnf_effect.loc[dict(runner=data.runners[0])] = -4
    post.runner_dnf_effect.loc[dict(runner=data.runners[1])] = 4
    weighted = replace(data, s_runner=np.array([0] * 99 + [1]))
    result = mv.integrated_target_moments(post, weighted, [2], seed=244, posterior_draws_per_chain=2)
    # Only 64 common donor starts: permit their binomial sampling error, but not
    # a unique-runner or finisher reference pool (either would be near/above .5).
    assert float(result.mean_probability.mean()) == pytest.approx(.99 * expit(-4) + .01 * expit(4), abs=.04)
    assert result.attrs["donor_population"] == "remaining_reporting_starts"


def test_integrated_target_moments_course_donors_and_default_oracle(setup):
    from scipy.special import expit

    _, _, data, heldout = setup
    post = fake_posterior(data, level=0.)
    post.runner_dnf_effect[:] = 0
    post.runner_dnf_effect.values[:, :, :4] = [-2, -1, 1, 2]
    # One distinct constant-risk donor per course, repeated as training starts.
    data = replace(data, s_course=np.repeat(np.arange(4), [3, 5, 2, 4]),
                   s_runner=np.repeat(np.arange(4), [3, 5, 2, 4]))
    kwargs = dict(seed=248, posterior_draws_per_chain=3, nodes=19)
    result = mv.integrated_target_moments(post, data, heldout, donor_scope="course", **kwargs)
    for did, effect in zip(heldout, [-2, -1, 1, 2]):
        p = expit(effect)
        np.testing.assert_allclose(result.mean_probability.sel(did=did), p)
        np.testing.assert_allclose(result.mean_count.sel(did=did), 4 * p / (1 - p))
    assert result.attrs["donor_scope"] == "course"
    # Adding/replacing other-course starts cannot affect this target's pool.
    altered = replace(data, s_runner=np.r_[data.s_runner[:3], np.full(50, 3)],
                      s_course=np.r_[data.s_course[:3], np.ones(50, dtype=int)])
    one = mv.integrated_target_moments(post, data, [2], donor_scope="course", **kwargs)
    xr.testing.assert_identical(one, mv.integrated_target_moments(post, altered, [2], donor_scope="course", **kwargs))
    # The default reproduces the original seeded all-start algorithm exactly.
    default = mv.integrated_target_moments(post, data, [2], **kwargs)
    xr.testing.assert_identical(default, mv.integrated_target_moments(post, data, [2], donor_scope="all", **kwargs))
    sampled = data.s_runner[mv._rng(248, "common", "common", 2, "integrated-donors").integers(0, len(data.s_runner), 19)]
    p = expit(post.runner_dnf_effect.values[0, 0, sampled]).astype(float).mean()
    np.testing.assert_allclose(default.mean_probability, p, rtol=1e-7)
    assert default.attrs["donor_scope"] == "all"


def test_integrated_target_moments_course_start_frequency(setup):
    from scipy.special import expit

    _, _, data, _ = setup
    post = fake_posterior(data, level=0.)
    post.runner_dnf_effect.values[:, :, :2] = [-4, 4]
    data = replace(data, s_runner=np.array([0] * 99 + [1, 1]),
                   s_course=np.array([0] * 100 + [1]))
    result = mv.integrated_target_moments(post, data, [2], seed=249, posterior_draws_per_chain=2,
                                           nodes=2000, donor_scope="course")
    assert float(result.mean_probability.mean()) == pytest.approx(.99 * expit(-4) + .01 * expit(4), abs=.005)


@pytest.mark.parametrize("case", ["scope", "empty_pool", "mixed_target", "start_shape", "start_negative",
                                 "start_out_of_range", "start_float", "finisher_shape", "finisher_invalid"])
def test_integrated_target_moments_course_invalid(setup, case):
    _, _, data, _ = setup
    post = fake_posterior(data)
    kwargs = dict(seed=250, posterior_draws_per_chain=2, donor_scope="course")
    if case == "scope":
        kwargs["donor_scope"] = "invalid"
    elif case == "empty_pool":
        data = replace(data, s_course=np.ones_like(data.s_course))
    elif case == "mixed_target":
        course = data.f_course.copy()
        course[np.flatnonzero(data.edition_ids[data.f_edition] == 2)[0]] = 1
        data = replace(data, f_course=course)
    elif case == "start_shape":
        data = replace(data, s_course=data.s_course[:-1])
    elif case == "start_negative":
        data = replace(data, s_course=np.full_like(data.s_course, -1))
    elif case == "start_out_of_range":
        data = replace(data, s_course=np.full_like(data.s_course, len(data.courses)))
    elif case == "start_float":
        data = replace(data, s_course=data.s_course.astype(float))
    elif case == "finisher_shape":
        data = replace(data, f_course=data.f_course[:-1])
    elif case == "finisher_invalid":
        data = replace(data, f_course=np.full_like(data.f_course, -1))
    with pytest.raises(ValueError):
        mv.integrated_target_moments(post, data, [2], **kwargs)


def test_integrated_target_moments_bounded_runner_and_finisher_arrays(setup, monkeypatch):
    _, _, data, _ = setup
    # Repeat finisher rows to require more than one 64-finisher chunk.
    data = replace(data, **{name: np.tile(getattr(data, name), 20)
                            for name in ("f_edition", "f_course", "f_gender", "f_age", "f_age2")})
    post = fake_posterior(data, tau=.4)
    original_values, original_expit = xr.DataArray.values, mv.expit
    runner_shapes, finisher_shapes = [], []

    def inspect_values(array):
        if array.name == "runner_dnf_effect":
            assert array.dims == ("runner",)
            runner_shapes.append(array.shape)
        return original_values.fget(array)

    def inspect_expit(values):
        assert values.ndim == 2 and values.shape[0] <= 64 and values.shape[1] == 64
        finisher_shapes.append(values.shape)
        return original_expit(values)

    monkeypatch.setattr(xr.DataArray, "values", property(inspect_values, original_values.fset))
    monkeypatch.setattr(mv, "expit", inspect_expit)
    result = mv.integrated_target_moments(post, data, [2], seed=245, posterior_draws_per_chain=3)
    assert runner_shapes == [(len(data.runners),)] * 6
    assert set(finisher_shapes) == {(64, 64), (16, 64)}
    assert len(finisher_shapes) == 6 * 8 * 2
    assert result.attrs["max_finisher_batch"] == 64


@pytest.mark.parametrize("coordinate", ["chain", "draw", "runner", "course"])
def test_integrated_target_moments_requires_explicit_coordinates(setup, coordinate):
    _, _, data, _ = setup
    post = fake_posterior(data).drop_vars(coordinate)
    with pytest.raises(ValueError, match=f"{coordinate} coordinates"):
        mv.integrated_target_moments(post, data, [2], seed=246, posterior_draws_per_chain=2)


@pytest.mark.parametrize("case", ["unknown_target", "zero_finishers", "zero_nodes", "zero_edition_nodes", "no_donors",
                                 "nan_effect", "nan_scalar", "bad_tau", "bad_coordinate", "bad_donor", "excess_draws"])
def test_integrated_target_moments_invalid_cases(setup, case):
    _, _, data, _ = setup
    post = fake_posterior(data)
    ids, kwargs = [2], dict(seed=247, posterior_draws_per_chain=2)
    if case == "unknown_target":
        ids = [1]
    elif case == "zero_finishers":
        data = replace(data, f_edition=np.zeros_like(data.f_edition))
    elif case == "zero_nodes":
        kwargs["nodes"] = 0
    elif case == "zero_edition_nodes":
        kwargs["edition_nodes"] = 0
    elif case == "no_donors":
        data = replace(data, s_runner=np.array([], dtype=int))
    elif case == "nan_effect":
        post.runner_dnf_effect[:] = np.nan
    elif case == "nan_scalar":
        post.dnf_gender_gap[:] = np.nan
    elif case == "bad_tau":
        post.tau_edition[:] = -1
    elif case == "bad_coordinate":
        post = post.isel(course=slice(None, None, -1))
    elif case == "bad_donor":
        data = replace(data, s_runner=np.array([-1]))
    elif case == "excess_draws":
        kwargs["posterior_draws_per_chain"] = 1000
    with pytest.raises(ValueError):
        mv.integrated_target_moments(post, data, ids, **kwargs)


def predictions(values, ids=(2, 1), model="joint"):
    return xr.DataArray(np.asarray(values).reshape(1, -1, len(ids)), dims=("chain", "draw", "did"),
                        coords={"chain": [0], "draw": np.arange(np.asarray(values).size // len(ids)), "did": list(ids)},
                        name="n_dnf", attrs={"model": model})


def test_score_exact_discrete_quantiles_wis_bias_and_id_alignment():
    pred = predictions([[0, 1], [1, 2], [2, 3], [3, 4]])
    truth = pd.DataFrame({"did": [1, 2], "n_dnf": [5, 1]})
    metadata = pd.DataFrame({"did": [1, 2], "course_id": ["b", "a"], "stratum": ["large", "small"]})
    score = mv.score_predictions(pred, truth, metadata).set_index("did")
    assert score.index.tolist() == [2, 1]
    assert score.loc[2, "median"] == 1
    assert score.loc[2, "mean"] == 1.5
    assert score.loc[2, "signed_mean_error"] == .5
    assert score.loc[2, "absolute_median_error"] == 0
    assert score.loc[2, "lower_50"] == 0
    assert score.loc[2, "upper_50"] == 2
    # Race 2: .25*2 + .1*3 + .03*3, divided by 3.5.
    assert score.loc[2, "wis"] == pytest.approx(.89 / 3.5)
    # Race 1: y=5, median=2, 50% [1,3], 80/94% [1,4].
    expected = (.5 * 3 + .25 * (2 + 4 * 2) + .1 * (3 + 10) + .03 * (3 + 2 / .06)) / 3.5
    assert score.loc[1, "wis"] == pytest.approx(expected)
    assert not score.loc[1, "coverage_94"]
    assert score.loc[2, "coverage_94"]
    assert score.loc[1, "signed_mean_error"] == -2.5
    # Truth row order and metadata order do not affect output.
    pd.testing.assert_frame_equal(score, mv.score_predictions(pred, truth.iloc[::-1], metadata.iloc[::-1]).set_index("did"))


@pytest.mark.parametrize("case", ["duplicate_truth", "duplicate_metadata", "duplicate_prediction", "missing_truth", "extra_metadata",
                                    "negative_count", "fractional_count", "nan_count", "nan_truth", "duplicate_chain"])
def test_score_rejects_bad_alignment_or_counts(case):
    pred = predictions([[1, 2], [2, 3]])
    truth = pd.DataFrame({"did": [1, 2], "n_dnf": [1, 2]})
    metadata = pd.DataFrame({"did": [1, 2], "course_id": ["a", "b"]})
    if case == "duplicate_truth":
        truth = pd.concat([truth, truth.iloc[[0]]])
    elif case == "duplicate_metadata":
        metadata = pd.concat([metadata, metadata.iloc[[0]]])
    elif case == "duplicate_prediction":
        pred = pred.assign_coords(did=[1, 1])
    elif case == "missing_truth":
        truth = truth.iloc[[0]]
    elif case == "extra_metadata":
        metadata.loc[2] = [3, "c"]
    elif case in ("negative_count", "fractional_count", "nan_count"):
        pred = pred.astype(float)
        pred.values[0, 0, 0] = {"negative_count": -1, "fractional_count": .5, "nan_count": np.nan}[case]
    elif case == "nan_truth":
        truth.loc[0, "n_dnf"] = np.nan
    elif case == "duplicate_chain":
        pred = xr.concat([pred, pred], dim="chain")
    with pytest.raises(ValueError):
        mv.score_predictions(pred, truth, metadata)


def test_sum_uses_joint_draw_not_marginal_endpoints():
    pred = predictions([[0, 10], [10, 0], [0, 10], [10, 0]])
    pred = pred.assign_coords(target_probability=(("chain", "draw", "did"), np.ones(pred.shape) * .5))
    total = mv.sum_predictions(pred)
    assert total.dims == ("chain", "draw")
    np.testing.assert_array_equal(total, 10)
    assert "target_probability" not in total.coords
    assert np.quantile(total, [.03, .97]).tolist() == [10, 10]
    assert np.quantile(pred, .03, axis=(0, 1)).sum() == 0  # demonstrably different


def test_course_bootstrap_pairing_sensitivity_and_strata():
    # Three races, course a has two: race-weighted difference=2, course-weighted=3.
    common = pd.DataFrame({"did": [1, 2, 3], "course_id": ["a", "a", "b"], "distance_bin": ["short", "short", "long"],
                           "finisher_bin": [0, 1, 1], "stratum": ["s0", "s1", "l1"],
                           "coverage_50": [False, False, True], "coverage_80": [True, True, True],
                           "coverage_94": [True, True, True], "n_dnf": [1, 1, 1]})
    joint = common.assign(model="joint", wis=[1., 1., 7.])
    baseline = common.assign(model="baseline", wis=[1., 1., 1.])
    scores = pd.concat([joint, baseline], ignore_index=True)
    a = mv.summarize_scores(scores, seed=227, bootstraps=1000)
    b = mv.summarize_scores(scores.sample(frac=1, random_state=8), seed=227, bootstraps=1000)
    for key in a:
        pd.testing.assert_frame_equal(a[key], b[key])
    comparison = a["comparison"].set_index("weighting")
    assert comparison.loc["race_weighted", "mean_wis_difference"] == 2
    assert comparison.loc["equal_course", "mean_wis_difference"] == 3
    assert comparison.loc["race_weighted", "lower_94"] == 0
    assert comparison.loc["race_weighted", "upper_94"] == 6
    assert "evaluation" in comparison.interval_kind.iloc[0]
    coverage = a["coverage"].set_index(["model", "level"])
    assert coverage.loc[("joint", 50), "coverage"] == pytest.approx(1 / 3)
    assert coverage.loc[("joint", 50), "lower_94"] == 0
    assert coverage.loc[("joint", 50), "upper_94"] == 1
    assert {"by_distance_bin", "by_finisher_bin", "by_stratum", "paired_by_distance_bin"} <= set(a)
    with pytest.raises(ValueError, match="Duplicate"):
        mv.summarize_scores(pd.concat([scores, scores.iloc[[0]]]), seed=1)
    with pytest.raises(ValueError, match="match"):
        mv.summarize_scores(scores.iloc[:-1], seed=1)
    mismatch = scores.copy()
    mismatch.loc[0, "n_dnf"] = 999
    with pytest.raises(ValueError, match="metadata mismatch"):
        mv.summarize_scores(mismatch, seed=1)


def fingerprint_inputs(setup):
    frame, visible, data, _ = setup
    manifest = mv.make_race_manifest(frame, data.courses, seed=228)
    # This test's manually specified heldout differs from the random split; declare
    # precisely the selected IDs to align fingerprint reporting-training exclusion.
    manifest.selected = manifest.did.isin(data.known_nonreporting_ids)
    reporting = pd.DataFrame({"did": manifest.loc[~manifest.selected, "did"], "series_id": 1,
                               "year": 2011, "log_distance": 0., "n_starters": 10,
                               "kind": "reports", "known_reporting": True})
    settings = dict(model_version="v2", extraction_version="v1", prediction_version=mv.PREDICTION_VERSION,
                    seeds={"split": 228, "fit": 229}, sampler={"chains": 2, "draws": 50, "tune": 50},
                    coordinates={"courses": data.courses.tolist(), "runners": data.runners.tolist()})
    return visible, reporting, manifest, settings


def test_fingerprint_permutation_hidden_truth_and_consumed_changes(setup):
    visible, reporting, manifest, settings = fingerprint_inputs(setup)
    original = mv.validation_fingerprint(visible, reporting, manifest, settings)
    assert len(original) == 64
    assert original == mv.validation_fingerprint(visible.sample(frac=1, random_state=1),
                                                 reporting.sample(frac=1, random_state=2), manifest.iloc[::-1], settings)
    visible["n_dnf"] = 54321
    visible["n_starters"] = 12345
    assert original == mv.validation_fingerprint(visible, reporting, manifest, settings)
    for field in ("age", "time_ms", "distance_km"):
        changed = visible.copy()
        changed.loc[0, field] += 1
        assert original != mv.validation_fingerprint(changed, reporting, manifest, settings)
    changed = reporting.copy()
    changed.loc[changed.index[0], "n_starters"] += 1
    assert original != mv.validation_fingerprint(visible, changed, manifest, settings)
    for key, value in (("model_version", "v3"), ("seeds", {"split": 999}), ("sampler", {"draws": 51}),
                       ("coordinates", {"runners": settings["coordinates"]["runners"][::-1]})):
        assert original != mv.validation_fingerprint(visible, reporting, manifest, {**settings, key: value})
    with pytest.raises(ValueError, match="settings missing"):
        mv.validation_fingerprint(visible, reporting, manifest, {})
    leaked = reporting.copy()
    leaked.loc[leaked.index[0], "did"] = manifest.loc[manifest.selected, "did"].iloc[0]
    with pytest.raises(ValueError, match="Held-out IDs"):
        mv.validation_fingerprint(visible, leaked, manifest, settings)


def tiny_trace(*, chains=2, draws=4):
    return az.InferenceData(posterior=xr.Dataset({"theta": (("chain", "draw"), np.ones((chains, draws))),
                                                  "effect": (("chain", "draw", "runner"), np.ones((chains, draws, 3), dtype=np.float32))},
                                                 coords={"chain": np.arange(chains), "draw": np.arange(draws), "runner": ["a", "b", "c"]}),
                            sample_stats=xr.Dataset({"diverging": (("chain", "draw"), np.zeros((chains, draws), dtype=bool))}))


def test_atomic_artifacts_trace_metadata_and_roundtrip(tmp_path):
    path = tmp_path / "nested" / "config.json"
    mv.atomic_json(path, {"array": np.array([1, 2]), "number": np.int64(3)})
    assert json.loads(path.read_text()) == {"array": [1, 2], "number": 3}
    frame = pd.DataFrame({"did": [1, 2], "selected": [True, False]})
    mv.atomic_write_frame(tmp_path / "manifest.parquet", frame)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "manifest.parquet"), frame)
    pred = predictions([[1, 2], [2, 3]])
    predpath = tmp_path / "predictions.nc"
    assert mv.load_predictions_cache(predpath, metadata={}) is None
    mv.atomic_write_predictions(predpath, pred, metadata={"fingerprint": "abc"})
    loaded = mv.load_predictions_cache(predpath, metadata={"fingerprint": "abc"})
    np.testing.assert_array_equal(loaded, pred)
    with pytest.raises(ValueError, match="metadata mismatch"):
        mv.load_predictions_cache(predpath, metadata={"fingerprint": "different"})
    trace = tiny_trace()
    tracepath = tmp_path / "trace.nc"
    metadata = dict(fingerprint="abc", chains=2, draws=4, scalar_names=["theta"])
    assert mv.load_cached_trace(tracepath, metadata, chains=2, draws=4) is None
    mv.save_trace_atomic(trace, tracepath, metadata)
    assert "validation_metadata" not in trace.posterior.attrs
    loaded = mv.load_cached_trace(tracepath, metadata, chains=2, draws=4)
    xr.testing.assert_equal(loaded.posterior.effect, trace.posterior.effect)
    loaded.close()
    with pytest.raises(ValueError, match="metadata mismatch"):
        mv.load_cached_trace(tracepath, {**metadata, "fingerprint": "different"}, chains=2, draws=4)
    with pytest.raises(ValueError, match="disagree"):
        mv.load_cached_trace(tracepath, metadata, chains=4, draws=4)
    with pytest.raises(ValueError, match="declare"):
        mv.save_trace_atomic(trace, tracepath, {})


@pytest.mark.parametrize("case", ["wrong_chain_count", "wrong_draw_count", "nonfinite_scalar", "missing_variable", "stats_missing", "stats_short"])
def test_trace_rejects_incomplete_and_retains_good_cache(tmp_path, case):
    trace = tiny_trace()
    path = tmp_path / "trace.nc"
    metadata = dict(chains=2, draws=4, scalar_names=["theta"])
    mv.save_trace_atomic(trace, path, metadata)
    before = path.read_bytes()
    if case == "wrong_chain_count":
        trace = tiny_trace(chains=1)
    elif case == "wrong_draw_count":
        trace = tiny_trace(draws=3)
    elif case == "nonfinite_scalar":
        trace.posterior.theta[1, 3] = np.nan
    elif case == "missing_variable":
        trace.posterior = trace.posterior.rename(theta="other")
    elif case == "stats_missing":
        trace = az.InferenceData(posterior=trace.posterior)
    elif case == "stats_short":
        trace.sample_stats = trace.sample_stats.isel(draw=slice(0, 2))
    with pytest.raises(ValueError):
        mv.save_trace_atomic(trace, path, metadata)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".trace.nc.*"))


def test_atomic_failed_writer_keeps_previous_artifact(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    mv.atomic_json(path, {"value": "good"})
    original = path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated disk rename failure")

    monkeypatch.setattr(mv.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        mv.atomic_json(path, {"value": "bad"})
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json"]


def test_corrupt_cache_is_not_silently_a_miss(tmp_path):
    path = tmp_path / "corrupt.nc"
    path.write_bytes(b"not a netcdf")
    with pytest.raises((OSError, ValueError)):
        mv.load_cached_trace(path, {}, chains=2, draws=4)
    with pytest.raises((OSError, ValueError)):
        mv.load_predictions_cache(path, metadata={})
    assert path.read_bytes() == b"not a netcdf"


def test_store_reservation_recovery_refuses_partial_and_never_deletes(tmp_path, monkeypatch):
    from utils import sampling
    store = tmp_path / "draws.zarr"
    meta = {"fingerprint": "abc", "draws": 4, "chains": 2}
    assert mv.recover_trace_store(store, metadata=meta, chains=2, draws=4, scalar_names=["theta"]) is None
    assert mv.prepare_trace_store(store, metadata=meta) == store
    with pytest.raises(FileExistsError, match="retain"):
        mv.prepare_trace_store(store, metadata=meta)
    store.mkdir()
    sentinel = store / "partial-draws"
    sentinel.write_bytes(b"preserve")
    monkeypatch.setattr(sampling, "load_store", lambda *args: None)
    with pytest.raises(ValueError, match="Incomplete/unreadable.*retained"):
        mv.recover_trace_store(store, metadata=meta, chains=2, draws=4, scalar_names=["theta"])
    assert sentinel.read_bytes() == b"preserve"
    monkeypatch.setattr(sampling, "load_store", lambda *args: tiny_trace())
    recovered = mv.recover_trace_store(store, metadata=meta, chains=2, draws=4, scalar_names=["theta"])
    assert recovered.posterior.sizes["chain"] == 2
    with pytest.raises(ValueError, match="metadata mismatch"):
        mv.recover_trace_store(store, metadata={**meta, "fingerprint": "wrong"}, chains=2, draws=4, scalar_names=["theta"])
    assert sentinel.exists()


@pytest.mark.parametrize("ids", [[1, 1], [1, "1"], [True], [1.5], [np.nan], [-1]])
def test_invalid_ids_never_silently_coerce(frame, ids):
    with pytest.raises(ValueError):
        mv.mask_results(frame, ids)


def test_prediction_cache_keeps_diagnostic_probabilities(tmp_path, setup):
    _, _, data, heldout = setup
    post = fake_posterior(data)
    pred = mv.predict_hidden_counts(post, data, heldout, seed=231, posterior_draws_per_chain=3, nodes=2)
    path = tmp_path / "prediction.nc"
    metadata = {"fingerprint": "abc", "seed": 231, "nodes": 2}
    mv.atomic_write_predictions(path, pred, metadata=metadata)
    loaded = mv.load_predictions_cache(path, metadata=metadata)
    np.testing.assert_array_equal(loaded, pred)
    xr.testing.assert_equal(loaded.target_probability, pred.target_probability)
    corrupted = loaded.copy(deep=True)
    corrupted.coords["target_probability"].values[0, 0, 0] = np.nan
    # Simulate a corrupt external artifact, bypassing the validating writer.
    corrupted.to_netcdf(tmp_path / "corrupt-probabilities.nc")
    with pytest.raises(ValueError, match="probability diagnostic"):
        mv.load_predictions_cache(tmp_path / "corrupt-probabilities.nc", metadata=metadata)


def test_full_snapshot_sha_is_separate_provenance(tmp_path):
    path = tmp_path / "snapshot"
    path.write_bytes(b"source snapshot")
    import hashlib
    assert mv.source_snapshot_sha(path) == hashlib.sha256(b"source snapshot").hexdigest()
    path.write_bytes(b"other hidden counts")
    assert mv.source_snapshot_sha(path) != hashlib.sha256(b"source snapshot").hexdigest()
