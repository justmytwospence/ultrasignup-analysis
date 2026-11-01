"""Tiny synthetic v2 extraction, isolation and runner-transformation tests."""

from dataclasses import fields, replace
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from _dnf_model_fixtures import dnf_input_dict
from utils.dnf_model import DNFInputs, DNF_SCALARS, FINISH_SCALARS, build_unobserved_model, derive_runner_effects

REFERENCE = json.loads((Path(__file__).parent / "fixtures/dnf_model_reference.json").read_text())["models"]["dnf"]


def _data():
    return DNFInputs(**dnf_input_dict())


def test_public_input_contract_matches_approved_plan():
    assert [f.name for f in fields(DNFInputs)] == [
        "courses", "runners", "course_x", "runner_gender", "log_pace", "f_course", "f_runner", "f_gender", "f_x",
        "f_age", "f_age2", "edition_ids", "f_edition", "s_course", "s_runner", "s_gender", "s_age", "dnf", "starts",
        "editions", "unknown", "known_nonreporting_ids",
    ]
    assert _data().known_nonreporting_ids == ()


def test_joint_logp_gradient_order_and_coordinates_match_frozen_original():
    data = _data()
    model = build_unobserved_model(data)
    assert [v.name for v in model.free_RVs] == REFERENCE["free_rvs"]
    assert [v.name for v in model.value_vars] == REFERENCE["value_vars"]
    assert model.coords == {"gender": ("M", "F"), "course": ("course-a", "course-b"),
                            "runner": (10, 11, 12, 13), "edition": ("101", "102", "103", "104")}
    assert set(FINISH_SCALARS + DNF_SCALARS) <= model.named_vars.keys()
    assert [v.name for v in model.observed_RVs] == ["log_pace", "dnf"]
    assert {v.name for v in model.deterministics} == {
        "course_effect", "runner_effect", "rho_course", "course_dnf_effect", "rho_runner", "runner_dnf_effect", "edition_dnf_effect",
    }
    logp, gradient = model.compile_logp(), model.compile_dlogp()
    for ref in REFERENCE["points"]:
        point = {k: np.asarray(v) for k, v in ref["point"].items()}
        np.testing.assert_allclose(logp(point), ref["logp"], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(gradient(point), ref["gradient"], rtol=1e-10, atol=1e-10)


def test_naive_branch_matches_original_builder():
    import pandas as pd
    import pymc as pm
    import pytensor.tensor as pt
    from utils import finish_time_model as ftm

    env = dict(np=np, pd=pd, pm=pm, pt=pt, ftm=ftm, **dnf_input_dict())
    exec(REFERENCE["original_builder_source"], env)
    original = env["build_unobserved_model"]("naive")
    model = build_unobserved_model(_data(), unknown="naive")
    point = {k: np.asarray(v) for k, v in REFERENCE["points"][1]["point"].items()}
    np.testing.assert_allclose(model.compile_logp()(point), original.compile_logp()(point), rtol=1e-11)
    np.testing.assert_allclose(model.compile_dlogp()(point), original.compile_dlogp()(point), rtol=1e-10, atol=1e-10)


def test_varying_posted_row_genders_match_original_without_normalization():
    import pandas as pd
    import pymc as pm
    import pytensor.tensor as pt
    from utils import finish_time_model as ftm

    inputs = dnf_input_dict()
    # Runner 10 is centered with first-observed M, but later posts F; runner 11
    # likewise has a reporting start posted M despite being centered with F.
    inputs["f_gender"][-2] = 1
    inputs["s_gender"][1] = 0
    before = {name: inputs[name].copy() for name in ("runner_gender", "f_gender", "s_gender")}
    env = dict(np=np, pd=pd, pm=pm, pt=pt, ftm=ftm, **inputs)
    exec(REFERENCE["original_builder_source"], env)
    original = env["build_unobserved_model"]()
    extracted = build_unobserved_model(DNFInputs(**inputs))
    assert [v.name for v in extracted.value_vars] == [v.name for v in original.value_vars]
    original_logp, extracted_logp = original.compile_logp(), extracted.compile_logp()
    original_grad, extracted_grad = original.compile_dlogp(), extracted.compile_dlogp()
    for ref in REFERENCE["points"]:
        point = {k: np.asarray(v) for k, v in ref["point"].items()}
        np.testing.assert_allclose(extracted_logp(point), original_logp(point), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(extracted_grad(point), original_grad(point), rtol=1e-10, atol=1e-10)
    for name, values in before.items():
        np.testing.assert_array_equal(inputs[name], values)


def test_reporting_only_skips_unknown_potential_and_retains_zero_finisher_race():
    data = _data()
    data.unknown[:] = False
    data.editions["reports_dnfs"] = True
    data.editions = data.editions.drop(columns=["p_reports_prior", "p_lists_dns"])
    model = build_unobserved_model(data)
    assert model.potentials == []
    assert "104" in model.coords["edition"]  # DNF-only reporting race
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_entirely_masked_race_has_exactly_zero_dnf_or_unknown_contribution():
    data = replace(_data(), known_nonreporting_ids=(103,))
    # No reporting-model values needed at all when every silent race is known.
    data.editions = data.editions.drop(columns=["p_reports_prior", "p_lists_dns"])
    model = build_unobserved_model(data)
    assert model.potentials == []
    assert "103" in model.coords["edition"]
    # Edition 103 retains all finisher pace observations, no reporting starts.
    assert model["log_pace"].eval().shape == (len(data.log_pace),)
    assert model["dnf"].eval().shape == (len(data.dnf),)
    point = model.initial_point()
    likelihood = model.compile_logp(vars=model.observed_RVs)
    before = likelihood(point)
    point["edition_raw"][2] = 4.0
    assert likelihood(point) == before  # exact zero, not clipped reporting r=1e-9
    grad = model.compile_dlogp(vars=[model["edition_raw"]])(model.initial_point())
    assert grad[2] == 0.0
    # Arbitrary evaluator-only counts and reporting probabilities are not consumed.
    data.editions["n_dnf"] = [1000, 1000, 1000000]
    data.editions["p_reports_prior"] = np.nan
    data.editions["p_lists_dns"] = np.nan
    changed = build_unobserved_model(data)
    np.testing.assert_equal(model.compile_logp()(model.initial_point()), changed.compile_logp()(changed.initial_point()))


def test_known_nonreporter_is_excluded_but_genuine_silence_still_contributes():
    data = replace(_data(), known_nonreporting_ids=(103,))
    # Hide reporting race 102 too, but leave it genuinely silent, not known.
    keep = data.starts["event_distance_id"].to_numpy() != 102
    data.starts = data.starts.loc[keep].reset_index(drop=True)
    for name in ("s_course", "s_runner", "s_gender", "s_age", "dnf"):
        setattr(data, name, getattr(data, name)[keep])
    data.unknown[1] = True
    data.editions.loc[102, "reports_dnfs"] = False
    data.editions.loc[103, ["p_reports_prior", "p_lists_dns"]] = np.nan
    model = build_unobserved_model(data)
    assert [v.name for v in model.potentials] == ["unreported"]
    point = model.initial_point()
    fn = model.compile_logp(vars=model.potentials)
    before = fn(point)
    point["edition_raw"][2] = 7.0
    assert fn(point) == before
    point["edition_raw"][1] = 7.0
    assert fn(point) != before


@pytest.mark.parametrize("column", ["p_reports_prior", "p_lists_dns"])
@pytest.mark.parametrize("value", [np.nan, -0.1, 1.1, np.inf])
def test_genuine_silence_requires_finite_reporting_probabilities(column, value):
    data = _data()
    data.editions.loc[103, column] = value
    with pytest.raises(ValueError, match=column):
        build_unobserved_model(data)


def test_missing_probability_column_fails_explicitly():
    data = _data()
    data.editions = data.editions.drop(columns="p_lists_dns")
    with pytest.raises(ValueError, match="p_lists_dns"):
        build_unobserved_model(data)


@pytest.mark.parametrize("mode", ["bad", "", None])
def test_unknown_mode_rejected(mode):
    with pytest.raises(ValueError, match="unknown must"):
        build_unobserved_model(_data(), unknown=mode)


@pytest.mark.parametrize("name,value", [("f_course", -1), ("s_course", 2), ("f_runner", -1), ("s_runner", 4),
                                        ("f_edition", 3), ("f_gender", 2), ("s_gender", -1)])
def test_invalid_indices_rejected(name, value):
    data = _data()
    getattr(data, name)[0] = value
    with pytest.raises(ValueError, match=name):
        build_unobserved_model(data)


def test_no_reporting_starts_rejected():
    data = _data()
    data.starts = data.starts.iloc[:0]
    for name in ("s_course", "s_runner", "s_gender", "s_age", "dnf"):
        setattr(data, name, getattr(data, name)[:0])
    with pytest.raises(ValueError, match="reporting training start"):
        build_unobserved_model(data)


@pytest.mark.parametrize("gender", [np.zeros(4, dtype=int), np.ones(4, dtype=int), np.array([0, 1, -1, 1]), np.array([0., 1., 0., 1.])])
def test_invalid_gender_normalization_rejected(gender):
    with pytest.raises(ValueError, match="gender"):
        build_unobserved_model(replace(_data(), runner_gender=gender))


def test_known_reporting_start_and_invalid_known_id_rejected():
    with pytest.raises(ValueError, match="silent edition"):
        build_unobserved_model(replace(_data(), known_nonreporting_ids=(101,)))
    with pytest.raises(ValueError, match="silent edition"):
        build_unobserved_model(replace(_data(), known_nonreporting_ids=(999,)))
    data = replace(_data(), known_nonreporting_ids=(101,))
    data.unknown[0] = True
    with pytest.raises(ValueError, match="reporting starts"):
        build_unobserved_model(data)


def _raw_posterior():
    rng = np.random.default_rng(sum(map(ord, "runner-extraction-test")))
    return xr.Dataset({
        "runner_raw": (("chain", "draw", "runner"), rng.normal(size=(2, 3, 4))),
        "runner_dnf_raw": (("chain", "draw", "runner"), rng.normal(size=(2, 3, 4))),
        "omega": (("chain", "draw"), rng.uniform(0.1, 0.4, size=(2, 3))),
        "omega_dnf": (("chain", "draw"), rng.uniform(0.2, 0.9, size=(2, 3))),
        "rho_runner": (("chain", "draw"), rng.uniform(-0.8, 0.8, size=(2, 3))),
        "dnf_runner_pace_quadratic": (("chain", "draw"), rng.normal(size=(2, 3))),
    }, coords={"chain": [2, 7], "draw": [10, 20, 30], "runner": [10, 11, 12, 13]})


@pytest.mark.parametrize("variant", ["full", "uncorrelated", "pace_only"])
def test_runner_effect_derivation_matches_original_float32_transform(variant, monkeypatch):
    post, gender = _raw_posterior(), _data().runner_gender
    if variant == "uncorrelated":
        post = post.drop_vars(["rho_runner", "dnf_runner_pace_quadratic"])
    if variant == "pace_only":
        post = post.drop_vars("runner_dnf_raw")
    before = post.copy(deep=True)
    avg = np.eye(2)[gender] / np.bincount(gender, minlength=2)
    raw = post["runner_raw"].values
    z1 = (raw - (raw @ avg)[..., gender]).astype(np.float32)
    sc = lambda name: post[name].values[..., None].astype(np.float32)
    pace = sc("omega") * z1
    expected_dnf = None
    if "runner_dnf_raw" in post:
        raw = post["runner_dnf_raw"].values
        z2 = (raw - (raw @ avg)[..., gender]).astype(np.float32)
        if "rho_runner" in post:
            r = sc("rho_runner")
            expected_dnf = sc("omega_dnf") * (r * z1 + np.sqrt(1 - r**2) * z2)
            expected_dnf += sc("dnf_runner_pace_quadratic") * (z1**2 - 1)
        else:
            expected_dnf = sc("omega_dnf") * z2
    getter = xr.DataArray.values.fget

    def require_chain_slice(array):
        if array.name in ("runner_raw", "runner_dnf_raw"):
            assert "chain" not in array.dims, "Raw effects must be sliced before loading .values"
        return getter(array)

    with monkeypatch.context() as patch:
        patch.setattr(xr.DataArray, "values", property(require_chain_slice))
        result = derive_runner_effects(post, gender)
    np.testing.assert_array_equal(result["runner_effect"], pace)
    assert result["runner_effect"].dtype == np.float32
    for group in (0, 1):
        np.testing.assert_allclose(result["runner_effect"].values[..., gender == group].mean(-1), 0, atol=2e-8)
    if expected_dnf is not None:
        np.testing.assert_array_equal(result["runner_dnf_effect"], expected_dnf)
        assert result["runner_dnf_effect"].dtype == np.float32
    assert "runner_raw" not in result and "runner_dnf_raw" not in result
    for dim in ("chain", "draw", "runner"):
        xr.testing.assert_identical(result[dim], post[dim])
    xr.testing.assert_identical(post, before)


def test_runner_effect_derivation_rejects_missing_gender_and_wrong_length():
    with pytest.raises(ValueError, match="Both gender"):
        derive_runner_effects(_raw_posterior(), np.zeros(4, dtype=int))
    with pytest.raises(ValueError, match="length"):
        derive_runner_effects(_raw_posterior(), np.array([0, 1]))
