"""Synthetic mathematical oracles for the isolated entrant-population prototype.

No database, historical trace, or notebook is used here. Adaptive integration and
finite enumeration are deliberately independent of the model's quadrature and
symbolic graph. The single 10+10 nutpie run checks execution, not convergence or
scientific recovery.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, fields, replace
from itertools import product
import json
from math import factorial

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad, simpson
from scipy.special import expit, gammaln, logit, logsumexp
from scipy.stats import gamma, nbinom, norm, poisson
import xarray as xr

from utils import joint_field_model as jfm


PARAMETER_DIMS = {
    "course_mean": ("course",), "course_dnf_level": ("course",),
    "edition_pace": ("edition",), "edition_dnf": ("edition",),
    "runner_z1": ("runner",), "runner_z2": ("runner",),
    "entry_course": ("course",), "anonymous_course": ("course",),
    "theta_course": ("course", "group"),
    "age_linear": ("sex",), "age_quadratic": ("sex",), "sigma": ("sex",),
    **{name: () for name in (
        "pace_gap", "exponent_gap", "dnf_gender_gap", "dnf_age_linear",
        "dnf_age_quadratic", "omega", "omega_dnf", "rho_runner", "quadratic",
        "entry_mean", "entry_pace", "anonymous_mean", "anonymous_distance",
        "field_shape",
    )},
}


def _complete_data():
    return jfm.JointFieldData(
        editions=pd.DataFrame([
            ("report", "c-a", 50., True), ("target-a", "c-a", 50., True),
            ("target-b", "c-b", 100., True), ("zero", "c-b", 100., True),
        ], columns=["did", "course_id", "distance_km", "reported"]),
        registry=pd.DataFrame([
            ("r0", 1), ("r1", 5), ("r2", 3), ("unused", 0),
        ], columns=["runner_id", "group"]),
        candidates=pd.DataFrame([
            ("report", "r0", "finish", 2.0), ("report", "r1", "dnf", np.nan),
            ("report", "r2", "nonstarter", np.nan),
            ("target-a", "r0", "finish", 2.2), ("target-a", "r1", "dnf", np.nan),
            ("target-a", "r2", "nonstarter", np.nan),
            ("target-b", "r0", "dnf", np.nan), ("target-b", "r1", "nonstarter", np.nan),
        ], columns=["did", "runner_id", "state", "log_pace"]),
        anonymous=pd.DataFrame([
            ("a0", "report", 1, "finish", 1.9),
            ("a1", "report", 4, "finish", 2.1),
            ("a2", "report", 0, "dnf", np.nan),
            ("a3", "report", 5, "dnf", np.nan),
            ("a4", "target-a", 0, "finish", 2.05),
            ("a5", "target-a", 3, "finish", 2.3),
            ("a6", "target-a", 2, "dnf", np.nan),
            ("a7", "target-b", 5, "dnf", np.nan),
        ], columns=["observation_id", "did", "group", "state", "log_pace"]),
    )


@pytest.fixture
def complete():
    return _complete_data()


@pytest.fixture
def data(complete):
    return jfm.mask_field_editions(complete, ["target-a", "target-b", "zero"])


def _copy(data):
    return replace(data, **{f.name: getattr(data, f.name).copy(deep=True) for f in fields(data)})


def _ids(data):
    return {
        "course": sorted(data.editions.course_id.astype(str).unique()),
        "edition": sorted(data.editions.did.astype(str).unique()),
        "runner": sorted(data.registry.runner_id.astype(str).unique()),
    }


def _parameters(data, *, homogeneous=False):
    sizes = {key: len(value) for key, value in _ids(data).items()}
    c, e, r = (sizes[key] for key in ("course", "edition", "runner"))
    parameters = {
        "course_mean": np.linspace(2., 2.15, c),
        "course_dnf_level": np.linspace(-1., -.6, c),
        "edition_pace": np.linspace(-.04, .05, e),
        "edition_dnf": np.linspace(-.2, .25, e),
        "runner_z1": np.linspace(-.7, .8, r),
        "runner_z2": np.linspace(.6, -.5, r),
        "entry_course": np.linspace(-.15, .2, c),
        "anonymous_course": np.linspace(-.1, .15, c),
        "theta_course": np.tile(np.array([.1, .15, .2, .25, .18, .12]), (c, 1)),
        "age_linear": np.array([.025, .035]),
        "age_quadratic": np.array([.012, .009]),
        "sigma": np.array([.24, .31]),
        "pace_gap": .08, "exponent_gap": .04, "dnf_gender_gap": .2,
        "dnf_age_linear": .07, "dnf_age_quadratic": .03,
        "omega": .28, "omega_dnf": .55, "rho_runner": -.35, "quadratic": .18,
        "entry_mean": logit(.4), "entry_pace": -.3,
        "anonymous_mean": np.log(3.), "anonymous_distance": .2, "field_shape": 4.,
    }
    if homogeneous:
        for name in ("edition_pace", "edition_dnf", "entry_course", "anonymous_course",
                     "age_linear", "age_quadratic"):
            parameters[name] = np.zeros_like(parameters[name])
        for name in ("pace_gap", "exponent_gap", "dnf_gender_gap", "dnf_age_linear",
                     "dnf_age_quadratic", "omega", "omega_dnf", "quadratic",
                     "entry_pace", "anonymous_distance"):
            parameters[name] = 0.
        parameters["course_mean"][:] = 2.
        parameters["course_dnf_level"][:] = logit(.3)
        parameters["sigma"][:] = .3
    return parameters


def _posterior(data, parameters=None, *, draws=12, chains=2):
    """Physical fixed draws, with nondefault labels to expose positional mistakes."""
    parameters = _parameters(data) if parameters is None else parameters
    # Only obtain label conventions from the model, never numerical predictions.
    model = jfm.build_joint_field_model(data, quadrature_order=3)
    coords = {key: list(value) for key, value in model.coords.items() if key in {
        "course", "edition", "runner", "sex", "group",
    }}
    coords.update(chain=np.arange(chains) * 7 + 3, draw=np.arange(draws) * 2 + 101)
    variables = {}
    for name, dims in PARAMETER_DIMS.items():
        value = np.asarray(parameters[name])
        variables[name] = (("chain", "draw", *dims),
                           np.broadcast_to(value, (chains, draws, *value.shape)).copy())
    return xr.Dataset(variables, coords=coords)


def _predict(posterior, data, *, seed=104, draws=12):
    return jfm.predict_missing_field_counts(
        posterior, data, seed=seed, draws_per_chain=draws, quadrature_order=25,
    )


def _fingerprint(data, **overrides):
    options = dict(quadrature_order=15, sampler="nutpie", seed=103, version="test-v1")
    options.update(overrides)
    return jfm.field_fingerprint(data, options.pop("priors", jfm.JointFieldPrior()), **options)


def _normal_expectation(function):
    """Independent adaptive integral, with no Gauss-Hermite nodes from the model."""
    def outer(z1):
        value = quad(lambda z2: function(z1, z2) * norm.pdf(z2), -np.inf, np.inf,
                     epsabs=2e-9, epsrel=2e-8)[0]
        return value * norm.pdf(z1)
    return quad(outer, -np.inf, np.inf, epsabs=2e-9, epsrel=2e-8)[0]


def _outcome_function(parameters, data, did, group):
    """Literal scientific formula; intentionally no production parameter helper."""
    ids = _ids(data)
    edition = data.editions.set_index("did").loc[did]
    c, e = ids["course"].index(str(edition.course_id)), ids["edition"].index(str(did))
    sex, age = divmod(group, 3)
    age = (-2., 0., 2.)[age]
    h = (-.5, .5)[sex]
    x = np.log(edition.distance_km / 50.)
    p = parameters
    base_mu = (p["course_mean"][c] + h * (p["pace_gap"] + p["exponent_gap"] * x)
               + p["age_linear"][sex] * age + p["age_quadratic"][sex] * age**2
               + p["edition_pace"][e])
    base_eta = (p["course_dnf_level"][c] + h * p["dnf_gender_gap"]
                + p["dnf_age_linear"] * age + p["dnf_age_quadratic"] * age**2
                + p["edition_dnf"][e])
    def outcome(z1, z2):
        mu = base_mu + p["omega"] * z1
        eta = (base_eta + p["omega_dnf"] * (p["rho_runner"] * z1
               + np.sqrt(1. - p["rho_runner"]**2) * z2)
               + p["quadratic"] * (z1**2 - 1.))
        return mu, expit(eta), p["sigma"][sex]
    return outcome


# Named participation: normalized elementary probabilities and exhaustive collapse.


def test_named_states_partition_and_suppressed_hidden_probability():
    entry = np.array([-2., 0., 1.4])
    dnf = np.array([.8, -.7, 2.1])
    pi, p = expit(entry), expit(dnf)
    logs = jfm.named_state_logprobs(entry, dnf)
    assert set(logs) == {"finish", "dnf", "nonstarter", "unseen"}
    np.testing.assert_allclose(np.exp(logs["finish"]), pi * (1. - p), atol=1e-15)
    np.testing.assert_allclose(np.exp(logs["dnf"]), pi * p, atol=1e-15)
    np.testing.assert_allclose(np.exp(logs["nonstarter"]), 1. - pi, atol=1e-15)
    np.testing.assert_allclose(logsumexp([logs[key] for key in ("finish", "dnf", "nonstarter")], axis=0), 0., atol=1e-15)
    np.testing.assert_allclose(np.exp(logs["unseen"]), (1. - pi) + pi * p, atol=1e-15)
    np.testing.assert_allclose(jfm.named_hidden_probability(entry, dnf), pi * p / (1. - pi + pi * p))


@pytest.mark.parametrize("entry,dnf", list(product([-1000., 1000.], repeat=2)))
def test_named_extreme_logits_stay_finite(entry, dnf):
    logs = jfm.named_state_logprobs(entry, dnf)
    assert all(np.isfinite(value).all() for value in logs.values())
    log_pi = -np.logaddexp(0., -entry)
    log_not_pi = -np.logaddexp(0., entry)
    log_p = -np.logaddexp(0., -dnf)
    expected = np.exp(log_pi + log_p - np.logaddexp(log_not_pi, log_pi + log_p))
    assert jfm.named_hidden_probability(entry, dnf) == pytest.approx(expected)


@pytest.mark.parametrize("states", [("unseen", "unseen"), ("finish", "unseen"), ()])
def test_two_candidate_collapsed_likelihood_and_hidden_moments(data, states):
    candidates = pd.DataFrame([
        ("target-a", f"r{i}", state, 2.1 if state == "finish" else np.nan)
        for i, state in enumerate(states)
    ], columns=data.candidates.columns)
    tiny = replace(data, candidates=candidates, anonymous=data.anonymous.iloc[:0].copy())
    parameters = _parameters(tiny, homogeneous=True)
    parameters["entry_pace"] = .35
    parameters["omega_dnf"] = .4
    parameters["quadratic"] = .12
    ids = _ids(tiny)
    pi, dnf = [], []
    for i in range(len(states)):
        r = ids["runner"].index(f"r{i}")
        group = int(tiny.registry.set_index("runner_id").loc[f"r{i}", "group"])
        pi.append(expit(parameters["entry_mean"] + .35 * parameters["runner_z1"][r]))
        dnf.append(_outcome_function(parameters, tiny, "target-a", group)(
            parameters["runner_z1"][r], parameters["runner_z2"][r])[1])
    # Outcomes are known nonstarter, DNF, finish. Retain only configurations
    # compatible with the suppressed observations, then normalize explicitly.
    mass, first, second = 0., 0., 0.
    for configuration in product(range(3), repeat=len(states)):
        weight, hidden = 1., 0
        for i, outcome in enumerate(configuration):
            weight *= (1. - pi[i], pi[i] * dnf[i], pi[i] * (1. - dnf[i]))[outcome]
            if states[i] == "finish":
                weight *= float(outcome == 2) * norm.pdf(2.1, 2., .3)
            else:
                weight *= float(outcome != 2)
                hidden += outcome == 1
        mass += weight
        first += weight * hidden
        second += weight * hidden**2
    components = jfm.field_loglik_components_numpy(parameters, tiny, quadrature_order=25)
    assert float(components.named.sel(did="target-a")) == pytest.approx(np.log(mass), abs=1e-12)
    conditional = [pi[i] * dnf[i] / (1. - pi[i] + pi[i] * dnf[i])
                   for i, state in enumerate(states) if state == "unseen"]
    assert first / mass == pytest.approx(sum(conditional))
    assert second / mass - (first / mass)**2 == pytest.approx(sum(h * (1. - h) for h in conditional))
    # Connect the enumerated moment to the real predictor, not just a helper.
    prediction = _predict(_posterior(tiny, parameters), tiny)
    q = jfm.field_loglik_components_numpy(parameters, tiny, quadrature_order=25).Q.sel(did="target-a").item()
    anonymous_mean = parameters["field_shape"] * (1. - q) / (parameters["field_shape"] / 3. + q)
    np.testing.assert_allclose(prediction.expected_total.sel(did="target-a"), first / mass + anonymous_mean)


# Anonymous point-process counts: adaptive Gamma integration and enumeration.


@pytest.mark.parametrize("n,t,shape,rate", [(0, 0., 4., 2.), (0, .7, 4., 1.3), (3, 1., 2.5, .8), (7, .4, 4., 3.)])
def test_gamma_intensity_integral_against_adaptive_quadrature(n, t, shape, rate):
    expected, error = quad(lambda intensity: intensity**n * np.exp(-t * intensity)
                           * gamma.pdf(intensity, a=shape, scale=1. / rate), 0., np.inf,
                           epsabs=1e-11, epsrel=1e-10)
    assert error < 1e-8 * (1. + expected)
    assert jfm.gamma_intensity_log_integral(n, t, shape=shape, rate=rate) == pytest.approx(np.log(expected), abs=2e-10)


@pytest.mark.parametrize("finishers", [0, 1, 4])
def test_anonymous_negative_binomial_normalization_and_moments(finishers):
    a, b, q = 4., 1.2, .65
    shape, success = jfm.anonymous_count_parameters(finishers, q, shape=a, rate=b)
    assert shape == a + finishers
    assert success == pytest.approx((b + q) / (b + 1.))
    k = np.arange(100)
    # Integrate Poisson thinnings against the Gamma prior, then divide by the
    # independently integrated visible-finisher probability.
    denominator = quad(lambda lam: poisson.pmf(finishers, lam * q) * gamma.pdf(lam, a=a, scale=1. / b),
                       0., np.inf, epsabs=1e-11)[0]
    enumerated = np.array([
        quad(lambda lam: poisson.pmf(finishers, lam * q) * poisson.pmf(int(hidden), lam * (1. - q))
             * gamma.pdf(lam, a=a, scale=1. / b), 0., np.inf, epsabs=1e-12)[0] / denominator
        for hidden in k[:25]
    ])
    probabilities = nbinom.pmf(k, shape, success)
    np.testing.assert_allclose(probabilities[:25], enumerated, atol=2e-10, rtol=2e-5)
    assert probabilities.sum() == pytest.approx(1., abs=1e-12)
    mean = (probabilities * k).sum()
    assert mean == pytest.approx((a + finishers) * (1. - q) / (b + q))
    assert (probabilities * (k - mean)**2).sum() == pytest.approx((a + finishers) * (1. - q) * (b + 1.) / (b + q)**2)


def test_count_parameter_helpers_broadcast():
    f, q = np.array([0, 2, 5]), np.array([.2, .5, .8])
    shape, success = jfm.anonymous_count_parameters(f, q, shape=4., rate=np.array([1., 2., 3.]))
    np.testing.assert_allclose(shape, 4. + f)
    np.testing.assert_allclose(success, (np.array([1., 2., 3.]) + q) / np.array([2., 3., 4.]))
    result = jfm.gamma_intensity_log_integral(f, q, shape=4., rate=2.)
    np.testing.assert_allclose(result, gammaln(4. + f) - gammaln(4.) + 4. * np.log(2.) - (4. + f) * np.log(2. + q))


def test_anonymous_likelihood_preserves_factorials_and_empty_editions(data):
    p = _parameters(data, homogeneous=True)
    result = jfm.field_loglik_components_numpy(p, data, quadrature_order=15)
    for edition in data.editions.itertuples(index=False):
        c = _ids(data)["course"].index(edition.course_id)
        rows = data.anonymous[data.anonymous.did == edition.did]
        finish = rows[rows.state == "finish"]
        dnf = rows[rows.state == "dnf"]
        f, k = len(finish), len(dnf)
        a, b, q = 4., 4. / 3., .7
        # Independent unordered marked-Poisson integral, not the production
        # Gamma helper or an algebraically duplicated log-gamma expression.
        n, t = (f + k, 1.) if edition.reported else (f, q)
        integral = quad(lambda lam: lam**n * np.exp(-t * lam) * gamma.pdf(lam, a=a, scale=1. / b),
                        0., np.inf, epsabs=1e-11)[0]
        expected = np.log(integral) - np.log(factorial(f))
        if edition.reported:
            expected -= np.log(factorial(k))
        expected += sum(np.log(p["theta_course"][c, int(row.group)] * q)
                        + norm.logpdf(row.log_pace, 2., .3) for row in finish.itertuples())
        expected += sum(np.log(p["theta_course"][c, int(row.group)] * (1. - q)) for row in dnf.itertuples())
        assert result.anonymous.sel(did=edition.did).item() == pytest.approx(expected, abs=1e-10)
    # No observed rows is evidence about intensity, not an omitted edition.
    assert result.anonymous.sel(did="zero").item() < 0.
    np.testing.assert_allclose(result.Q, .7)
    np.testing.assert_allclose(np.exp(result.log_Q), result.Q)
    np.testing.assert_allclose(np.exp(result.log_one_minus_Q), .3)
    xr.testing.assert_allclose(result.total, result.named + result.anonymous)
    xr.testing.assert_allclose(jfm.field_loglik_numpy(p, data), result.total)


def test_homogeneous_improper_limit_and_exact_one_over_n_identity():
    # This limit is only proper when at least one finisher was observed.
    f, q = 3, .62
    shape, success = jfm.anonymous_count_parameters(f, q, shape=1e-9, rate=1e-9)
    k = np.arange(100)
    np.testing.assert_allclose(nbinom.pmf(k, shape, success), nbinom.pmf(k, f, q), atol=5e-10)
    # N=F+K, prior P(N) proportional to 1/N, likelihood Binomial(F | N,q).
    # The exact normalizer is 1/F, independent of q.
    n = f + k
    log_weight = (gammaln(n + 1) - gammaln(f + 1) - gammaln(k + 1)
                  + f * np.log(q) + k * np.log1p(-q) - np.log(n))
    assert np.exp(log_weight).sum() == pytest.approx(1. / f, abs=1e-13)
    np.testing.assert_allclose(np.exp(log_weight) * f, nbinom.pmf(k, f, q), atol=1e-14)


# Independent continuous mark integration, including the Gaussian identity.


def test_anonymous_marks_against_adaptive_two_trait_integrals(data):
    p = _parameters(data)
    marks = jfm.anonymous_mark_logprob_numpy(p, data, quadrature_order=35)
    for group in (0, 4):
        did = "report"
        c = _ids(data)["course"].index("c-a")
        outcome = _outcome_function(p, data, did, group)
        expected_p = _normal_expectation(lambda z1, z2: outcome(z1, z2)[1])
        assert marks.log_p_group.sel(did=did).values[group] == pytest.approx(np.log(expected_p), abs=2e-7)
        assert marks.log_q_group.sel(did=did).values[group] == pytest.approx(np.log1p(-expected_p), abs=2e-7)
        assert marks.log_H_D.sel(did=did).values[group] == pytest.approx(np.log(p["theta_course"][c, group] * expected_p), abs=2e-7)
    # Directly integrate q * Normal(y | mu(z1), sigma) on the prior z-grid.
    # Nonzero quadratic DNF and pace/DNF correlation make factorization wrong.
    for observation_id in ("a0", "a5"):
        row = data.anonymous.set_index("observation_id").loc[observation_id]
        c = _ids(data)["course"].index(data.editions.set_index("did").loc[row.did, "course_id"])
        outcome = _outcome_function(p, data, row.did, int(row.group))
        def integrand(z1, z2):
            mu, dnf, sigma = outcome(z1, z2)
            return (1. - dnf) * norm.pdf(row.log_pace, mu, sigma)
        expected = p["theta_course"][c, int(row.group)] * _normal_expectation(integrand)
        assert marks.log_H_F.sel(anonymous_finish=observation_id).item() == pytest.approx(np.log(expected), abs=2e-7)


def test_finisher_mark_density_integrates_to_Q_and_dnf_complement(data):
    p = _parameters(data)
    rows = []
    grids = []
    for group in range(6):
        base, _, sigma = _outcome_function(p, data, "report", group)(0., 0.)
        sd = np.sqrt(p["omega"]**2 + sigma**2)
        grid = np.linspace(base - 9. * sd, base + 9. * sd, 121)
        grids.append(grid)
        rows.extend((f"integration-{group}-{i}", "report", group, "finish", y)
                    for i, y in enumerate(grid))
    integration_data = replace(data, anonymous=pd.DataFrame(rows, columns=data.anonymous.columns))
    marks = jfm.anonymous_mark_logprob_numpy(p, integration_data, quadrature_order=35)
    masses = []
    for group, grid in enumerate(grids):
        labels = [f"integration-{group}-{i}" for i in range(len(grid))]
        densities = np.exp(marks.log_H_F.sel(anonymous_finish=labels).values)
        masses.append(simpson(densities, x=grid))
    diagnostic = jfm.field_loglik_components_numpy(p, integration_data, quadrature_order=35)
    assert sum(masses) == pytest.approx(diagnostic.Q.sel(did="report").item(), abs=2e-7)
    dnf_mass = np.exp(marks.log_H_D.sel(did="report")).sum().item()
    assert sum(masses) + dnf_mass == pytest.approx(1., abs=2e-7)


@pytest.mark.parametrize("order", [3, 15, 25, 35])
def test_standard_normal_quadrature_normalization_and_moments(order):
    nodes, weights = jfm.normal_quadrature(order)
    assert np.asarray(nodes).shape == np.asarray(weights).shape == (order,)
    assert np.all(weights > 0.)
    assert np.sum(weights) == pytest.approx(1., abs=1e-14)
    for power, expected in [(1, 0.), (2, 1.), (3, 0.), (4, 3.)]:
        assert np.dot(weights, nodes**power) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("order", [0, 1, 2, 3.5, True, "15"])
def test_quadrature_rejects_invalid_order(order):
    with pytest.raises((TypeError, ValueError)):
        jfm.normal_quadrature(order)


def test_adjacent_quadrature_orders_at_fixed_moderate_parameters(data):
    p = _parameters(data)
    results = [jfm.field_loglik_numpy(p, data, quadrature_order=order) for order in (15, 25, 35)]
    for lower, upper in zip(results, results[1:]):
        np.testing.assert_allclose(lower, upper, atol=1e-3, rtol=0.)


# Visible schema, external exposure, masking and truth isolation.


def _assert_same_data(left, right):
    for field in fields(left):
        pd.testing.assert_frame_equal(getattr(left, field.name), getattr(right, field.name))


def test_data_contract_demographic_support_and_unused_registry(data):
    assert [field.name for field in fields(data)] == [
        "editions", "registry", "candidates", "anonymous", "demographics",
    ]
    jfm.validate_field_data(data)
    np.testing.assert_array_equal(data.demographics.sex, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(data.demographics.age, [20, 40, 60, 20, 40, 60])
    np.testing.assert_array_equal(data.demographics.h, [-.5, -.5, -.5, .5, .5, .5])
    np.testing.assert_array_equal(data.demographics.A, [-2., 0., 2., -2., 0., 2.])
    assert "unused" not in set(data.candidates.runner_id)
    # Registries/candidates are external exposure, not reconstructed from starts.
    tiny = replace(data, candidates=data.candidates.iloc[:0].copy(), anonymous=data.anonymous.iloc[:0].copy())
    jfm.validate_field_data(tiny)
    model = jfm.build_joint_field_model(tiny, quadrature_order=3)
    assert set(model.coords["runner"]) == set(data.registry.runner_id)
    assert set(model.coords["edition"]) == set(data.editions.did)


def test_mask_is_copy_preserves_exposure_visible_rows_and_zero_finishers(complete):
    original = _copy(complete)
    masked = jfm.mask_field_editions(complete, ["target-a", "target-b", "zero"])
    _assert_same_data(original, complete)
    pd.testing.assert_frame_equal(masked.registry, complete.registry)
    assert not masked.editions.set_index("did").loc[["target-a", "target-b", "zero"], "reported"].any()
    assert masked.editions.set_index("did").loc["report", "reported"]
    pd.testing.assert_frame_equal(masked.candidates[["did", "runner_id"]], complete.candidates[["did", "runner_id"]])
    for table_name in ("candidates", "anonymous"):
        before, after = getattr(complete, table_name), getattr(masked, table_name)
        pd.testing.assert_frame_equal(before[before.state == "finish"].reset_index(drop=True),
                                      after[after.state == "finish"].reset_index(drop=True))
    assert set(masked.candidates.loc[masked.candidates.did == "target-b", "state"]) == {"unseen"}
    assert not masked.anonymous.did.isin(["target-b", "zero"]).any()
    assert {"target-b", "zero"} <= set(masked.editions.did)
    jfm.validate_field_data(masked)
    # Returned frames must not alias their evaluator inputs.
    masked.registry.loc[0, "group"] = 5
    masked.candidates.loc[0, "log_pace"] = 123.
    _assert_same_data(original, complete)


def test_hidden_outcomes_cannot_change_masked_inputs_or_fingerprint(complete):
    altered = _copy(complete)
    hidden = altered.candidates.did.isin(["target-a", "target-b"]) & altered.candidates.state.ne("finish")
    altered.candidates.loc[hidden, "state"] = ["nonstarter", "dnf", "dnf", "dnf"]
    extra = altered.anonymous[altered.anonymous.did.isin(["target-a", "target-b"]) & altered.anonymous.state.eq("dnf")].copy()
    extra.observation_id = "extra-hidden-" + extra.observation_id
    extra.group = 4
    altered.anonymous = pd.concat([altered.anonymous, extra], ignore_index=True)
    altered.anonymous.loc[altered.anonymous.state.eq("dnf") & altered.anonymous.did.ne("report"), "group"] = 0
    dids = ["target-a", "target-b", "zero"]
    left = jfm.mask_field_editions(complete, dids)
    right = jfm.mask_field_editions(altered, dids)
    _assert_same_data(left, right)
    assert _fingerprint(left) == _fingerprint(right)
    p = _parameters(left)
    xr.testing.assert_identical(jfm.field_loglik_numpy(p, left), jfm.field_loglik_numpy(p, right))


@pytest.mark.parametrize("table,column,value", [
    ("editions", "distance_km", 0.), ("editions", "distance_km", -1.),
    ("editions", "distance_km", np.nan), ("editions", "distance_km", np.inf),
    ("editions", "reported", "True"), ("editions", "did", None),
    ("editions", "course_id", ""), ("registry", "runner_id", None),
    ("registry", "group", -1), ("registry", "group", 6),
    ("registry", "group", 1.5), ("registry", "group", True),
    ("candidates", "did", "unknown"), ("candidates", "runner_id", "unknown"),
    ("candidates", "state", "dns"), ("candidates", "state", None),
    ("candidates", "log_pace", np.inf), ("candidates", "log_pace", np.nan),
    ("anonymous", "did", "unknown"), ("anonymous", "observation_id", "r0"),
    ("anonymous", "group", 6), ("anonymous", "state", "unseen"),
    ("anonymous", "log_pace", np.inf), ("anonymous", "log_pace", np.nan),
])
def test_invalid_visible_values_are_rejected(data, table, column, value):
    bad = _copy(data)
    # Object dtype permits deliberately invalid bool/string assignments on all
    # supported pandas versions without relying on coercion warnings.
    getattr(bad, table)[column] = getattr(bad, table)[column].astype(object)
    getattr(bad, table).loc[0, column] = value
    with pytest.raises((ValueError, TypeError)):
        jfm.validate_field_data(bad)


@pytest.mark.parametrize("table", ["editions", "registry", "candidates", "anonymous"])
def test_duplicate_identifiers_and_edges_are_rejected(data, table):
    frame = getattr(data, table)
    with pytest.raises(ValueError):
        jfm.validate_field_data(replace(data, **{table: pd.concat([frame, frame.iloc[:1]], ignore_index=True)}))


@pytest.mark.parametrize("table", ["editions", "registry", "candidates", "anonymous", "demographics"])
def test_truth_and_other_extra_columns_are_rejected(data, table):
    extra = getattr(data, table).assign(total_dnf=999999)
    with pytest.raises(ValueError):
        jfm.validate_field_data(replace(data, **{table: extra}))
    with pytest.raises(ValueError):
        jfm.build_joint_field_model(replace(data, **{table: extra}), quadrature_order=3)


@pytest.mark.parametrize("table,state", [("candidates", "dnf"), ("candidates", "nonstarter"), ("anonymous", "dnf")])
def test_suppressed_states_cannot_leak_hidden_outcomes(data, table, state):
    bad = _copy(data)
    frame = getattr(bad, table)
    index = frame[frame.did.eq("target-a")].index[0]
    frame.loc[index, ["state", "log_pace"]] = [state, np.nan]
    with pytest.raises(ValueError):
        jfm.validate_field_data(bad)


def test_nonfinish_paces_no_reporting_and_changed_demographics_rejected(data):
    bad = _copy(data)
    bad.candidates.loc[bad.candidates.state.eq("unseen"), "log_pace"] = 2.
    with pytest.raises(ValueError):
        jfm.validate_field_data(bad)
    with pytest.raises(ValueError):
        jfm.validate_field_data(replace(data, editions=data.editions.assign(reported=False)))
    demographic = data.demographics.copy()
    demographic.loc[0, "A"] = -.2
    with pytest.raises(ValueError):
        jfm.validate_field_data(replace(data, demographics=demographic))
    with pytest.raises(ValueError):
        jfm.mask_field_editions(data, ["not-an-edition"])


def test_canonical_string_ids_row_order_and_fingerprint_settings(data):
    shuffled = replace(data, **{
        name: getattr(data, name).sample(frac=1., random_state=13).reset_index(drop=True)
        for name in ("editions", "registry", "candidates", "anonymous")
    })
    assert _fingerprint(data) == _fingerprint(shuffled)
    p = _parameters(data)
    xr.testing.assert_identical(jfm.field_loglik_numpy(p, data), jfm.field_loglik_numpy(p, shuffled))
    for settings in ({"seed": 104}, {"version": "v2"}, {"sampler": "other"}, {"quadrature_order": 25},
                     {"priors": replace(jfm.JointFieldPrior(), anonymous_mean_sd=.5)}):
        assert _fingerprint(data) != _fingerprint(data, **settings)
    changed = _copy(data)
    changed.candidates.loc[0, "log_pace"] += .01
    assert _fingerprint(data) != _fingerprint(changed)
    # Lexical rather than numeric sorting; integers and strings canonicalize.
    numeric = jfm.JointFieldData(
        editions=pd.DataFrame([(2, 2, 100., True), (10, 10, 50., False)], columns=data.editions.columns),
        registry=pd.DataFrame([(2, 0), (10, 1)], columns=data.registry.columns),
        candidates=pd.DataFrame([(2, 2, "finish", 2.), (10, 10, "unseen", np.nan)], columns=data.candidates.columns),
        anonymous=data.anonymous.iloc[:0].copy(),
    )
    strings = replace(numeric,
                      editions=numeric.editions.assign(did=["2", "10"], course_id=["2", "10"]),
                      registry=numeric.registry.assign(runner_id=["2", "10"]),
                      candidates=numeric.candidates.assign(did=["2", "10"], runner_id=["2", "10"]))
    assert _fingerprint(numeric) == _fingerprint(strings)
    model = jfm.build_joint_field_model(numeric, quadrature_order=3)
    for dim in ("course", "edition", "runner"):
        assert tuple(model.coords[dim]) == ("10", "2")
    assert jfm.field_loglik_numpy(_parameters(numeric), numeric).did.values.tolist() == ["10", "2"]


def test_prior_is_frozen_serializable_and_sensitivity_is_explicit():
    prior = jfm.JointFieldPrior()
    assert prior.field_shape == 4.
    assert prior.demographic_concentration == 20.
    assert prior.anonymous_mean_sd == 1.
    assert json.loads(json.dumps(asdict(prior))) == asdict(prior)
    assert replace(prior, anonymous_mean_sd=.5) != prior
    with pytest.raises(FrozenInstanceError):
        prior.anonymous_mean_sd = .5


# The symbolic graph must evaluate the same observation likelihood, without priors.


@pytest.fixture(scope="module")
def compiled_model():
    data = jfm.mask_field_editions(_complete_data(), ["target-a", "target-b", "zero"])
    model = jfm.build_joint_field_model(data, quadrature_order=15)
    return data, model, jfm.compile_field_loglik(model)


def test_continuous_graph_components_at_initial_and_fixed_perturbations(compiled_model):
    data, model, evaluate = compiled_model
    assert dict(jfm.PARAMETER_DIMS) == PARAMETER_DIMS
    assert len(model.potentials) == 2
    assert all(np.dtype(variable.dtype).kind == "f" for variable in model.free_RVs)
    points = jfm.field_reference_points(model)
    assert set(points) == {"initial", "perturbed_plus", "perturbed_minus"}
    for label, point in points.items():
        assert set(point) == {variable.name for variable in model.value_vars}
        parameters = jfm.field_parameters(model, point)
        assert set(parameters) == set(PARAMETER_DIMS)
        assert all(np.isfinite(value).all() for value in parameters.values())
        expected = jfm.field_loglik_components_numpy(parameters, data, quadrature_order=15)
        actual = evaluate(point)
        assert set(actual) == {"named", "anonymous", "total", "gradient"}
        for key in ("named", "anonymous", "total"):
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-10, atol=2e-10, err_msg=label)
        np.testing.assert_allclose(actual["total"], actual["named"] + actual["anonymous"])
        assert np.isfinite(actual["gradient"]).all()
        # Named terms have an additional direct, non-quadrature oracle.
        for did in _ids(data)["edition"]:
            expected_named = 0.
            for row in data.candidates[data.candidates.did == did].itertuples():
                c = _ids(data)["course"].index(data.editions.set_index("did").loc[did, "course_id"])
                r = _ids(data)["runner"].index(row.runner_id)
                group = int(data.registry.set_index("runner_id").loc[row.runner_id, "group"])
                z1, z2 = parameters["runner_z1"][r], parameters["runner_z2"][r]
                mu, p, sigma = _outcome_function(parameters, data, did, group)(z1, z2)
                pi = expit(parameters["entry_mean"] + parameters["entry_course"][c] + parameters["entry_pace"] * z1)
                if row.state == "finish":
                    expected_named += np.log(pi * (1. - p)) + norm.logpdf(row.log_pace, mu, sigma)
                elif row.state == "dnf":
                    expected_named += np.log(pi * p)
                elif row.state == "nonstarter":
                    expected_named += np.log1p(-pi)
                else:
                    expected_named += np.log(1. - pi + pi * p)
            assert actual["named"][_ids(data)["edition"].index(did)] == pytest.approx(expected_named, abs=2e-10)


def test_likelihood_gradient_in_transformed_value_order_by_central_differences(compiled_model):
    data, model, evaluate = compiled_model
    point = {name: np.array(value, copy=True) for name, value in
             jfm.field_reference_points(model)["perturbed_plus"].items()}
    actual = evaluate(point)["gradient"]
    finite_difference = []
    step = 2e-5
    for variable in model.value_vars:
        name = variable.name
        for index in np.ndindex(point[name].shape):
            plus, minus = deepcopy(point), deepcopy(point)
            plus[name][index] += step
            minus[name][index] -= step
            high = jfm.field_loglik_numpy(jfm.field_parameters(model, plus), data, quadrature_order=15).sum().item()
            low = jfm.field_loglik_numpy(jfm.field_parameters(model, minus), data, quadrature_order=15).sum().item()
            finite_difference.append((high - low) / (2. * step))
    np.testing.assert_allclose(actual, finite_difference, atol=3e-5, rtol=3e-5)


def test_transformed_likelihood_gradient_excludes_priors_with_independent_scalar_oracle(compiled_model):
    data, model, evaluate = compiled_model
    point = deepcopy(model.initial_point())
    # Remove risk heterogeneity at this test point, not from the model/prior.
    # A near-zero positive scale keeps the point finite in sampler coordinates.
    point["omega_dnf_log__"] = np.array(-30.)
    point["quadratic"] = np.array(0.)
    point["dnf_distance"] = np.array(0.)
    point["dnf_gender_gap"] = np.array(0.)
    point["dnf_age_linear"] = np.array(0.)
    point["dnf_age_quadratic"] = np.array(0.)
    point["course_z1"] = np.zeros_like(point["course_z1"])
    point["course_z2"] = np.zeros_like(point["course_z2"])
    point["edition_z1"] = np.zeros_like(point["edition_z1"])
    point["edition_z2"] = np.zeros_like(point["edition_z2"])
    physical = jfm.field_parameters(model, point)
    dnf_eta = physical["course_dnf_level"][0]
    def oracle(eta):
        q, p = expit(-eta), expit(eta)
        total = 0.
        for row in data.candidates.itertuples():
            c = _ids(data)["course"].index(data.editions.set_index("did").loc[row.did, "course_id"])
            r = _ids(data)["runner"].index(row.runner_id)
            pi = expit(physical["entry_mean"] + physical["entry_course"][c]
                       + physical["entry_pace"] * physical["runner_z1"][r])
            # Pace densities do not depend on the differentiated DNF intercept.
            total += np.log({"finish": pi * q, "dnf": pi * p,
                             "nonstarter": 1. - pi, "unseen": 1. - pi + pi * p}[row.state])
        for edition in data.editions.itertuples():
            rows = data.anonymous[data.anonymous.did == edition.did]
            f, k = rows.state.eq("finish").sum(), rows.state.eq("dnf").sum()
            c = _ids(data)["course"].index(edition.course_id)
            a = physical["field_shape"]
            mean = np.exp(physical["anonymous_mean"] + physical["anonymous_distance"] * np.log(edition.distance_km / 50.)
                          + physical["anonymous_course"][c])
            b = a / mean
            n, t = (f + k, 1.) if edition.reported else (f, q)
            total += gammaln(a + n) - gammaln(a) + a * np.log(b) - (a + n) * np.log(b + t)
            total += f * np.log(q) + k * np.log(p)
        return total
    step = 1e-5
    expected = (oracle(dnf_eta + step) - oracle(dnf_eta - step)) / (2. * step)
    offset = 0
    for variable in model.value_vars:
        if variable.name == "dnf_mean":
            break
        offset += point[variable.name].size
    else:
        pytest.fail("DNF intercept missing from continuous model")
    assert evaluate(point)["gradient"][offset] == pytest.approx(expected, abs=2e-7)


def test_prefit_adjacent_order_audit_records_likelihood_and_gradient_gate(data):
    audit = jfm.audit_field_quadrature(data, lower_order=15, upper_order=25)
    assert isinstance(audit, xr.Dataset)
    assert audit.attrs["passed"]
    assert set(audit.point.values) == {"initial", "perturbed_plus", "perturbed_minus"}
    assert audit.absolute_loglik_difference.max().item() < 1e-3
    assert audit.max_scaled_gradient_difference.max().item() < 1e-3
    assert np.isfinite(audit.to_array()).all()


@pytest.mark.parametrize("passes,expected,orders", [
    ([True], 15, [(15, 25)]), ([False, True], 25, [(15, 25), (25, 35)]),
    ([False, False], None, [(15, 25), (25, 35)]),
])
def test_quadrature_selection_escalates_once_then_stops(data, monkeypatch, passes, expected, orders):
    calls = []
    def audit(*args, lower_order, upper_order, **kwargs):
        calls.append((lower_order, upper_order))
        return xr.Dataset(attrs={"passed": passes[len(calls) - 1]})
    monkeypatch.setattr(jfm, "audit_field_quadrature", audit)
    if expected is None:
        with pytest.raises(ValueError):
            jfm.select_field_quadrature(data)
    else:
        chosen, evidence = jfm.select_field_quadrature(data)
        assert chosen == expected
        assert len(evidence) == len(orders)
    assert calls == orders


def test_postfit_audit_uses_evenly_spaced_shared_draws_without_refitting(data):
    posterior = _posterior(data, draws=43)
    # Distinct draw parameters ensure that an audit of only the first draw fails.
    posterior["quadratic"][:] = np.linspace(.1, .3, 43)[None, :]
    audit = jfm.audit_posterior_quadrature(posterior, data, quadrature_order=15)
    assert isinstance(audit, xr.Dataset)
    assert audit.attrs["passed"]
    assert dict(audit.sizes) == {"chain": 2, "draw": 20, "did": len(data.editions)}
    np.testing.assert_array_equal(audit.chain, posterior.chain)
    selected = np.linspace(0, 42, 20, dtype=int)
    np.testing.assert_array_equal(audit.draw, posterior.draw.values[selected])
    assert np.isfinite(audit.to_array()).all()
    assert audit.to_array().max().item() < 1e-3


# Posterior prediction: draw/identity alignment, conditional moments and streams.


def test_prediction_shape_labels_exact_sum_and_no_mutation(data):
    posterior = _posterior(data)
    original = posterior.copy(deep=True)
    visible = _copy(data)
    result = _predict(posterior, data)
    assert set(result.data_vars) == {"named_count", "anonymous_count", "total_count", "expected_total"}
    assert result.did.values.tolist() == ["target-a", "target-b", "zero"]
    np.testing.assert_array_equal(result.chain, posterior.chain)
    np.testing.assert_array_equal(result.draw, posterior.draw)
    for name in result.data_vars:
        assert result[name].dims == ("chain", "draw", "did")
        assert np.isfinite(result[name]).all()
        assert (result[name] >= 0).all()
    for name in ("named_count", "anonymous_count", "total_count"):
        assert np.issubdtype(result[name].dtype, np.integer)
    xr.testing.assert_equal(result.total_count, result.named_count + result.anonymous_count)
    assert (result.named_count.sel(did="zero") == 0).all()
    assert (result.expected_total.sel(did="zero") > 0).all()  # proper F=0
    assert result.attrs["endpoint_clips"] == 0
    xr.testing.assert_identical(original, posterior)
    _assert_same_data(visible, data)


def test_prediction_aligns_reordered_coordinates_and_rows_per_target(data):
    posterior = _posterior(data)
    expected = _predict(posterior, data)
    reverse = posterior.isel({dim: slice(None, None, -1) for dim in ("course", "edition", "runner", "sex", "group")})
    shuffled = replace(data, **{
        name: getattr(data, name).sample(frac=1., random_state=19).reset_index(drop=True)
        for name in ("editions", "registry", "candidates", "anonymous")
    })
    xr.testing.assert_identical(expected, _predict(reverse, shuffled))
    xr.testing.assert_identical(expected, _predict(posterior, data))
    changed_seed = _predict(posterior, data, seed=105)
    xr.testing.assert_equal(expected.expected_total, changed_seed.expected_total)
    assert not np.array_equal(expected.total_count, changed_seed.total_count)
    # Removing a prediction target from the output must not advance another
    # target's RNG. Turn the empty target into reporting without hidden labels.
    fewer = replace(data, editions=data.editions.assign(reported=data.editions.reported | data.editions.did.eq("zero")))
    subset = _predict(posterior, fewer)
    for name in expected.data_vars:
        xr.testing.assert_equal(expected[name].sel(did=subset.did), subset[name])
    # Chain order is also labels, never an RNG stream ordinal.
    reordered = _predict(posterior.isel(chain=slice(None, None, -1)), data)
    for name in expected.data_vars:
        xr.testing.assert_equal(expected[name], reordered[name].sel(chain=expected.chain))


def test_prediction_selects_common_evenly_spaced_posterior_draws(data):
    posterior = _posterior(data, draws=19)
    result = _predict(posterior, data, draws=5)
    labels = posterior.draw.values[np.linspace(0, 18, 5, dtype=int)]
    np.testing.assert_array_equal(result.draw, labels)
    direct = _predict(posterior.sel(draw=labels), data, draws=5)
    xr.testing.assert_identical(result, direct)


def test_prediction_conditional_expectations_and_zero_finisher_count_moments(data):
    # One suppressed zero-finisher edition, two supplied unseen candidates.
    tiny = replace(data, editions=data.editions[data.editions.did.isin(["report", "target-b"])].copy(),
                   candidates=data.candidates[data.candidates.did.isin(["report", "target-b"])].copy(),
                   anonymous=data.anonymous[data.anonymous.did.eq("report")].copy())
    parameters = _parameters(tiny, homogeneous=True)
    posterior = _posterior(tiny, parameters, draws=512)
    prediction = _predict(posterior, tiny, draws=512)
    pi, p, q, a, b = .4, .3, .7, 4., 4. / 3.
    hidden = pi * p / (1. - pi + pi * p)
    named_mean, named_var = 2. * hidden, 2. * hidden * (1. - hidden)
    anonymous_mean = a * p / (b + q)
    anonymous_var = a * p * (b + 1.) / (b + q)**2
    np.testing.assert_allclose(prediction.expected_total, named_mean + anonymous_mean, atol=1e-12)
    n = prediction.sizes["chain"] * prediction.sizes["draw"]
    # Fixed seed, broad Monte Carlo error bounds; exact moments above are the
    # primary oracle, not a noisy coverage claim from this synthetic experiment.
    assert prediction.named_count.mean().item() == pytest.approx(named_mean, abs=7. * np.sqrt(named_var / n))
    assert prediction.anonymous_count.mean().item() == pytest.approx(anonymous_mean, abs=7. * np.sqrt(anonymous_var / n))
    assert prediction.total_count.mean().item() == pytest.approx(named_mean + anonymous_mean, abs=7. * np.sqrt((named_var + anonymous_var) / n))
    assert prediction.named_count.max() <= 2
    assert prediction.named_count.var().item() == pytest.approx(named_var, abs=.08)
    assert prediction.anonymous_count.var().item() == pytest.approx(anonymous_var, abs=.3)


def test_prediction_uses_fitted_target_edition_effect_not_fresh_random_effect(data):
    parameters = _parameters(data, homogeneous=True)
    posterior = _posterior(data, parameters)
    posterior["edition_dnf"].loc[{"edition": "target-a"}] = -2.
    posterior["edition_dnf"].loc[{"edition": "target-b"}] = 2.
    result = _predict(posterior, data)
    for did, effect in [("target-a", -2.), ("target-b", 2.), ("zero", 0.)]:
        p = expit(logit(.3) + effect)
        q = 1. - p
        f = sum(data.anonymous.did.eq(did) & data.anonymous.state.eq("finish"))
        unseen = sum(data.candidates.did.eq(did) & data.candidates.state.eq("unseen"))
        expected = unseen * .4 * p / (.6 + .4 * p) + (4. + f) * p / (4. / 3. + q)
        np.testing.assert_allclose(result.expected_total.sel(did=did), expected, atol=1e-12)
    assert result.expected_total.sel(did="target-b").min() > result.expected_total.sel(did="target-a").max()


@pytest.mark.parametrize("dimension", ["course", "edition", "runner", "sex", "group", "chain", "draw"])
def test_prediction_rejects_wrong_or_duplicate_coordinates(data, dimension):
    posterior = _posterior(data)
    bad = posterior.assign_coords({dimension: ["wrong"] * posterior.sizes[dimension]})
    with pytest.raises(ValueError):
        _predict(bad, data)
    if dimension in ("course", "edition", "runner", "sex", "group"):
        bad = posterior.isel({dimension: slice(1, None)})
        with pytest.raises(ValueError):
            _predict(bad, data)


@pytest.mark.parametrize("missing", ["field_shape", "theta_course", "edition_dnf", "runner_z1"])
def test_prediction_requires_complete_physical_parameters(data, missing):
    with pytest.raises(ValueError):
        _predict(_posterior(data).drop_vars(missing), data)


@pytest.mark.parametrize("parameter,value", [
    ("sigma", 0.), ("omega", -1.), ("rho_runner", 1.01),
    ("theta_course", -.1), ("field_shape", 0.), ("course_dnf_level", np.nan),
])
def test_prediction_rejects_invalid_physical_parameters(data, parameter, value):
    posterior = _posterior(data)
    posterior[parameter][:] = value
    with pytest.raises(ValueError):
        _predict(posterior, data)


@pytest.mark.parametrize("draws", [0, -1, 13, 1.5, True])
def test_prediction_rejects_insufficient_or_invalid_draw_requests(data, draws):
    with pytest.raises((ValueError, TypeError)):
        _predict(_posterior(data), data, draws=draws)


def test_prediction_rejects_non_dataset_and_truth_arguments(data):
    posterior = _posterior(data)
    with pytest.raises(TypeError):
        _predict({name: value.values for name, value in posterior.data_vars.items()}, data)
    with pytest.raises(TypeError):
        jfm.predict_missing_field_counts(posterior, data, seed=104, draws_per_chain=12, truth={"total_dnf": 999})


def test_no_observations_still_contribute_reporting_and_suppressed_intensity(data):
    tiny = replace(data, candidates=data.candidates.iloc[:0].copy(), anonymous=data.anonymous.iloc[:0].copy())
    p = _parameters(tiny, homogeneous=True)
    result = jfm.field_loglik_components_numpy(p, tiny)
    np.testing.assert_array_equal(result.named, 0.)
    for edition in tiny.editions.itertuples():
        t = 1. if edition.reported else .7
        assert result.anonymous.sel(did=edition.did).item() == pytest.approx(4. * np.log((4. / 3.) / (4. / 3. + t)))


def test_gaussian_conditional_identity_remains_accurate_for_narrow_pace_density(data):
    p = _parameters(data)
    p.update(omega=.8, sigma=np.array([.03, .025]))
    row = data.anonymous.set_index("observation_id").loc["a5"]
    outcome = _outcome_function(p, data, row.did, int(row.group))
    def integrand(z1, z2):
        mu, risk, sigma = outcome(z1, z2)
        return (1. - risk) * norm.pdf(row.log_pace, mu, sigma)
    c = _ids(data)["course"].index("c-a")
    expected = np.log(p["theta_course"][c, int(row.group)] * _normal_expectation(integrand))
    # A prior-normal GH grid would miss this narrow pace peak. The conditional
    # Gaussian identity integrates the smooth risk instead of that peak.
    marks = jfm.anonymous_mark_logprob_numpy(p, data, quadrature_order=15)
    assert marks.log_H_F.sel(anonymous_finish="a5").item() == pytest.approx(expected, abs=2e-6)


def test_prediction_reports_machine_precision_endpoint_clips(data):
    p = _parameters(data, homogeneous=True)
    p["course_dnf_level"][:] = -1000.
    result = _predict(_posterior(data, p), data)
    assert result.attrs["endpoint_clips"] > 0
    np.testing.assert_allclose(result.expected_total, 0., atol=1e-300)
    np.testing.assert_array_equal(result.total_count, 0)


def test_homogeneous_predictions_preserve_streams_across_quadrature_orders(data):
    posterior = _posterior(data, _parameters(data, homogeneous=True))
    lower = jfm.predict_missing_field_counts(posterior, data, seed=104, draws_per_chain=12, quadrature_order=15)
    upper = _predict(posterior, data)
    for name in ("named_count", "anonymous_count", "total_count"):
        xr.testing.assert_equal(lower[name], upper[name])
    xr.testing.assert_allclose(lower.expected_total, upper.expected_total)


def test_numeric_and_string_id_collision_is_rejected(data):
    registry = data.registry.copy()
    registry.loc[0, "runner_id"], registry.loc[1, "runner_id"] = 2, "2"
    with pytest.raises(ValueError):
        jfm.validate_field_data(replace(data, registry=registry))


# Independent forward simulations: fixed design, separate truths and seed streams.


def _stage_seed(scenario, stage):
    return sum(map(ord, f"2.5_dnf_field:{scenario}:{stage}:v1"))


def _simulate(scenario):
    shared = "shared" if scenario in {
        "benign", "race_variation", "participation_selection", "population_misspecification",
    } else scenario
    return jfm.simulate_joint_field(
        scenario=scenario, seed=_stage_seed(scenario, "outcomes"),
        design_seed=_stage_seed(shared, "design"), nuisance_seed=_stage_seed(shared, "nuisance"),
    )


@pytest.fixture(scope="module")
def simulations():
    return {scenario: _simulate(scenario) for scenario in (
        "prior", "smoke", "benign", "race_variation", "participation_selection", "population_misspecification",
    )}


@pytest.mark.parametrize("scenario", ["prior", "smoke", "benign", "race_variation", "participation_selection", "population_misspecification"])
def test_simulator_dimensions_mask_rule_truth_counts_and_probability_formulas(simulations, scenario):
    simulation = simulations[scenario]
    assert isinstance(simulation, jfm.JointFieldSimulation)
    assert simulation.scenario == scenario
    jfm.validate_field_data(simulation.data)
    complete = simulation.truth["complete_data"]
    assert complete.editions.reported.all()
    courses, per_course, runners, candidates = (2, 2, 12, 8) if scenario == "smoke" else (6, 4, 60, 40)
    assert complete.editions.course_id.nunique() == courses
    assert len(complete.editions) == courses * per_course
    assert len(complete.registry) == runners
    assert complete.candidates.groupby("did").size().eq(candidates).all()
    assert not complete.candidates[["did", "runner_id"]].duplicated().any()
    assert not set(complete.anonymous.observation_id) & set(complete.registry.runner_id)
    assert simulation.data.editions.reported.eq(False).sum() == courses
    for c, (_, editions) in enumerate(complete.editions.groupby("course_id", sort=True)):
        ids = sorted(editions.did.astype(str))
        masked = simulation.data.editions.set_index("did").loc[ids, "reported"]
        assert masked.eq(False).sum() == 1
        assert not masked.loc[ids[c % per_course]]
    if scenario != "smoke":
        assert complete.editions.groupby("course_id").distance_km.first().tolist() == [50., 50., 80., 100., 130., 160.]
    truths = simulation.truth
    assert set(truths) == {"complete_data", "candidates", "anonymous", "editions"}
    assert {"z1", "z2", "entry_probability", "dnf_probability"} <= set(truths["candidates"])
    assert {"z1", "z2", "dnf_probability"} <= set(truths["anonymous"])
    count_truth = truths["editions"].set_index("did")
    for did in complete.editions.did:
        named = complete.candidates[complete.candidates.did.eq(did)]
        anon = complete.anonymous[complete.anonymous.did.eq(did)]
        assert count_truth.loc[did, "named_starters"] == named.state.ne("nonstarter").sum()
        assert count_truth.loc[did, "anonymous_starters"] == len(anon)
        assert count_truth.loc[did, "named_dnf"] == named.state.eq("dnf").sum()
        assert count_truth.loc[did, "anonymous_dnf"] == anon.state.eq("dnf").sum()
        assert count_truth.loc[did, "total_dnf"] == named.state.eq("dnf").sum() + anon.state.eq("dnf").sum()
        assert np.isfinite(count_truth.loc[did, "lambda"]) and count_truth.loc[did, "lambda"] > 0.
    # Independently reconstruct probabilities from physical truths, not the
    # simulator's helper. Check both named and anonymous latent transformations.
    p = simulation.parameters
    for row in truths["candidates"].head(20).itertuples():
        group = int(complete.registry.set_index("runner_id").loc[row.runner_id, "group"])
        c = _ids(complete)["course"].index(complete.editions.set_index("did").loc[row.did, "course_id"])
        assert row.entry_probability == pytest.approx(expit(p["entry_mean"] + p["entry_course"][c] + p["entry_pace"] * row.z1))
        if row.state != "nonstarter":
            risk = _outcome_function(p, complete, row.did, group)(row.z1, row.z2)[1]
            assert row.dnf_probability == pytest.approx(risk)
    for row in truths["anonymous"].head(12).itertuples():
        risk = _outcome_function(p, complete, row.did, int(row.group))(row.z1, row.z2)[1]
        assert row.dnf_probability == pytest.approx(risk)
    assert simulation.seeds["outcomes"] == _stage_seed(scenario, "outcomes")
    assert _fingerprint(simulation.data) == _fingerprint(jfm.mask_field_editions(
        complete, simulation.data.editions.loc[~simulation.data.editions.reported, "did"].tolist()))


def test_simulation_scenarios_share_design_and_nuisance_but_not_outcomes(simulations):
    scenarios = ["benign", "race_variation", "participation_selection", "population_misspecification"]
    reference = simulations["benign"]
    for scenario in scenarios:
        simulation = simulations[scenario]
        pd.testing.assert_frame_equal(simulation.data.registry, reference.data.registry)
        pd.testing.assert_frame_equal(simulation.data.candidates[["did", "runner_id"]], reference.data.candidates[["did", "runner_id"]])
        for name in ("runner_z1", "runner_z2", "course_mean", "course_dnf_level", "entry_course",
                     "anonymous_course", "theta_course", "edition_pace"):
            np.testing.assert_array_equal(simulation.parameters[name], reference.parameters[name])
        p = simulation.parameters
        assert p["omega"] == .3
        assert p["omega_dnf"] == .8
        assert p["quadratic"] == .25
        assert p["anonymous_mean"] == np.log(40.)
        assert p["anonymous_distance"] == 0.
        assert p["entry_mean"] == pytest.approx(logit(1. / 3.))
        assert p["tau_edition_pace"] == .1
        np.testing.assert_array_equal(p["sigma"], [.2, .2])
        assert p["tau_edition_dnf"] == (.3 if scenario == "benign" else .9)
        assert p["rho_edition"] == (0. if scenario == "benign" else .6)
        assert p["entry_pace"] == (0. if scenario in {"benign", "race_variation"} else -.8)
        assert simulation.seeds["design"] == _stage_seed("shared", "design")
        assert simulation.seeds["nuisance"] == _stage_seed("shared", "nuisance")
    assert len({simulations[s].seeds["outcomes"] for s in scenarios}) == 4
    assert simulations["smoke"].parameters["anonymous_mean"] == np.log(8.)
    prior = simulations["prior"].parameters
    assert prior["anonymous_mean"] != np.log(40.)
    assert not np.array_equal(prior["sigma"], [.2, .2])


def test_simulator_is_reproducible_and_does_not_call_likelihood_or_builder(simulations, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Forward simulation must not use the fitted likelihood or quadrature")
    for name in ("build_joint_field_model", "field_loglik_numpy", "field_loglik_components_numpy",
                 "anonymous_mark_logprob_numpy", "normal_quadrature"):
        monkeypatch.setattr(jfm, name, forbidden)
    repeat = _simulate("smoke")
    original = simulations["smoke"]
    _assert_same_data(repeat.data, original.data)
    _assert_same_data(repeat.truth["complete_data"], original.truth["complete_data"])
    for name in ("candidates", "anonymous", "editions"):
        pd.testing.assert_frame_equal(repeat.truth[name], original.truth[name])
    for name in original.parameters:
        np.testing.assert_array_equal(repeat.parameters[name], original.parameters[name])
    other_outcomes = jfm.simulate_joint_field(
        scenario="smoke", seed=_stage_seed("smoke", "outcomes") + 1,
        design_seed=original.seeds["design"], nuisance_seed=original.seeds["nuisance"],
    )
    pd.testing.assert_frame_equal(repeat.data.candidates[["did", "runner_id"]], other_outcomes.data.candidates[["did", "runner_id"]])
    for name in original.parameters:
        np.testing.assert_array_equal(other_outcomes.parameters[name], original.parameters[name])
    assert not repeat.truth["candidates"].equals(other_outcomes.truth["candidates"])


@pytest.mark.parametrize("scenario", ["unknown", "selection", "", None])
def test_simulator_rejects_unknown_scenario(scenario):
    with pytest.raises(ValueError):
        jfm.simulate_joint_field(scenario=scenario, seed=1, design_seed=2, nuisance_seed=3)


def test_simulation_truth_object_and_truth_columns_are_not_model_inputs(simulations):
    simulation = simulations["smoke"]
    with pytest.raises(TypeError):
        jfm.build_joint_field_model(simulation, quadrature_order=3)
    with pytest.raises(TypeError):
        _fingerprint(simulation)
    contaminated = replace(simulation.data, candidates=simulation.truth["candidates"])
    with pytest.raises(ValueError):
        jfm.build_joint_field_model(contaminated, quadrature_order=3)
    with pytest.raises(TypeError):
        jfm.build_joint_field_model(simulation.data, truth=simulation.truth)


def test_tiny_two_chain_nutpie_execution_only():
    """Exactly one 2-chain, 10-tune/10-draw execution check; no diagnostics claim."""
    import pymc as pm

    complete = _complete_data()
    tiny = replace(complete,
                   editions=complete.editions[complete.editions.did.isin(["report", "target-a"])].copy(),
                   registry=complete.registry[complete.registry.runner_id.isin(["r0", "r1"])].copy(),
                   candidates=complete.candidates[complete.candidates.did.isin(["report", "target-a"])
                                                  & complete.candidates.runner_id.isin(["r0", "r1"])].copy(),
                   anonymous=complete.anonymous[complete.anonymous.observation_id.isin(["a0", "a2", "a4"])].copy())
    tiny = jfm.mask_field_editions(tiny, ["target-a"])
    model = jfm.build_joint_field_model(tiny, quadrature_order=3)
    with model:
        trace = pm.sample(draws=10, tune=10, chains=2, cores=1, nuts_sampler="nutpie",
                          random_seed=_stage_seed("smoke", "fit"), progressbar=False,
                          compute_convergence_checks=False)
    posterior = trace["posterior"]
    assert posterior.sizes["chain"] == 2
    assert posterior.sizes["draw"] == 10
    for name in ("named_loglik", "anonymous_loglik", "field_loglik"):
        assert np.isfinite(posterior[name]).all()
    # Real nutpie output must retain the complete physical mapping, including
    # constant field shape, so prediction needs no truths or implicit defaults.
    assert set(jfm.PARAMETER_DIMS) <= set(posterior)
    prediction = jfm.predict_missing_field_counts(
        posterior, tiny, seed=_stage_seed("smoke", "prediction"),
        draws_per_chain=10, quadrature_order=3,
    )
    assert dict(prediction.sizes) == {"chain": 2, "draw": 10, "did": 1}
    assert np.isfinite(prediction.expected_total).all()
    # Sampling success is not convergence, calibration, or parameter recovery.


def test_empty_registry_and_empty_observation_tables_have_compilable_likelihood(data):
    tiny = replace(data, registry=data.registry.iloc[:0].copy(),
                   candidates=data.candidates.iloc[:0].copy(),
                   anonymous=data.anonymous.iloc[:0].copy())
    model = jfm.build_joint_field_model(tiny, quadrature_order=15)
    point = model.initial_point()
    physical = jfm.field_parameters(model, point)
    evaluated = jfm.compile_field_loglik(model)(point)
    np.testing.assert_array_equal(evaluated["named"], 0.)
    np.testing.assert_allclose(evaluated["total"], jfm.field_loglik_numpy(physical, tiny), atol=1e-12)
    assert np.isfinite(evaluated["gradient"]).all()
    assert np.isfinite(model.compile_logp()(point))


def test_nondefault_field_shape_is_explicit_in_likelihood_and_prediction(data):
    parameters = _parameters(data, homogeneous=True)
    parameters["field_shape"] = 2.5
    shape, mean, q = 2.5, 3., .7
    rate = shape / mean
    result = jfm.field_loglik_components_numpy(parameters, data)
    # No observed rows at zero: integrate exp(-q lambda) against Gamma(a,a/m).
    mass = quad(lambda lam: np.exp(-q * lam) * gamma.pdf(lam, a=shape, scale=1. / rate),
                0., np.inf, epsabs=1e-11)[0]
    assert result.anonymous.sel(did="zero").item() == pytest.approx(np.log(mass), abs=1e-10)
    prediction = _predict(_posterior(data, parameters), data)
    np.testing.assert_allclose(prediction.expected_total.sel(did="zero"), shape * (1. - q) / (rate + q))
    priors = replace(jfm.JointFieldPrior(), field_shape=shape)
    model = jfm.build_joint_field_model(data, priors=priors)
    assert jfm.field_parameters(model)["field_shape"] == shape
    incomplete = dict(parameters)
    incomplete.pop("field_shape")
    with pytest.raises(ValueError, match="field_shape"):
        jfm.field_loglik_numpy(incomplete, data)
