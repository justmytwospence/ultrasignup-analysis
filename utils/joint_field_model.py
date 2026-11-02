"""Isolated synthetic joint entrant-population model; never reads production data.

The observation unit is an edition, including supplied editions with no rows.
Named eligibility is external; anonymous identities are one-off and disjoint.
All pace values are log minutes/km conditional on finishing. No ascertainment
filter, reporting classifier, empirical donor pool, or evaluator truth is used.

Parameter arrays use lexicographically sorted string IDs (``field_coordinates``).
``PARAMETER_DIMS`` defines the complete *physical* likelihood/prediction mapping;
notably ``field_shape`` is required, never silently replaced by a default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.special import expit, gammaln, logsumexp
import xarray as xr


GROUPS = ("M20", "M40", "M60", "F20", "F40", "F60")
_TABLE_COLUMNS = {
    "editions": ("did", "course_id", "distance_km", "reported"),
    "registry": ("runner_id", "group"),
    "candidates": ("did", "runner_id", "state", "log_pace"),
    "anonymous": ("observation_id", "did", "group", "state", "log_pace"),
}


def demographic_design() -> pd.DataFrame:
    """The fixed six-row synthetic support, not a real-data age binning rule."""
    return pd.DataFrame({"group": np.arange(6), "sex": [0, 0, 0, 1, 1, 1],
                         "age": [20, 40, 60, 20, 40, 60],
                         "h": [-.5, -.5, -.5, .5, .5, .5],
                         "A": [-2., 0., 2., -2., 0., 2.]})


@dataclass
class JointFieldData:
    """Visible-only tables. Extra columns, including hidden totals, are rejected.

    ``state`` is finish/dnf/nonstarter when reported, finish/unseen otherwise.
    Anonymous rows have finish/dnf states, and only finish in suppressed editions.
    Only finish rows contain finite ``log_pace``; all other values must be null.
    Tables can have zero rows but must retain their declared columns.
    """

    editions: pd.DataFrame
    registry: pd.DataFrame
    candidates: pd.DataFrame
    anonymous: pd.DataFrame
    demographics: pd.DataFrame = field(default_factory=demographic_design)


@dataclass(frozen=True)
class JointFieldPrior:
    """Fixed illustrative prototype priors; ``asdict`` is JSON serializable.

    Outcome priors retain the existing numerical assumptions, not its builder.
    The new field/entry/edition priors are illustrative, not elicited or tuned.
    Zero Normal locations and unit raw-Normal scales are fixed by specification.
    """

    pace_mean_loc: float = float(np.log(8.))
    pace_mean_sd: float = .3
    pace_gap_sd: float = .2
    exponent_mean_loc: float = .1
    exponent_mean_sd: float = .1
    exponent_gap_sd: float = .1
    age_linear_sd: float = .1
    age_quadratic_sd: float = .05
    tau_sd: float = .3
    omega_sd: float = .3
    sigma_sd: float = .3
    dnf_mean_loc: float = -1.4
    dnf_mean_sd: float = 1.
    dnf_distance_sd: float = 1.
    dnf_gender_gap_sd: float = .5
    dnf_age_linear_sd: float = .5
    dnf_age_quadratic_sd: float = .25
    tau_dnf_sd: float = 1.
    omega_dnf_sd: float = 1.
    quadratic_sd: float = .5
    correlation_alpha: float = 2.
    correlation_beta: float = 2.
    tau_edition_pace_sd: float = .1
    tau_edition_dnf_sd: float = .5
    entry_mean_loc: float = float(-np.log(2.))
    entry_mean_sd: float = 1.
    entry_course_sd_scale: float = .7
    entry_pace_sd: float = 1.
    anonymous_mean_loc: float = float(np.log(40.))
    anonymous_mean_sd: float = 1.
    anonymous_distance_sd: float = .5
    anonymous_course_sd_scale: float = .7
    field_shape: float = 4.
    demographic_concentration: float = 20.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or (not name.endswith("_loc") and value <= 0):
                raise ValueError(f"Invalid prior constant {name}")


@dataclass(frozen=True)
class JointFieldSimulation:
    """Separate visible data from evaluator-only truth and generating parameters.

    truth['complete_data'] is an all-reporting JointFieldData; other truth tables
    are candidates, anonymous and editions (latent traits, probabilities/counts).
    This object itself is never accepted by a builder, fingerprint or predictor.
    """

    data: JointFieldData
    truth: dict
    parameters: dict
    scenario: str
    seeds: dict


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
        "entry_mean", "entry_pace", "anonymous_mean", "anonymous_distance", "field_shape")},
}


def _ids(series, name):
    if series.isna().any() or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (str, int, np.integer))
        or (isinstance(v, str) and not v.strip()) for v in series
    ):
        raise ValueError(f"{name} must contain nonempty string or integer IDs")
    # Do not silently merge distinct identities after string canonicalization.
    if len(set(series)) != len(set(series.astype(str))):
        raise ValueError(f"Ambiguous string/integer IDs in {name}")
    return series.astype(str)


def _groups(series):
    values = series.to_numpy()
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in values):
        raise ValueError("group must be an integer index 0..5")
    if len(values) and not np.isin(values, np.arange(6)).all():
        raise ValueError("group must be an integer index 0..5")


def validate_field_data(data: JointFieldData) -> None:
    """Reject invalid input explicitly, without filtering rows or inferring exposure."""
    if not isinstance(data, JointFieldData):
        raise TypeError("Expected visible JointFieldData, not a simulation or truth object")
    for name, columns in _TABLE_COLUMNS.items():
        table = getattr(data, name)
        if not isinstance(table, pd.DataFrame) or set(table.columns) != set(columns) or table.columns.has_duplicates:
            raise ValueError(f"{name} must have exactly columns {columns}")
    design = demographic_design()
    if (not isinstance(data.demographics, pd.DataFrame)
            or list(data.demographics.columns) != list(design.columns)
            or data.demographics.shape != design.shape
            or not np.array_equal(data.demographics.to_numpy(), design.to_numpy())):
        raise ValueError("demographics must equal the fixed six-row design")
    ed, reg, cand, anon = data.editions, data.registry, data.candidates, data.anonymous
    did = _ids(ed.did, "editions.did")
    courses = _ids(ed.course_id, "course_id")
    runners = _ids(reg.runner_id, "registry.runner_id")
    if did.duplicated().any() or runners.duplicated().any():
        raise ValueError("Duplicate edition or registry ID")
    distance = pd.to_numeric(ed.distance_km, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(distance).all() or (distance <= 0).any():
        raise ValueError("distance_km must be finite and positive")
    if ed.assign(course_id=courses).groupby("course_id").distance_km.nunique().gt(1).any():
        raise ValueError("A course must have one fixed distance")
    if not all(isinstance(v, (bool, np.bool_)) for v in ed.reported) or not ed.reported.any():
        raise ValueError("reported must be boolean with at least one reporting edition")
    _groups(reg.group)
    _groups(anon.group)
    reported = dict(zip(did, ed.reported, strict=True))
    for name, rows in (("candidates", cand), ("anonymous", anon)):
        row_did = _ids(rows.did, f"{name}.did")
        if not set(row_did) <= set(did):
            raise ValueError(f"Unknown edition reference in {name}")
        if name == "candidates":
            rid = _ids(rows.runner_id, "candidates.runner_id")
            if not set(rid) <= set(runners):
                raise ValueError("Unknown registered candidate runner")
            if pd.DataFrame({"did": row_did, "runner_id": rid}).duplicated().any():
                raise ValueError("Duplicate candidate edge")
        else:
            oid = _ids(rows.observation_id, "anonymous.observation_id")
            if oid.duplicated().any():
                raise ValueError("Duplicate anonymous observation ID")
            if set(oid) & set(runners):
                raise ValueError("Anonymous identities must be disjoint from named registry")
        for state, edition in zip(rows.state, row_did, strict=True):
            allowed = ({"finish", "dnf", "nonstarter"} if reported[edition] else {"finish", "unseen"})
            if name == "anonymous":
                allowed = {"finish", "dnf"} if reported[edition] else {"finish"}
            if not isinstance(state, str) or state not in allowed:
                raise ValueError(f"Illegal {name} state {state!r} for edition {edition}")
        pace = pd.to_numeric(rows.log_pace, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        finish = rows.state.to_numpy() == "finish"
        if not np.isfinite(pace[finish]).all() or not rows.log_pace[~finish].isna().all():
            raise ValueError("Only finishers may have log_pace, which must be finite")


def _canonical(data):
    validate_field_data(data)
    tables = {}
    for name, columns in _TABLE_COLUMNS.items():
        table = getattr(data, name).loc[:, columns].copy(deep=True)
        for column in set(columns) & {"did", "course_id", "runner_id", "observation_id"}:
            table[column] = table[column].astype(str)
        keys = {"editions": ["did"], "registry": ["runner_id"],
                "candidates": ["did", "runner_id"], "anonymous": ["observation_id"]}[name]
        table = table.sort_values(keys).reset_index(drop=True)
        table.attrs = {}
        tables[name] = table
    return tables


def field_coordinates(data: JointFieldData) -> dict:
    """Array/trace labels; edition/course/runner IDs are sorted strings."""
    tables = _canonical(data)
    return {"course": sorted(tables["editions"].course_id.unique()),
            "edition": tables["editions"].did.tolist(),
            "runner": tables["registry"].runner_id.tolist(),
            "sex": ["M", "F"], "group": list(GROUPS)}


def _layout(data):
    tables = _canonical(data)
    coords = field_coordinates(data)
    e, r, c, a = (tables[k] for k in ("editions", "registry", "candidates", "anonymous"))
    ei = {v: i for i, v in enumerate(coords["edition"])}
    ci = {v: i for i, v in enumerate(coords["course"])}
    ri = {v: i for i, v in enumerate(coords["runner"])}
    course_index = e.course_id.map(ci).to_numpy(dtype=int)
    course_x = np.log(e.drop_duplicates("course_id").set_index("course_id")
                      .loc[coords["course"], "distance_km"].to_numpy(dtype=float) / 50.)
    return {**tables, "coords": coords, "ec": course_index, "x": course_x,
            "ce": c.did.map(ei).to_numpy(dtype=int), "cr": c.runner_id.map(ri).to_numpy(dtype=int),
            "rg": r.group.to_numpy(dtype=int), "ae": a.did.map(ei).to_numpy(dtype=int),
            "ag": a.group.to_numpy(dtype=int), "E": len(e), "C": len(ci), "R": len(ri)}


def _parameters(parameters: Mapping, layout):
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a complete physical parameter mapping")
    result = {}
    for name, dims in PARAMETER_DIMS.items():
        if name not in parameters:
            raise ValueError(f"Missing physical parameter {name}")
        value = np.asarray(parameters[name], dtype=float)
        shape = tuple(len(layout["coords"][d]) for d in dims)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape {shape}")
        result[name] = value
    if result["field_shape"] <= 0 or np.any(result["sigma"] <= 0):
        raise ValueError("field_shape and sigma must be positive")
    if result["omega"] < 0 or result["omega_dnf"] < 0 or abs(result["rho_runner"]) > 1:
        raise ValueError("Invalid runner scale or correlation")
    theta = result["theta_course"]
    if (theta < 0).any() or not np.allclose(theta.sum(axis=1), 1., rtol=0, atol=1e-10):
        raise ValueError("theta_course rows must be probability simplexes")
    return result


def normal_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite rule for a standard Normal, normalized to unit weight."""
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or order < 3:
        raise ValueError("quadrature order must be an integer >= 3")
    nodes, weights = np.polynomial.hermite.hermgauss(int(order))
    weights = weights / np.sqrt(np.pi)
    if not np.isfinite(nodes).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Nonfinite quadrature rule")
    return nodes * np.sqrt(2.), weights / weights.sum()


def _grid(order):
    nodes, weights = normal_quadrature(order)
    z1, z2 = np.meshgrid(nodes, nodes, indexing="ij")
    return z1.ravel(), z2.ravel(), np.log(np.outer(weights, weights).ravel())


def named_state_logprobs(entry_logit, dnf_logit) -> dict:
    """Collapsed attendance/outcome probabilities, stable even for large logits."""
    le = -np.logaddexp(0., -np.asarray(entry_logit))
    ln = -np.logaddexp(0., np.asarray(entry_logit))
    lp = -np.logaddexp(0., -np.asarray(dnf_logit))
    lq = -np.logaddexp(0., np.asarray(dnf_logit))
    return {"finish": le + lq, "dnf": le + lp,
            "nonstarter": ln, "unseen": np.logaddexp(ln, le + lp)}


def named_hidden_probability(entry_logit, dnf_logit):
    """P(hidden DNF | unseen, parameters), not unconditional DNF risk."""
    terms = named_state_logprobs(entry_logit, dnf_logit)
    return np.exp(terms["dnf"] - terms["unseen"])


def gamma_intensity_log_integral(n, t, *, shape, rate):
    """log E[lambda**n exp(-t lambda)] under Gamma(shape, rate)."""
    n, t, shape, rate = np.broadcast_arrays(n, t, shape, rate)
    if (not all(np.isfinite(v).all() for v in (n, t, shape, rate))
            or (n < 0).any() or (n != np.floor(n)).any() or (t < 0).any()
            or (shape <= 0).any() or (rate <= 0).any()):
        raise ValueError("Invalid Gamma intensity arguments")
    return gammaln(shape + n) - gammaln(shape) + shape * np.log(rate) - (shape + n) * np.log(rate + t)


def anonymous_count_parameters(F, Q, *, shape, rate):
    """Conditional NB(shape+F, success=(rate+Q)/(rate+1)); valid at F=0."""
    F, Q, shape, rate = np.broadcast_arrays(F, Q, shape, rate)
    if (not all(np.isfinite(v).all() for v in (F, Q, shape, rate))
            or (F < 0).any() or (F != np.floor(F)).any() or (Q < 0).any() or (Q > 1).any()
            or (shape <= 0).any() or (rate <= 0).any()):
        raise ValueError("Invalid conditional count arguments")
    return shape + F, (rate + Q) / (rate + 1.)


def _normal_logpdf(y, mu, sd):
    return -.5 * ((y - mu) / sd)**2 - np.log(sd) - .5 * np.log(2. * np.pi)


def _numpy_base(p, L):
    # Group indexes six probabilities; sex=g//3 indexes two coefficients/noises.
    sex = np.arange(6) // 3
    h = np.where(sex == 0, -.5, .5)
    age = np.tile([-2., 0., 2.], 2)
    c = L["ec"]
    mu = (p["course_mean"][c, None] + h * (p["pace_gap"] + p["exponent_gap"] * L["x"][c, None])
          + p["age_linear"][sex] * age + p["age_quadratic"][sex] * age**2
          + p["edition_pace"][:, None])
    eta = (p["course_dnf_level"][c, None] + h * p["dnf_gender_gap"]
           + p["dnf_age_linear"] * age + p["dnf_age_quadratic"] * age**2
           + p["edition_dnf"][:, None])
    return mu, eta


def _numpy_runner_risk(p, z1, z2):
    return (p["omega_dnf"] * (p["rho_runner"] * z1 + np.sqrt(1. - p["rho_runner"]**2) * z2)
            + p["quadratic"] * (z1**2 - 1.))


def _numpy_marks(p, L, order):
    z1, z2, lw = _grid(order)
    mu, eta = _numpy_base(p, L)
    risk = eta[:, :, None] + _numpy_runner_risk(p, z1, z2)
    lq = logsumexp(lw - np.logaddexp(0., risk), axis=-1)
    lp = logsumexp(lw - np.logaddexp(0., -risk), axis=-1)
    with np.errstate(divide="ignore"):
        lt = np.log(p["theta_course"][L["ec"]])
    log_Q = logsumexp(lt + lq, axis=1)
    log_P = logsumexp(lt + lp, axis=1)
    finish = np.flatnonzero(L["anonymous"].state.to_numpy() == "finish")
    e, g = L["ae"][finish], L["ag"][finish]
    y = L["anonymous"].log_pace.to_numpy(dtype=float)[finish]
    sd = p["sigma"][g // 3]
    variance = p["omega"]**2 + sd**2
    # Bayes' Gaussian identity moves the z1 grid to z1 | Y=y, leaving z2 iid.
    conditional_z1 = (p["omega"] * (y - mu[e, g]) / variance)[:, None] + (sd / np.sqrt(variance))[:, None] * z1
    conditional_eta = eta[e, g, None] + _numpy_runner_risk(p, conditional_z1, z2)
    hf = (lt[e, g] + _normal_logpdf(y, mu[e, g], np.sqrt(variance))
          + logsumexp(lw - np.logaddexp(0., conditional_eta), axis=-1))
    return {"log_q_group": lq, "log_p_group": lp, "log_Q": log_Q,
            "log_one_minus_Q": log_P, "log_H_F": hf, "log_H_D": lt + lp,
            "finish": finish}


def anonymous_mark_logprob_numpy(parameters, data, *, quadrature_order=15) -> xr.Dataset:
    """Expose integrated mark probabilities for independent numerical oracles."""
    L = _layout(data)
    p = _parameters(parameters, L)
    marks = _numpy_marks(p, L, quadrature_order)
    values = {k: (("did", "group"), marks[k]) for k in ("log_q_group", "log_p_group", "log_H_D")}
    values["log_H_F"] = (("anonymous_finish",), marks["log_H_F"])
    return xr.Dataset(values, coords={"did": L["coords"]["edition"], "group": list(GROUPS),
                                     "anonymous_finish": L["anonymous"].observation_id.to_numpy()[marks["finish"]]})


def _numpy_named(p, L):
    mu, eta = _numpy_base(p, L)
    e, r = L["ce"], L["cr"]
    g = L["rg"][r]
    z1, z2 = p["runner_z1"][r], p["runner_z2"][r]
    entry = p["entry_mean"] + p["entry_course"][L["ec"][e]] + p["entry_pace"] * z1
    dnf = eta[e, g] + _numpy_runner_risk(p, z1, z2)
    terms = named_state_logprobs(entry, dnf)
    state = L["candidates"].state.to_numpy()
    row_logp = np.zeros(len(e))
    for key, logp in terms.items():
        use = state == key
        row_logp[use] = logp[use]
    finish = state == "finish"
    y = L["candidates"].log_pace.to_numpy(dtype=float)[finish]
    row_logp[finish] += _normal_logpdf(y, mu[e[finish], g[finish]] + p["omega"] * z1[finish],
                                      p["sigma"][g[finish] // 3])
    return np.bincount(e, weights=row_logp, minlength=L["E"]), entry, dnf


def _numpy_components(p, L, order):
    marks = _numpy_marks(p, L, order)
    named, _, _ = _numpy_named(p, L)
    f = marks["finish"]
    d = np.flatnonzero(L["anonymous"].state.to_numpy() == "dnf")
    F = np.bincount(L["ae"][f], minlength=L["E"])
    K = np.bincount(L["ae"][d], minlength=L["E"])
    reported = L["editions"].reported.to_numpy(dtype=bool)
    n = F + K
    a = p["field_shape"]
    log_b = np.log(a) - p["anonymous_mean"] - p["anonymous_distance"] * L["x"][L["ec"]] - p["anonymous_course"][L["ec"]]
    log_t = np.where(reported, 0., marks["log_Q"])
    anonymous = (gammaln(a + n) - gammaln(a) + a * log_b
                 - (a + n) * np.logaddexp(log_b, log_t) - gammaln(F + 1) - gammaln(K + 1))
    anonymous += np.bincount(L["ae"][f], weights=marks["log_H_F"], minlength=L["E"])
    anonymous += np.bincount(L["ae"][d], weights=marks["log_H_D"][L["ae"][d], L["ag"][d]], minlength=L["E"])
    return {"named": named, "anonymous": anonymous, "total": named + anonymous,
            "Q": np.exp(marks["log_Q"]), "log_Q": marks["log_Q"],
            "log_one_minus_Q": marks["log_one_minus_Q"]}


def field_loglik_components_numpy(parameters, data, *, quadrature_order=15) -> xr.Dataset:
    """Per-edition likelihood and integration diagnostics; no prior terms."""
    L = _layout(data)
    p = _parameters(parameters, L)
    values = _numpy_components(p, L, quadrature_order)
    return xr.Dataset({k: ("did", v) for k, v in values.items()}, coords={"did": L["coords"]["edition"]})


def field_loglik_numpy(parameters, data, *, quadrature_order=15) -> xr.DataArray:
    """Log likelihood (named + anonymous) per edition, without prior density."""
    return field_loglik_components_numpy(parameters, data, quadrature_order=quadrature_order)["total"]


def mask_field_editions(complete_data: JointFieldData, dids) -> JointFieldData:
    """Copy and hide outcomes, preserving registry, candidate edges and empty races."""
    tables = _canonical(complete_data)
    ids = [str(v) for v in dids]
    if len(set(ids)) != len(ids) or not set(ids) <= set(tables["editions"].did):
        raise ValueError("Mask IDs must be unique known editions")
    e, c, a = (tables[k] for k in ("editions", "candidates", "anonymous"))
    e.loc[e.did.isin(ids), "reported"] = False
    c.loc[c.did.isin(ids) & c.state.ne("finish"), "state"] = "unseen"
    tables["anonymous"] = a.loc[~(a.did.isin(ids) & a.state.eq("dnf"))].reset_index(drop=True)
    result = JointFieldData(**tables)
    validate_field_data(result)
    return result


def field_fingerprint(data, priors, *, quadrature_order, sampler, seed, version) -> str:
    """Hash canonical *visible* tables and every prior/numerical/sampling setting."""
    tables = _canonical(data)
    normal_quadrature(quadrature_order)
    if not isinstance(priors, JointFieldPrior):
        raise TypeError("priors must be JointFieldPrior")
    payload = {"tables": {k: v.astype(object).where(v.notna(), None).to_dict(orient="records") for k, v in tables.items()},
               "demographics": demographic_design().to_dict(orient="list"), "priors": asdict(priors),
               "quadrature_order": int(quadrature_order), "sampler": sampler, "seed": int(seed), "version": version}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _prior_parameter_draw(rng, priors, C, E, R):
    """Independent forward prior sampler: no PyMC, likelihood or quadrature calls."""
    p = {}
    for name in ("pace_mean", "exponent_mean", "dnf_mean", "entry_mean", "anonymous_mean"):
        p[name] = rng.normal(getattr(priors, name + "_loc"), getattr(priors, name + "_sd"))
    for name in ("pace_gap", "exponent_gap", "dnf_distance", "dnf_gender_gap", "dnf_age_linear",
                 "dnf_age_quadratic", "quadratic", "entry_pace", "anonymous_distance"):
        p[name] = rng.normal(0., getattr(priors, name + "_sd"))
    for name in ("age_linear", "age_quadratic"):
        p[name] = rng.normal(0., getattr(priors, name + "_sd"), 2)
    for name in ("tau", "omega", "tau_dnf", "omega_dnf", "tau_edition_pace", "tau_edition_dnf"):
        p[name] = abs(rng.normal(0., getattr(priors, name + "_sd")))
    p["sigma"] = abs(rng.normal(0., priors.sigma_sd, 2))
    for name in ("entry_course_sd", "anonymous_course_sd"):
        p[name] = abs(rng.normal(0., getattr(priors, name + "_scale")))
    for name in ("rho_course", "rho_runner", "rho_edition"):
        p[name] = 2. * rng.beta(priors.correlation_alpha, priors.correlation_beta) - 1.
    for name, size in (("course_z1", C), ("course_z2", C), ("edition_z1", E), ("edition_z2", E),
                       ("runner_z1", R), ("runner_z2", R), ("entry_course_raw", C), ("anonymous_course_raw", C)):
        p[name] = rng.normal(size=size)
    p["theta_global"] = rng.dirichlet(np.ones(6))
    p["theta_course"] = np.array([rng.dirichlet(priors.demographic_concentration * p["theta_global"]) for _ in range(C)])
    p["field_shape"] = priors.field_shape
    return p


def simulate_joint_field(*, scenario, seed, design_seed, nuisance_seed, priors=JointFieldPrior()) -> JointFieldSimulation:
    """Independent forward simulation, with separate design/nuisance/outcome RNGs.

    Main cases share nuisance/design seeds but use distinct outcome seeds. Prior
    draws have no truth overrides. Misspecification shifts only anonymous z1 by
    +/- .75 by course; the fitted population deliberately remains N(0,1).
    """
    scenarios = {"prior", "smoke", "benign", "race_variation", "participation_selection", "population_misspecification"}
    if scenario not in scenarios:
        raise ValueError(f"Unknown simulation scenario {scenario!r}")
    design_rng, nuisance_rng, rng = (np.random.default_rng(s) for s in (design_seed, nuisance_seed, seed))
    smoke = scenario == "smoke"
    distances = np.array([50., 100.] if smoke else [50., 50., 80., 100., 130., 160.])
    C, per_course, R, candidates_per_edition = len(distances), (2 if smoke else 4), (12 if smoke else 60), (8 if smoke else 40)
    E = C * per_course
    courses = [f"course-{c}" for c in range(C)]
    runners = [f"runner-{r:02d}" for r in range(R)]
    eds = [(f"edition-{e:02d}", courses[e // per_course], distances[e // per_course], True) for e in range(E)]
    editions = pd.DataFrame(eds, columns=_TABLE_COLUMNS["editions"])
    registry = pd.DataFrame({"runner_id": runners, "group": np.arange(R) % 6})
    # This graph is sampled before, and independently of, all latent traits/outcomes.
    graph = [np.sort(design_rng.choice(R, candidates_per_edition, replace=False)) for _ in range(E)]
    p = _prior_parameter_draw(nuisance_rng, priors, C, E, R)
    if smoke:
        p["anonymous_mean"] = np.log(8.)
    elif scenario != "prior":
        p.update(anonymous_mean=np.log(40.), anonymous_distance=0.,
                 entry_mean=-np.log(2.), tau_edition_pace=.1, omega=.3, omega_dnf=.8,
                 sigma=np.array([.2, .2]), quadratic=.25)
        variation = scenario in {"race_variation", "participation_selection", "population_misspecification"}
        selection = scenario in {"participation_selection", "population_misspecification"}
        p.update(tau_edition_dnf=.9 if variation else .3, rho_edition=.6 if variation else 0.,
                 entry_pace=-.8 if selection else 0.)
    x = np.log(distances / 50.)
    p["course_mean"] = p["pace_mean"] + p["exponent_mean"] * x + p["tau"] * p["course_z1"]
    p["course_dnf_level"] = (p["dnf_mean"] + p["dnf_distance"] * x + p["tau_dnf"] *
                             (p["rho_course"] * p["course_z1"] + np.sqrt(1 - p["rho_course"]**2) * p["course_z2"]))
    p["edition_pace"] = p["tau_edition_pace"] * p["edition_z1"]
    p["edition_dnf"] = p["tau_edition_dnf"] * (p["rho_edition"] * p["edition_z1"] + np.sqrt(1 - p["rho_edition"]**2) * p["edition_z2"])
    p["entry_course"] = p["entry_course_sd"] * p["entry_course_raw"]
    p["anonymous_course"] = p["anonymous_course_sd"] * p["anonymous_course_raw"]
    named_rows, anon_rows, edition_truth = [], [], []

    def outcome(e, g, z1, z2):
        c, sex = e // per_course, g // 3
        age, h = [-2., 0., 2.][g % 3], [-.5, .5][sex]
        mu = (p["course_mean"][c] + h * (p["pace_gap"] + p["exponent_gap"] * x[c])
              + p["age_linear"][sex] * age + p["age_quadratic"][sex] * age**2
              + p["omega"] * z1 + p["edition_pace"][e])
        eta = (p["course_dnf_level"][c] + h * p["dnf_gender_gap"] + p["dnf_age_linear"] * age
               + p["dnf_age_quadratic"] * age**2 + p["edition_dnf"][e]
               + p["omega_dnf"] * (p["rho_runner"] * z1 + np.sqrt(1 - p["rho_runner"]**2) * z2)
               + p["quadratic"] * (z1*z1 - 1.))
        probability = expit(eta)
        dnf = rng.random() < probability
        return ("dnf" if dnf else "finish"), (np.nan if dnf else rng.normal(mu, p["sigma"][sex])), probability

    for e in range(E):
        c, did = e // per_course, eds[e][0]
        ns = nd = ad = 0
        for r in graph[e]:
            z1, z2 = p["runner_z1"][r], p["runner_z2"][r]
            entry = expit(p["entry_mean"] + p["entry_course"][c] + p["entry_pace"] * z1)
            if rng.random() < entry:
                state, pace, risk = outcome(e, r % 6, z1, z2)
                ns += 1
                nd += state == "dnf"
            else:
                state, pace, risk = "nonstarter", np.nan, np.nan
            named_rows.append((did, runners[r], state, pace, z1, z2, entry, risk))
        mean = np.exp(p["anonymous_mean"] + p["anonymous_distance"] * x[c] + p["anonymous_course"][c])
        intensity = rng.gamma(p["field_shape"], mean / p["field_shape"])
        if not np.isfinite(intensity):
            raise ValueError("Nonfinite simulated field intensity; do not redraw")
        N = rng.poisson(intensity)
        for j in range(N):
            g = rng.choice(6, p=p["theta_course"][c])
            z1, z2 = rng.normal(size=2)
            if scenario == "population_misspecification":
                z1 += .75 if c % 2 == 0 else -.75
            state, pace, risk = outcome(e, g, z1, z2)
            ad += state == "dnf"
            anon_rows.append((f"anonymous-{e:02d}-{j:06d}", did, g, state, pace, z1, z2, risk))
        edition_truth.append((did, ns, N, nd, ad, nd + ad, intensity))
    named = pd.DataFrame(named_rows, columns=(*_TABLE_COLUMNS["candidates"], "z1", "z2", "entry_probability", "dnf_probability"))
    anon = pd.DataFrame(anon_rows, columns=(*_TABLE_COLUMNS["anonymous"], "z1", "z2", "dnf_probability"))
    complete = JointFieldData(editions, registry, named.loc[:, _TABLE_COLUMNS["candidates"]].copy(),
                              anon.loc[:, _TABLE_COLUMNS["anonymous"]].copy())
    mask = [eds[c * per_course + c % per_course][0] for c in range(C)]
    visible = mask_field_editions(complete, mask)
    truth = {"complete_data": complete, "candidates": named, "anonymous": anon,
             "editions": pd.DataFrame(edition_truth, columns=("did", "named_starters", "anonymous_starters", "named_dnf", "anonymous_dnf", "total_dnf", "lambda"))}
    return JointFieldSimulation(visible, truth, p, scenario,
                                {"outcomes": seed, "design": design_seed, "nuisance": nuisance_seed})


def _field_likelihood_pt(p, L, order):
    """Continuous marginalized graph; no random anonymous traits or missing counts."""
    import pytensor.tensor as pt

    z1, z2, lw = _grid(order)
    c = L["ec"]
    sex = np.arange(6) // 3
    h, age = np.repeat([-.5, .5], 3), np.tile([-2., 0., 2.], 2)
    base_mu = (p["course_mean"][c, None] + h * (p["pace_gap"] + p["exponent_gap"] * L["x"][c, None])
               + p["age_linear"][sex] * age + p["age_quadratic"][sex] * age**2 + p["edition_pace"][:, None])
    base_eta = (p["course_dnf_level"][c, None] + h * p["dnf_gender_gap"]
                + p["dnf_age_linear"] * age + p["dnf_age_quadratic"] * age**2 + p["edition_dnf"][:, None])

    def runner_risk(first, second):
        return (p["omega_dnf"] * (p["rho_runner"] * first + pt.sqrt(1. - p["rho_runner"]**2) * second)
                + p["quadratic"] * (first**2 - 1.))

    e, r = L["ce"], L["cr"]
    g = L["rg"][r]
    named = pt.zeros(L["E"], dtype="float64")
    if len(e):
        logit_entry = p["entry_mean"] + p["entry_course"][c[e]] + p["entry_pace"] * p["runner_z1"][r]
        logit_dnf = base_eta[e, g] + runner_risk(p["runner_z1"][r], p["runner_z2"][r])
        le, ln = -pt.softplus(-logit_entry), -pt.softplus(logit_entry)
        lp, lq = -pt.softplus(-logit_dnf), -pt.softplus(logit_dnf)
        for state, terms in (("finish", le + lq), ("dnf", le + lp), ("nonstarter", ln),
                             ("unseen", pt.logaddexp(ln, le + lp))):
            rows = np.flatnonzero(L["candidates"].state.to_numpy() == state)
            if not len(rows):
                continue
            value = terms[rows]
            if state == "finish":
                y = L["candidates"].log_pace.to_numpy(dtype=float)[rows]
                mu = base_mu[e[rows], g[rows]] + p["omega"] * p["runner_z1"][r[rows]]
                sd = p["sigma"][g[rows] // 3]
                value = value - .5 * ((y - mu) / sd)**2 - pt.log(sd) - .5 * np.log(2. * np.pi)
            named = pt.inc_subtensor(named[e[rows]], value)

    eta_grid = base_eta[:, :, None] + runner_risk(z1, z2)
    log_qg = pt.logsumexp(lw - pt.softplus(eta_grid), axis=-1)
    log_pg = pt.logsumexp(lw - pt.softplus(-eta_grid), axis=-1)
    log_theta = pt.log(p["theta_course"][c])
    log_Q = pt.logsumexp(log_theta + log_qg, axis=1)
    anon = L["anonymous"]
    f = np.flatnonzero(anon.state.to_numpy() == "finish")
    d = np.flatnonzero(anon.state.to_numpy() == "dnf")
    F = np.bincount(L["ae"][f], minlength=L["E"])
    K = np.bincount(L["ae"][d], minlength=L["E"])
    a = p["field_shape"]
    log_b = pt.log(a) - p["anonymous_mean"] - p["anonymous_distance"] * L["x"][c] - p["anonymous_course"][c]
    log_t = pt.where(L["editions"].reported.to_numpy(dtype=bool), 0., log_Q)
    anonymous = (pt.gammaln(a + F + K) - pt.gammaln(a) + a * log_b
                 - (a + F + K) * pt.logaddexp(log_b, log_t) - gammaln(F + 1) - gammaln(K + 1))
    if len(f):
        e, g = L["ae"][f], L["ag"][f]
        y = anon.log_pace.to_numpy(dtype=float)[f]
        sd = p["sigma"][g // 3]
        variance = p["omega"]**2 + sd**2
        residual = y - base_mu[e, g]
        cond_z1 = (p["omega"] * residual / variance)[:, None] + (sd / pt.sqrt(variance))[:, None] * z1
        cond_eta = base_eta[e, g, None] + runner_risk(cond_z1, z2)
        mark = (log_theta[e, g] - .5 * residual**2 / variance - .5 * pt.log(variance)
                - .5 * np.log(2. * np.pi) + pt.logsumexp(lw - pt.softplus(cond_eta), axis=-1))
        anonymous = pt.inc_subtensor(anonymous[e], mark)
    if len(d):
        e, g = L["ae"][d], L["ag"][d]
        anonymous = pt.inc_subtensor(anonymous[e], log_theta[e, g] + log_pg[e, g])
    return named, anonymous


def build_joint_field_model(data, *, priors=JointFieldPrior(), quadrature_order=15):
    """Build the isolated, noncentered continuous model with two Potentials.

    Each deterministic log-likelihood vector is indexed by ``edition``. Fixed
    ``field_shape`` is saved as a deterministic so posterior mappings remain
    complete, including under changed prototype assumptions. No anonymous
    per-observation quadrature tensors are retained in traces.
    """
    import pymc as pm
    import pytensor.tensor as pt

    if not isinstance(priors, JointFieldPrior):
        raise TypeError("priors must be JointFieldPrior")
    L = _layout(data)
    normal_quadrature(quadrature_order)
    with pm.Model(coords=L["coords"]) as model:
        p = {}
        # Numerical outcome priors retained from dnf_model; new priors are
        # illustrative prototype assumptions fixed before observing simulations.
        for name in ("pace_mean", "exponent_mean", "dnf_mean", "entry_mean", "anonymous_mean"):
            p[name] = pm.Normal(name, getattr(priors, name + "_loc"), getattr(priors, name + "_sd"))
        for name in ("pace_gap", "exponent_gap", "dnf_distance", "dnf_gender_gap", "dnf_age_linear",
                     "dnf_age_quadratic", "quadratic", "entry_pace", "anonymous_distance"):
            p[name] = pm.Normal(name, 0., getattr(priors, name + "_sd"))
        for name in ("age_linear", "age_quadratic"):
            p[name] = pm.Normal(name, 0., getattr(priors, name + "_sd"), dims="sex")
        for name in ("tau", "omega", "tau_dnf", "omega_dnf", "tau_edition_pace", "tau_edition_dnf"):
            p[name] = pm.HalfNormal(name, getattr(priors, name + "_sd"))
        p["sigma"] = pm.HalfNormal("sigma", priors.sigma_sd, dims="sex")
        for name in ("entry_course_sd", "anonymous_course_sd"):
            p[name] = pm.HalfNormal(name, getattr(priors, name + "_scale"))
        # Scalar Beta correlations avoid LKJCholeskyCov's numba cumsum(axis=).
        for name in ("rho_course", "rho_runner", "rho_edition"):
            p[name] = pm.Deterministic(name, 2. * pm.Beta(name + "_01", priors.correlation_alpha, priors.correlation_beta) - 1.)
        for name, dim in (("course_z1", "course"), ("course_z2", "course"),
                          ("edition_z1", "edition"), ("edition_z2", "edition"),
                          ("runner_z1", "runner"), ("runner_z2", "runner"),
                          ("entry_course_raw", "course"), ("anonymous_course_raw", "course")):
            p[name] = pm.Normal(name, 0., 1., dims=dim)
        # No registry zero-sum transform: named and anonymous traits share N(0,I).
        p["course_mean"] = pm.Deterministic("course_mean", p["pace_mean"] + p["exponent_mean"] * L["x"] + p["tau"] * p["course_z1"], dims="course")
        p["course_dnf_level"] = pm.Deterministic("course_dnf_level", p["dnf_mean"] + p["dnf_distance"] * L["x"]
            + p["tau_dnf"] * (p["rho_course"] * p["course_z1"] + pt.sqrt(1. - p["rho_course"]**2) * p["course_z2"]), dims="course")
        p["edition_pace"] = pm.Deterministic("edition_pace", p["tau_edition_pace"] * p["edition_z1"], dims="edition")
        p["edition_dnf"] = pm.Deterministic("edition_dnf", p["tau_edition_dnf"] *
            (p["rho_edition"] * p["edition_z1"] + pt.sqrt(1. - p["rho_edition"]**2) * p["edition_z2"]), dims="edition")
        p["entry_course"] = pm.Deterministic("entry_course", p["entry_course_sd"] * p["entry_course_raw"], dims="course")
        p["anonymous_course"] = pm.Deterministic("anonymous_course", p["anonymous_course_sd"] * p["anonymous_course_raw"], dims="course")
        p["theta_global"] = pm.Dirichlet("theta_global", np.ones(6), dims="group")
        p["theta_course"] = pm.Dirichlet("theta_course", priors.demographic_concentration * p["theta_global"], dims=("course", "group"))
        p["field_shape"] = pm.Deterministic("field_shape", pt.as_tensor_variable(np.float64(priors.field_shape)))
        named, anonymous = _field_likelihood_pt(p, L, quadrature_order)
        pm.Deterministic("named_loglik", named, dims="edition")
        pm.Deterministic("anonymous_loglik", anonymous, dims="edition")
        pm.Deterministic("field_loglik", named + anonymous, dims="edition")
        pm.Potential("named_likelihood", named.sum())
        pm.Potential("anonymous_likelihood", anonymous.sum())
    model.joint_field_layout = L
    model.joint_field_priors = priors
    model.joint_field_order = quadrature_order
    return model


def field_reference_points(model) -> dict:
    """Initial and two fixed sinusoidal perturbations in transformed coordinates.

    Perturb every free coordinate by +/-0.15*sin(1,2,...), including simplex,
    positive-scale and correlation transforms, without RNG or outcome selection.
    Points can be reused across orders built on the same data/prior specification.
    """
    initial = {k: np.asarray(v).copy() for k, v in model.initial_point().items()}
    result = {"initial": initial}
    for label, sign in (("perturbed_plus", 1.), ("perturbed_minus", -1.)):
        cursor, point = 0, {}
        for variable in model.value_vars:
            value = initial[variable.name]
            delta = .15 * np.sin(np.arange(cursor + 1, cursor + value.size + 1)).reshape(value.shape)
            point[variable.name] = value + sign * delta
            cursor += value.size
        result[label] = point
    return result


def _point_function(model, expressions):
    import pytensor

    outputs = model.replace_rvs_by_values(expressions)
    function = pytensor.function(model.value_vars, outputs, on_unused_input="ignore")
    names = [v.name for v in model.value_vars]

    def evaluate(point):
        if set(point) != set(names):
            raise ValueError("Point keys must match model's transformed value variables")
        return function(*(np.asarray(point[name]) for name in names))

    return evaluate


def field_parameters(model, point=None) -> dict:
    """Evaluate a complete physical mapping at a transformed model point.

    This is deterministic (unlike evaluating unresolved RVs with ``.eval()``).
    Results can be passed directly to ``field_loglik_numpy`` or assembled with
    ``PARAMETER_DIMS`` and ``field_coordinates`` into a prediction Dataset.
    """
    if not hasattr(model, "joint_field_layout"):
        raise TypeError("Expected a build_joint_field_model model")
    if not hasattr(model, "_field_parameter_function"):
        model._field_parameter_function = _point_function(model, [model[name] for name in PARAMETER_DIMS])
    values = model._field_parameter_function(model.initial_point() if point is None else point)
    return _parameters(dict(zip(PARAMETER_DIMS, values, strict=True)), model.joint_field_layout)


def compile_field_loglik(model):
    """Return callable(point) -> named/anonymous/total/gradient NumPy arrays.

    Gradient is of the summed likelihood, NOT the prior or transform Jacobian,
    in flattened ``model.value_vars`` order (the sampler's coordinates).
    """
    import pytensor
    import pytensor.tensor as pt

    expressions = model.replace_rvs_by_values([model["named_loglik"], model["anonymous_loglik"], model["field_loglik"]])
    gradients = pt.grad(expressions[-1].sum(), model.value_vars, disconnected_inputs="ignore")
    gradient = pt.concatenate([g.ravel() for g in gradients])
    function = pytensor.function(model.value_vars, [*expressions, gradient], on_unused_input="ignore")
    names = [v.name for v in model.value_vars]

    def evaluate(point):
        if set(point) != set(names):
            raise ValueError("Point keys must match model's transformed value variables")
        values = function(*(np.asarray(point[name]) for name in names))
        return dict(zip(("named", "anonymous", "total", "gradient"), values, strict=True))

    return evaluate


def audit_field_quadrature(data, *, priors=JointFieldPrior(), points=None, lower_order=15, upper_order=25,
                           loglik_tolerance=1e-3, gradient_tolerance=1e-3) -> xr.Dataset:
    """Pre-fit adjacent-order check at initial + fixed perturbed points by default.

    Gates are strict: max absolute per-edition difference <1e-3 and max gradient
    difference / (1+abs(higher-order gradient)) <1e-3. Nonfinite evidence fails.
    Gradients use the total likelihood in transformed sampler coordinates.
    No sampling, prior-density terms or automatic scientific repair occurs here.
    """
    if upper_order <= lower_order or loglik_tolerance <= 0 or gradient_tolerance <= 0:
        raise ValueError("Require increasing quadrature orders and positive tolerances")
    lower = build_joint_field_model(data, priors=priors, quadrature_order=lower_order)
    upper = build_joint_field_model(data, priors=priors, quadrature_order=upper_order)
    if points is None:
        points = field_reference_points(lower)
    if not isinstance(points, Mapping) or not points:
        raise ValueError("points must be a nonempty labeled mapping of model points")
    evaluate_low, evaluate_high = compile_field_loglik(lower), compile_field_loglik(upper)
    differences, gradients, low_values, high_values = [], [], [], []
    finite = True
    for point in points.values():
        lo, hi = evaluate_low(point), evaluate_high(point)
        finite &= all(np.isfinite(v).all() for result in (lo, hi) for v in result.values())
        differences.append(np.abs(lo["total"] - hi["total"]))
        gradients.append(np.max(np.abs(lo["gradient"] - hi["gradient"]) / (1. + np.abs(hi["gradient"]))))
        low_values.append(lo["total"])
        high_values.append(hi["total"])
    differences, gradients = np.array(differences), np.array(gradients)
    passed = finite and np.all(differences < loglik_tolerance) and np.all(gradients < gradient_tolerance)
    return xr.Dataset({"absolute_loglik_difference": (("point", "did"), differences),
                       "max_scaled_gradient_difference": ("point", gradients),
                       "lower_loglik": (("point", "did"), low_values), "upper_loglik": (("point", "did"), high_values)},
                      coords={"point": list(points), "did": lower.joint_field_layout["coords"]["edition"]},
                      attrs={"passed": bool(passed), "finite": bool(finite), "lower_order": lower_order,
                             "upper_order": upper_order, "loglik_tolerance": loglik_tolerance,
                             "gradient_tolerance": gradient_tolerance, "gradient_coordinates": "transformed summed likelihood"})


def select_field_quadrature(data, *, priors=JointFieldPrior(), points=None):
    """Return (chosen_order, list_of_audits); stop if both prescribed gates fail."""
    audits = []
    for lower, upper in ((15, 25), (25, 35)):
        audit = audit_field_quadrature(data, priors=priors, points=points, lower_order=lower, upper_order=upper)
        audits.append(audit)
        if audit.attrs["passed"]:
            return lower, audits
    error = ValueError("Quadrature unresolved at 25 versus 35: stop before sampling")
    error.audits = audits
    raise error


def _posterior_inputs(posterior, data, draws):
    if not isinstance(posterior, xr.Dataset):
        raise TypeError("posterior must be an xarray Dataset with physical parameters")
    for dim in ("chain", "draw"):
        if (dim not in posterior.coords or dim not in posterior.dims or posterior.sizes[dim] == 0
                or posterior.get_index(dim).has_duplicates or posterior.get_index(dim).hasnans):
            raise ValueError(f"Missing, empty or duplicate posterior {dim} coordinates")
    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, (int, np.integer)) or not 1 <= draws <= posterior.sizes["draw"]:
        raise ValueError("Requested posterior draws exceed available draws or are not positive")
    L = _layout(data)
    selectors = {}
    for dim, labels in L["coords"].items():
        if (dim not in posterior.coords or posterior.get_index(dim).has_duplicates
                or list(posterior[dim].dims) != [dim] or set(posterior[dim].values) != set(labels)):
            raise ValueError(f"Wrong posterior {dim} coordinates")
        selectors[dim] = labels
    for name, dims in PARAMETER_DIMS.items():
        if name not in posterior or set(posterior[name].dims) != {"chain", "draw", *dims}:
            raise ValueError(f"Missing or incorrectly dimensioned posterior parameter {name}")
    indices = np.linspace(0, posterior.sizes["draw"] - 1, draws, dtype=int)
    return posterior.sel(selectors).isel(draw=indices), L


def _posterior_parameters(draw, L):
    return _parameters({name: draw[name].transpose(*dims).values for name, dims in PARAMETER_DIMS.items()}, L)


def audit_posterior_quadrature(posterior, data, *, quadrature_order, draws_per_chain=20, tolerance=1e-3) -> xr.Dataset:
    """Adjacent-order likelihood audit at 20 evenly spaced draws/chain by default.

    Returns failed evidence rather than rerunning a fit. Only chosen orders 15
    and 25 are supported; their reference orders are 25 and 35 respectively.
    """
    if quadrature_order not in (15, 25) or tolerance <= 0:
        raise ValueError("Posterior audit requires chosen order 15 or 25 and positive tolerance")
    post, L = _posterior_inputs(posterior, data, draws_per_chain)
    differences = np.empty((post.sizes["chain"], post.sizes["draw"], L["E"]))
    finite = True
    for i in range(post.sizes["chain"]):
        for j in range(post.sizes["draw"]):
            p = _posterior_parameters(post.isel(chain=i, draw=j), L)
            low = _numpy_components(p, L, quadrature_order)["total"]
            high = _numpy_components(p, L, quadrature_order + 10)["total"]
            finite &= np.isfinite(low).all() and np.isfinite(high).all()
            differences[i, j] = np.abs(low - high)
    return xr.Dataset({"absolute_loglik_difference": (("chain", "draw", "did"), differences)},
                      coords={"chain": post.chain, "draw": post.draw, "did": L["coords"]["edition"]},
                      attrs={"passed": bool(finite and np.all(differences < tolerance)), "finite": bool(finite),
                             "lower_order": quadrature_order, "upper_order": quadrature_order + 10, "tolerance": tolerance})


def _target_rng(seed, chain, draw, did, stream):
    payload = json.dumps([int(seed), str(chain), str(draw), str(did), stream], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest, dtype="<u4")))


def predict_missing_field_counts(posterior, data, *, seed, draws_per_chain=500, quadrature_order=15) -> xr.Dataset:
    """Conditional named Bernoulli + anonymous NB counts on common posterior draws.

    Uses each edition's fitted effects, never fresh edition-effect draws. Streams
    are keyed by chain/draw/did labels; input row/target order cannot change draws.
    Output includes the conditional expectation *before* count noise, appropriate
    for assessing Monte Carlo precision. Physical parameters must all be present.
    """
    normal_quadrature(quadrature_order)
    post, L = _posterior_inputs(posterior, data, draws_per_chain)
    targets = np.flatnonzero(~L["editions"].reported.to_numpy(dtype=bool))
    shape = (post.sizes["chain"], post.sizes["draw"], len(targets))
    named_count, anonymous_count = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    expected = np.zeros(shape)
    endpoint_clips = 0
    f = L["anonymous"].state.to_numpy() == "finish"
    F = np.bincount(L["ae"][f], minlength=L["E"])
    unseen = L["candidates"].state.to_numpy() == "unseen"
    for i, chain in enumerate(post.chain.values):
        for j, draw in enumerate(post.draw.values):
            p = _posterior_parameters(post.isel(chain=i, draw=j), L)
            marks = _numpy_marks(p, L, quadrature_order)
            _, entry, dnf = _numpy_named(p, L)
            hidden = named_hidden_probability(entry, dnf)
            if not np.isfinite(hidden).all() or (hidden < 0).any() or (hidden > 1).any():
                raise ValueError("Invalid conditional named probabilities")
            for k, e in enumerate(targets):
                did = L["coords"]["edition"][e]
                probs = hidden[unseen & (L["ce"] == e)]
                a = float(p["field_shape"])
                log_b = np.log(a) - p["anonymous_mean"] - p["anonymous_distance"] * L["x"][L["ec"][e]] - p["anonymous_course"][L["ec"][e]]
                # log-space NB probability and mean remain stable for large m.
                # Equivalent odds form uses Q + P = 1 and avoids subtracting
                # two nearly equal logs (which can round a probability above 1).
                log_success = -np.logaddexp(0., marks["log_one_minus_Q"][e]
                                            - np.logaddexp(log_b, marks["log_Q"][e]))
                success = np.exp(log_success)
                if not np.isfinite(success) or not 0. <= success <= 1.:
                    raise ValueError("Invalid anonymous conditional success probability")
                # Only representational endpoints are moved; never broad epsilon clipping.
                if success == 0.:
                    success = np.nextafter(0., 1.)
                    endpoint_clips += 1
                elif success == 1.:
                    success = np.nextafter(1., 0.)
                    endpoint_clips += 1
                nb_shape = a + F[e]
                mean = nb_shape * np.exp(marks["log_one_minus_Q"][e] - np.logaddexp(log_b, marks["log_Q"][e]))
                if not np.isfinite(mean):
                    raise ValueError("Nonfinite conditional count expectation")
                named_count[i, j, k] = _target_rng(seed, chain, draw, did, "named").binomial(1, probs).sum()
                try:
                    anonymous_count[i, j, k] = _target_rng(seed, chain, draw, did, "anonymous").negative_binomial(nb_shape, success)
                except (ValueError, OverflowError) as error:
                    raise ValueError(f"Conditional count exceeds numerical sampler range for {did}") from error
                expected[i, j, k] = probs.sum() + mean
    total = named_count + anonymous_count
    if (total < 0).any():
        raise ValueError("Conditional total count overflow")
    return xr.Dataset({"named_count": (("chain", "draw", "did"), named_count),
                       "anonymous_count": (("chain", "draw", "did"), anonymous_count),
                       "total_count": (("chain", "draw", "did"), total),
                       "expected_total": (("chain", "draw", "did"), expected)},
                      coords={"chain": post.chain, "draw": post.draw,
                              "did": np.array(L["coords"]["edition"])[targets]},
                      attrs={"quadrature_order": quadrature_order, "seed": int(seed),
                             "endpoint_clips": endpoint_clips, "edition_effects": "conditional fitted effects"})
