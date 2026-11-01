"""Shared v2 joint finish-time/hidden-DNF model from notebook 2.3.

The only validation extension is an explicit set of known nonreporting races:
their finishers inform pace, but they supply neither DNF nor silence likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from utils.finish_time_model import GENDERS

FINISH_SCALARS = ["pace_50k_mean", "pace_50k_gap", "distance_exponent_mean", "distance_exponent_gap", "tau", "omega", "sigma"]
DNF_SCALARS = ["dnf_mean", "dnf_distance", "dnf_gender_gap", "dnf_age_linear", "dnf_age_quadratic", "tau_dnf", "omega_dnf",
               "rho_course", "rho_runner", "dnf_runner_pace_quadratic", "tau_edition"]


@dataclass
class DNFInputs:
    """Ordered model arrays and their visible-frame identities; no hidden truth.

    ``edition_ids`` and ``editions`` describe races with finishers; reporting
    starts can additionally contain zero-finisher races. Gender is 0=M, 1=F.
    Arrays are used without dtype conversion or coordinate reordering. Posted
    row genders may vary for one runner: ``runner_gender`` is the original
    first-observed label used only for centering, not a replacement for rows.
    """

    courses: np.ndarray
    runners: pd.Index
    course_x: np.ndarray
    runner_gender: np.ndarray
    log_pace: np.ndarray
    f_course: np.ndarray
    f_runner: np.ndarray
    f_gender: np.ndarray
    f_x: np.ndarray
    f_age: np.ndarray
    f_age2: np.ndarray
    edition_ids: np.ndarray
    f_edition: np.ndarray
    s_course: np.ndarray
    s_runner: np.ndarray
    s_gender: np.ndarray
    s_age: np.ndarray
    dnf: np.ndarray
    starts: pd.DataFrame
    editions: pd.DataFrame
    unknown: np.ndarray
    known_nonreporting_ids: tuple[int, ...] = ()


def _gender_average(runner_gender):
    gender = np.asarray(runner_gender)
    if gender.ndim != 1 or not np.issubdtype(gender.dtype, np.integer) or not np.isin(gender, [0, 1]).all():
        raise ValueError("runner_gender must contain integer gender indices 0 and 1")
    counts = np.bincount(gender, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both gender groups need at least one runner for zero-sum normalization")
    return np.eye(2)[gender] / counts


def _validate_inputs(data):
    for name in ("courses", "runners", "edition_ids"):
        labels = pd.Index(getattr(data, name))
        if labels.empty or labels.hasnans or not labels.is_unique:
            raise ValueError(f"{name} must have nonempty, unique, present coordinates")
    if not isinstance(data.starts, pd.DataFrame) or not isinstance(data.editions, pd.DataFrame):
        raise ValueError("starts and editions must be data frames")
    if not data.editions.index.equals(pd.Index(data.edition_ids)):
        raise ValueError("editions must be indexed in edition_ids order")
    if len(data.starts) == 0:
        raise ValueError("At least one reporting training start is required")
    nf, ns = len(data.log_pace), len(data.starts)
    expected = {"course_x": len(data.courses), "runner_gender": len(data.runners),
                "unknown": len(data.edition_ids), "log_pace": nf}
    expected.update({name: nf for name in ("f_course", "f_runner", "f_gender", "f_x", "f_age", "f_age2", "f_edition")})
    expected.update({name: ns for name in ("s_course", "s_runner", "s_gender", "s_age", "dnf")})
    for name, size in expected.items():
        value = np.asarray(getattr(data, name))
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite vector of length {size}")
    for name, size in (("f_course", len(data.courses)), ("s_course", len(data.courses)),
                       ("f_runner", len(data.runners)), ("s_runner", len(data.runners)),
                       ("f_edition", len(data.edition_ids)), ("f_gender", 2), ("s_gender", 2)):
        value = np.asarray(getattr(data, name))
        if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0) or np.any(value >= size):
            raise ValueError(f"Invalid {name} indices")
    _gender_average(data.runner_gender)
    if not np.isin(data.dnf, [0, 1]).all():
        raise ValueError("dnf must be binary")
    if np.asarray(data.unknown).dtype != np.dtype(bool):
        raise ValueError("unknown must be a boolean vector")
    if data.starts["event_distance_id"].isna().any():
        raise ValueError("Reporting start race IDs must be present")
    known = data.known_nonreporting_ids
    if len(set(known)) != len(known) or not np.isin(known, data.edition_ids[data.unknown]).all():
        raise ValueError("known_nonreporting_ids must be unique silent edition IDs")
    if data.starts["event_distance_id"].isin(known).any():
        raise ValueError("Known nonreporting races cannot contribute reporting starts")


def build_unobserved_model(data: DNFInputs, *, unknown: str = "mixture") -> pm.Model:
    """Build unchanged 2.3 v2, with optional known-nonreporter exclusion.

    ``mixture`` retains uncertainty about genuine silence; ``naive`` treats
    genuine silence as reporting zero DNFs. Neither includes known nonreporters.
    The original finite probability clipping applies only to genuine silence.
    """
    if unknown not in ("mixture", "naive"):
        raise ValueError("unknown must be 'mixture' or 'naive'")
    _validate_inputs(data)
    courses, runners, course_x = data.courses, data.runners, data.course_x
    runner_gender, log_pace = data.runner_gender, data.log_pace
    f_course, f_runner, f_gender = data.f_course, data.f_runner, data.f_gender
    f_x, f_age, f_age2 = data.f_x, data.f_age, data.f_age2
    s_course, s_runner, s_gender, s_age, dnf = data.s_course, data.s_runner, data.s_gender, data.s_age, data.dnf
    _avg = _gender_average(runner_gender)
    _f_half = np.array([-0.5, 0.5])[f_gender]
    _s_half = np.array([-0.5, 0.5])[s_gender]
    silent = data.unknown & ~np.isin(data.edition_ids, data.known_nonreporting_ids)
    _u_ids = np.flatnonzero(silent)
    _u_rows = np.flatnonzero(silent[data.f_edition])
    _n_unknown = len(_u_ids)
    _u_edition = np.searchsorted(_u_ids, data.f_edition[_u_rows])
    if _n_unknown:
        probabilities = []
        for column in ("p_reports_prior", "p_lists_dns"):
            if column not in data.editions:
                raise ValueError(f"Missing {column} for genuinely silent races")
            values = data.editions[column].to_numpy()[_u_ids]
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                raise ValueError(f"Missing or invalid {column} for genuinely silent races")
            probabilities.append(np.clip(values, 1e-9, 1 - 1e-9))
        unknown_r, unknown_d = probabilities
        unknown_log_r_nodns = np.log(unknown_r) + np.log1p(-unknown_d)
        unknown_log_not_r = np.log1p(-unknown_r)

    def _zero_sum(raw):
        return raw - pm.math.dot(raw, _avg)[runner_gender]

    all_editions = np.union1d(data.edition_ids, data.starts["event_distance_id"].to_numpy())
    s_edition = np.searchsorted(all_editions, data.starts["event_distance_id"].to_numpy())
    f_edition_all = np.searchsorted(all_editions, data.edition_ids)[data.f_edition]
    with pm.Model(coords={"gender": list(GENDERS), "course": courses, "runner": runners,
                          "edition": [str(e) for e in all_editions]}) as model:
        _pace = pm.Normal("pace_50k_mean", np.log(8.0), 0.3)
        _pace_gap = pm.Normal("pace_50k_gap", 0.0, 0.2)
        _exp = pm.Normal("distance_exponent_mean", 0.1, 0.1)
        _exp_gap = pm.Normal("distance_exponent_gap", 0.0, 0.1)
        _a1 = pm.Normal("age_linear", 0.0, 0.1, dims="gender")
        _a2 = pm.Normal("age_quadratic", 0.0, 0.05, dims="gender")
        _tau = pm.HalfNormal("tau", 0.3)
        _omega = pm.HalfNormal("omega", 0.3)
        _sigma = pm.HalfNormal("sigma", 0.3, dims="gender")
        _course_mean = pm.Normal("course_mean", _pace + _exp * course_x, _tau, dims="course")
        _u = pm.Deterministic("course_effect", _course_mean - (_pace + _exp * course_x), dims="course")
        _z1 = _zero_sum(pm.Normal("runner_raw", 0.0, 1.0, dims="runner"))
        _v = pm.Deterministic("runner_effect", _omega * _z1, dims="runner")
        _mu = (_course_mean[f_course] + _f_half * (_pace_gap + _exp_gap * f_x)
               + _a1[f_gender] * f_age + _a2[f_gender] * f_age2 + _v[f_runner])
        pm.Normal("log_pace", _mu, _sigma[f_gender], observed=log_pace)

        _d0 = pm.Normal("dnf_mean", -1.4, 1.0)
        _dd = pm.Normal("dnf_distance", 0.0, 1.0)
        _dg = pm.Normal("dnf_gender_gap", 0.0, 0.5)
        _da1 = pm.Normal("dnf_age_linear", 0.0, 0.5)
        _da2 = pm.Normal("dnf_age_quadratic", 0.0, 0.25)
        _tau_d = pm.HalfNormal("tau_dnf", 1.0)
        _rho_c = pm.Deterministic("rho_course", 2 * pm.Beta("rho_course_01", 2, 2) - 1)
        _level = pm.Normal("course_dnf_level", _d0 + _dd * course_x + _tau_d * _rho_c * _u / _tau,
                           _tau_d * pt.sqrt(1 - _rho_c**2), dims="course")
        pm.Deterministic("course_dnf_effect", _level - (_d0 + _dd * course_x), dims="course")
        _omega_d = pm.HalfNormal("omega_dnf", 1.0)
        _z2 = _zero_sum(pm.Normal("runner_dnf_raw", 0.0, 1.0, dims="runner"))
        _rho_r = pm.Deterministic("rho_runner", 2 * pm.Beta("rho_runner_01", 2, 2) - 1)
        _q = pm.Normal("dnf_runner_pace_quadratic", 0.0, 0.5)
        _w = pm.Deterministic("runner_dnf_effect", _omega_d * (_rho_r * _z1 + pt.sqrt(1 - _rho_r**2) * _z2) + _q * (_z1**2 - 1), dims="runner")
        _tau_e = pm.HalfNormal("tau_edition", 0.5)
        _ed = pm.Deterministic("edition_dnf_effect", _tau_e * pm.Normal("edition_raw", 0.0, 1.0, dims="edition"), dims="edition")
        _eta = _level[s_course] + _s_half * _dg + _da1 * s_age + _da2 * s_age**2 + _w[s_runner] + _ed[s_edition]
        pm.Bernoulli("dnf", logit_p=_eta, observed=dnf)

        if _n_unknown:
            _rows = _u_rows
            _eta_u = (_level[f_course[_rows]] + np.array([-0.5, 0.5])[f_gender[_rows]] * _dg
                      + _da1 * f_age[_rows] + _da2 * f_age2[_rows] + _w[f_runner[_rows]] + _ed[f_edition_all[_rows]])
            _log_q = pt.inc_subtensor(pt.zeros(_n_unknown)[_u_edition], -pt.softplus(_eta_u))
            if unknown == "mixture":
                pm.Potential("unreported", pm.math.logaddexp(unknown_log_r_nodns + _log_q, unknown_log_not_r).sum())
            else:
                pm.Potential("unreported", _log_q.sum())
    return model


def derive_runner_effects(posterior: xr.Dataset, runner_gender) -> xr.Dataset:
    """Recover the original float32 effects, loading raw arrays one chain at a time.

    Gender centering is applied before the original correlated/quadratic DNF
    transformation. Raw variables are removed; input labels and scalars remain.
    """
    _avg = _gender_average(runner_gender)
    if len(runner_gender) != posterior.sizes["runner"]:
        raise ValueError("runner_gender length must match posterior runner coordinates")

    def _zero_sum(raw):
        return raw - (raw @ _avg)[..., runner_gender]

    def centered(name):
        return np.stack([
            _zero_sum(posterior[name].isel(chain=c).transpose("draw", "runner").values).astype(np.float32)
            for c in range(posterior.sizes["chain"])
        ])

    _z1 = centered("runner_raw")
    _sc = lambda v: posterior[v].transpose("chain", "draw").values[..., None].astype(np.float32)
    _out = {"runner_effect": _sc("omega") * _z1}
    if "runner_dnf_raw" in posterior:
        _z2 = centered("runner_dnf_raw")
        if "rho_runner" in posterior:
            _r = _sc("rho_runner")
            _w = _sc("omega_dnf") * (_r * _z1 + np.sqrt(1 - _r**2) * _z2)
            if "dnf_runner_pace_quadratic" in posterior:
                _w = _w + _sc("dnf_runner_pace_quadratic") * (_z1**2 - 1)
        else:
            _w = _sc("omega_dnf") * _z2
        _out["runner_dnf_effect"] = _w
    result = posterior.drop_vars([v for v in ("runner_raw", "runner_dnf_raw") if v in posterior])
    for name, values in _out.items():
        result[name] = xr.DataArray(values, dims=("chain", "draw", "runner"),
                                    coords={d: posterior[d] for d in ("chain", "draw", "runner")})
    return result
