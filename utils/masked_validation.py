"""Individual-race masking, bounded conditional count prediction and evaluation.

Truth is evaluator-only. Fit structures use a whitelist of visible row fields; neither
raw race totals nor production reporting exports belong in this module's inputs.
The target is reconstruction given known suppression, not reporting classification.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit

from utils import finish_time_model as ftm

if TYPE_CHECKING:
    from utils.dnf_model import DNFInputs

ALLOCATION_VERSION = "individual-race-v1"
PREDICTION_VERSION = "fresh-effect-start-weighted-v1"
VALIDATION_VERSION = "masked-validation-v1"
DISTANCE_EDGES = [42.0, 60.0, 90.0, 120.0, 165.0]
VISIBLE_FIELDS = ["event_distance_id", "participant_id", "name", "year", "distance_km",
                  "finished", "reports_dnfs", "gender", "age", "time_ms"]
REPORTING_FIELDS = ["did", "series_id", "year", "distance_km", "log_distance", "kind",
                    "known_reporting", "n_finishers", "n_dnf", "n_dns", "n_starters"]


def _require(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


def _ids(values, *, name="IDs", nonempty=True):
    values = list(values)
    if (nonempty and not values) or pd.Index(values).has_duplicates:
        raise ValueError(f"{name} must be nonempty and unique")
    try:
        array = np.asarray(values, dtype=np.int64)
        if (any(isinstance(v, (bool, np.bool_)) or (str(v) != str(int(i)) and v != i) for v, i in zip(values, array))
                or (array < 0).any() or pd.Index(array).has_duplicates):
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"{name} must be nonnegative integer race IDs") from None
    return array


def _courses(course_names):
    index = pd.Index(course_names)
    if not len(index) or index.has_duplicates or index.isna().any():
        raise ValueError("course_names must be nonempty, unique and nonmissing")
    return np.sort(index.to_numpy())


def _race_rows(frame):
    _require(frame, ["event_distance_id", "name", "year", "distance_km", "finished", "reports_dnfs"])
    if frame[["event_distance_id", "name", "year", "distance_km", "finished", "reports_dnfs"]].isna().any().any():
        raise ValueError("Missing race coordinates or flags")
    for flag in ("finished", "reports_dnfs"):
        if not pd.api.types.is_bool_dtype(frame[flag].dtype):
            raise ValueError(f"{flag} must be a boolean column")
    if not np.isfinite(frame[["year", "distance_km"]].to_numpy(dtype=float)).all() or (frame.distance_km <= 0).any():
        raise ValueError("Invalid year/distance coordinates")
    grouped = frame.groupby("event_distance_id", sort=True)
    for col in ["name", "year", "distance_km", "reports_dnfs"]:
        if (grouped[col].nunique() != 1).any():
            raise ValueError(f"Inconsistent {col} within race")
    result = grouped.agg(course_id=("name", "first"), year=("year", "first"),
                         distance_km=("distance_km", "first"), reports_dnfs=("reports_dnfs", "first"),
                         n_finishers=("finished", "sum")).reset_index(names="did")
    _ids(result.did, nonempty=False)
    return result


def make_race_manifest(processed, course_names, *, fraction=0.2, seed):
    """Freeze an observable-only stratified mask; allocation diagnostics live in attrs.

    Rows are reporting occurrences since 2010 with F>0 in [42,165] km. Eligibility
    additionally requires three such occurrences per course. Tercile edges and quota
    diagnostics are JSON columns as well as attrs, so parquet retains the protocol.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must lie strictly between zero and one")
    courses = _courses(course_names)
    races = _race_rows(processed[processed["name"].isin(courses)])
    races = races[races.reports_dnfs & (races.year >= 2010) & (races.n_finishers > 0)
                  & races.distance_km.between(42, 165)].copy().reset_index(drop=True)
    races["eligible"] = races.groupby("course_id").did.transform("size") >= 3
    races["distance_bin"] = pd.cut(races.distance_km, DISTANCE_EDGES, include_lowest=True,
                                    labels=["42-60", "60-90", "90-120", "120-165"]).astype(str)
    edges = {}
    races["finisher_bin"] = -1
    for distance in ["42-60", "60-90", "90-120", "120-165"]:
        pool = races.loc[races.eligible & (races.distance_bin == distance), "n_finishers"]
        if pool.empty:
            continue
        cut = np.unique(np.quantile(pool, [0, 1 / 3, 2 / 3, 1])).tolist()
        edges[distance] = cut
        # Duplicate edges collapse. A constant-size bin is a single stratum.
        rows = races.distance_bin == distance
        races.loc[rows, "finisher_bin"] = np.searchsorted(cut[1:-1], races.loc[rows, "n_finishers"], side="left")
    races["stratum"] = races.distance_bin + ":" + races.finisher_bin.astype(str)
    target = round(fraction * int(races.eligible.sum()))
    if target < 1:
        raise ValueError("No eligible holdout is possible at this fraction")
    sizes = races.loc[races.eligible].groupby("stratum", sort=True).size()
    ideal = target * sizes / sizes.sum()
    quotas = np.floor(ideal).astype(int)
    order = sorted(sizes.index, key=lambda s: (-(ideal[s] - quotas[s]), s))
    for stratum in order[:target - int(quotas.sum())]:
        quotas[stratum] += 1
    rng = np.random.default_rng(seed)
    candidates = rng.permutation(races.index[races.eligible].to_numpy())
    capacity = (races.loc[races.eligible].groupby("course_id").size() - 2).to_dict()
    selected, counts = set(), dict.fromkeys(sizes.index, 0)

    def accept(i):
        course, stratum = races.loc[i, ["course_id", "stratum"]]
        if capacity[course] <= 0:
            return False
        selected.add(int(i))
        capacity[course] -= 1
        counts[stratum] += 1
        return True

    for i in candidates:
        if counts[races.at[i, "stratum"]] < quotas[races.at[i, "stratum"]]:
            accept(i)
    for i in candidates:
        if len(selected) >= target:
            break
        if i not in selected:
            accept(i)
    if not selected:
        raise ValueError("No feasible holdout retaining two races per course")
    races["selected"] = races.index.isin(selected)
    races["seed"] = int(seed)
    races["allocation_version"] = ALLOCATION_VERSION
    races["finisher_edges_json"] = _json(edges)
    deviations = {s: {"ideal": float(ideal[s]), "quota": int(quotas[s]), "actual": counts[s],
                      "deviation": int(counts[s] - quotas[s])} for s in sizes.index}
    races["allocation_json"] = _json(deviations)
    races["actual_fraction"] = len(selected) / int(races.eligible.sum())
    races["requested_fraction"] = fraction
    races.attrs.update(finisher_edges=edges, allocation=deviations, target_count=target,
                       actual_count=len(selected), actual_fraction=float(races.actual_fraction.iloc[0]))
    return races.drop(columns="reports_dnfs").sort_values("did").reset_index(drop=True)


def mask_results(processed, heldout_ids):
    """Copy the frame, remove masked DNFs, and label retained masked finishers silent."""
    ids = _ids(heldout_ids)
    _require(processed, VISIBLE_FIELDS)
    if not set(ids) <= set(processed.event_distance_id):
        raise ValueError("Unknown held-out IDs")
    masked = processed.event_distance_id.isin(ids)
    result = processed.loc[~masked | processed.finished].copy()
    result.loc[result.event_distance_id.isin(ids), "reports_dnfs"] = False
    # Arbitrary race-total columns are not safe visible metadata. The fitting helper
    # applies a whitelist too, so pass-through fields can never leak into fit objects.
    result = result.drop(columns=[c for c in ("n_dnf", "n_dns", "n_starters", "n_entrants") if c in result])
    return result.reset_index(drop=True)


def heldout_truth(processed, heldout_ids):
    """Evaluator only: processed counts, never raw race totals."""
    ids = np.sort(_ids(heldout_ids))
    if not set(ids) <= set(processed.event_distance_id):
        raise ValueError("Unknown truth IDs")
    rows = processed[processed.event_distance_id.isin(ids)]
    truth = rows.groupby("event_distance_id").finished.agg(n_finishers="sum", total="size").reindex(ids)
    truth["n_dnf"] = truth.pop("total") - truth.n_finishers
    return truth.astype(int).reset_index(names="did")[["did", "n_dnf", "n_finishers"]]


def training_counts(visible, course_names):
    """Baseline-only counts: every year, reporting races with F>0 at fixed courses."""
    rows = visible[visible.name.isin(_courses(course_names))]
    races = _race_rows(rows)
    races = races[races.reports_dnfs & (races.n_finishers > 0)].copy()
    totals = rows.groupby("event_distance_id").size()
    races["n_dnf"] = races.did.map(totals) - races.n_finishers
    return races.drop(columns="reports_dnfs").reset_index(drop=True)


def prepare_dnf_inputs(visible, reporting_probabilities, course_names, *, known_nonreporting_ids=()) -> DNFInputs:
    """Rebuild all indices from visible finishers and remaining reporting starts.

    Only genuinely silent races require upstream probabilities. ``starts`` and
    ``editions`` contain whitelist fields, not hidden/source race totals. Runner
    coordinates preserve first appearance (finishers, then reporting starts).
    """
    from utils.dnf_model import DNFInputs

    _require(visible, VISIBLE_FIELDS)
    courses = _courses(course_names)
    frame = visible.loc[visible.name.isin(courses), VISIBLE_FIELDS].copy().reset_index(drop=True)
    frame.attrs = {}  # Source/evaluator metadata is not part of a fitting structure.
    races = _race_rows(frame)
    if (not np.isfinite(frame.age.to_numpy(dtype=float)).all()
            or frame[["participant_id", "gender", "age"]].isna().any().any() or not frame.gender.isin(["M", "F"]).all()):
        raise ValueError("Invalid runner coordinates or gender")
    if frame.duplicated(["event_distance_id", "participant_id"]).any():
        raise ValueError("Duplicate runner/race rows")
    if (frame.groupby("name").distance_km.nunique() > 1).any():
        raise ValueError("Inconsistent course distance")
    finishers = frame[frame.finished].reset_index(drop=True)
    starts = frame[frame.reports_dnfs].reset_index(drop=True)
    if set(finishers.name) != set(courses):
        raise ValueError("Missing finisher courses in fixed universe")
    if starts.empty:
        raise ValueError("No remaining reporting starts for runner reference population")
    known = tuple(sorted(_ids(known_nonreporting_ids, nonempty=False).tolist()))
    if not set(known) <= set(finishers.event_distance_id):
        raise ValueError("Known nonreporters must have retained finishers")
    known_rows = frame.event_distance_id.isin(known)
    if frame.loc[known_rows, "reports_dnfs"].any() or (~frame.loc[known_rows, "finished"]).any():
        raise ValueError("Known nonreporters must be masked before preparing inputs")
    runners = pd.Index(pd.concat([finishers.participant_id, starts.participant_id]).unique())
    runner_gender = (pd.concat([finishers, starts]).drop_duplicates("participant_id")
                     .set_index("participant_id").gender.reindex(runners).eq("F").to_numpy(dtype=int))
    if (np.bincount(runner_gender, minlength=2) == 0).any():
        raise ValueError("Both gender groups need runners for zero-sum normalization")
    z, log_pace = ftm.design(finishers)
    if not np.isfinite(z).all() or not np.isfinite(log_pace).all():
        raise ValueError("Nonfinite finish design")
    edition_ids, f_edition = np.unique(finishers.event_distance_id, return_inverse=True)
    editions = races.set_index("did").reindex(edition_ids).drop(columns="distance_km")
    editions["n_dnf"] = (starts.groupby("event_distance_id").finished.apply(lambda s: int((~s).sum()))
                           .reindex(edition_ids).fillna(0).astype(int))
    editions["course_idx"] = courses.searchsorted(editions.course_id)
    _require(reporting_probabilities, ["did", "p_reports_prior", "p_lists_dns"])
    _ids(reporting_probabilities.did, name="Probability IDs", nonempty=False)
    probabilities = reporting_probabilities.set_index("did")[["p_reports_prior", "p_lists_dns"]].astype(float)
    editions = editions.join(probabilities)
    genuine = ~editions.reports_dnfs & ~editions.index.isin(known)
    values = editions.loc[genuine, ["p_reports_prior", "p_lists_dns"]].to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Missing or invalid probabilities for genuinely silent races")
    # Known-suppressed and reporting targets cannot carry upstream hidden metadata.
    editions.loc[~genuine, ["p_reports_prior", "p_lists_dns"]] = np.nan
    course_x = np.log(frame.groupby("name").distance_km.first().reindex(courses).to_numpy() / ftm.REFERENCE_KM)
    return DNFInputs(courses=courses, runners=runners, course_x=course_x, runner_gender=runner_gender,
                     log_pace=log_pace, f_course=courses.searchsorted(finishers.name),
                     f_runner=runners.get_indexer(finishers.participant_id),
                     f_gender=finishers.gender.eq("F").to_numpy(dtype=int), f_x=z[:, 1], f_age=z[:, 2], f_age2=z[:, 3],
                     edition_ids=edition_ids, f_edition=f_edition,
                     s_course=courses.searchsorted(starts.name), s_runner=runners.get_indexer(starts.participant_id),
                     s_gender=starts.gender.eq("F").to_numpy(dtype=int),
                     s_age=((starts.age - ftm.AGE_CENTER) / ftm.AGE_SCALE).to_numpy(),
                     dnf=(~starts.finished).to_numpy(dtype=int), starts=starts, editions=editions,
                     unknown=(~editions.reports_dnfs).to_numpy(), known_nonreporting_ids=known)


def _count_frame(frame, *, observed):
    fields = ["did", "course_id", "distance_km", "n_finishers"] + (["n_dnf"] if observed else [])
    _require(frame, fields)
    _ids(frame.did)
    if frame[fields].isna().any().any():
        raise ValueError("Missing count data")
    numeric = frame[["distance_km", "n_finishers"] + (["n_dnf"] if observed else [])].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, :2] <= 0).any():
        raise ValueError("Counts require finite positive distance and F>0")
    for column in ["n_finishers"] + (["n_dnf"] if observed else []):
        v = frame[column].to_numpy()
        if (v < 0).any() or (v != np.floor(v)).any():
            raise ValueError("Counts must be nonnegative integers")
    return frame.sort_values("did").reset_index(drop=True)


def build_count_baseline(training_counts):
    """Course-shrunk NB baseline, with fixed DNF-per-finisher priors.

    exp(intercept) is a DNF/finisher ratio (Normal(-1.5,1) on its log).
    Slope Normal(0,1), course SD HalfNormal(1), and dispersion
    LogNormal(log(10),1) allow substantial heterogeneity and overdispersion.
    """
    import pymc as pm

    frame = _count_frame(training_counts, observed=True)
    courses = np.sort(frame.course_id.unique())
    # nutpie's streamed trace coordinate serializer does not accept ndarray
    # integer coordinates. Observation labels are strings; public predictions
    # still use explicit numeric did coordinates for evaluation alignment.
    with pm.Model(coords={"course": courses.tolist(), "did": [str(int(did)) for did in frame.did]}) as model:
        intercept = pm.Normal("intercept", -1.5, 1)
        slope = pm.Normal("slope", 0, 1)
        sd = pm.HalfNormal("course_sd", 1)
        raw = pm.Normal("course_raw", 0, 1, dims="course")
        effect = pm.Deterministic("course_effect", sd * raw, dims="course")
        dispersion = pm.LogNormal("dispersion", np.log(10), 1)
        mu = frame.n_finishers.to_numpy() * pm.math.exp(intercept + slope * np.log(frame.distance_km.to_numpy() / 50)
                                                        + effect[courses.searchsorted(frame.course_id)])
        pm.NegativeBinomial("count", mu=mu, alpha=dispersion, observed=frame.n_dnf.to_numpy(), dims="did")
    return model


def _posterior(posterior, draws):
    if not isinstance(posterior, xr.Dataset):
        raise TypeError("posterior must be an xarray Dataset")
    for dim in ("chain", "draw"):
        if dim not in posterior.dims or not posterior.sizes[dim] or posterior.get_index(dim).has_duplicates:
            raise ValueError(f"Missing or duplicate {dim} coordinates")
    if not isinstance(draws, (int, np.integer)) or draws < 1 or draws > posterior.sizes["draw"]:
        raise ValueError("Requested posterior draws exceed available draws or are not positive")
    return np.linspace(0, posterior.sizes["draw"] - 1, draws, dtype=int)


def _rng(seed, chain, draw, did, stream):
    # Hash coordinate labels, not loop positions: target order/batching cannot alter results.
    digest = hashlib.sha256(_json([int(seed), str(chain), str(draw), int(did), stream]).encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest, dtype="<u4")))


def _finite(values, name):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite {name}")
    return values


def _scalar(draw, name):
    value = _finite(draw[name].values, name)
    if value.size != 1:
        raise ValueError(f"{name} must be scalar per posterior draw")
    return float(value.item())


def _prediction_array(values, post, indices, ids, attrs):
    return xr.DataArray(values, dims=("chain", "draw", "did"),
                        coords={"chain": post.chain.values, "draw": post.draw.values[indices], "did": ids},
                        name="n_dnf", attrs={"prediction_version": PREDICTION_VERSION, **attrs})


def predict_count_baseline(posterior, targets, *, seed, posterior_draws_per_chain=500):
    """Return chain-preserving NB counts, selecting evenly spaced retained draws."""
    indices = _posterior(posterior, posterior_draws_per_chain)
    targets = _count_frame(targets, observed=False)
    if not set(targets.course_id) <= set(posterior.course.values):
        raise ValueError("Unknown target courses")
    out = np.empty((posterior.sizes["chain"], len(indices), len(targets)), dtype=np.int64)
    for c in range(len(out)):
        for j, d in enumerate(indices):
            draw = posterior.isel(chain=c, draw=int(d))
            alpha = _scalar(draw, "dispersion")
            if alpha <= 0:
                raise ValueError("dispersion must be positive")
            effects = _finite(draw.course_effect.sel(course=targets.course_id.to_numpy()).values, "course effects")
            log_mu = (np.log(targets.n_finishers.to_numpy()) + _scalar(draw, "intercept")
                      + _scalar(draw, "slope") * np.log(targets.distance_km.to_numpy() / 50) + effects)
            with np.errstate(over="ignore"):
                mu = _finite(np.exp(log_mu), "baseline mean")
            for k, did in enumerate(targets.did):
                rng = _rng(seed, posterior.chain.values[c], posterior.draw.values[d], did, "baseline-count")
                try:
                    out[c, j, k] = rng.negative_binomial(alpha, alpha / (alpha + mu[k]))
                except ValueError as exc:
                    raise ValueError("Baseline predictive count exceeds numerical range") from exc
    return _prediction_array(out, posterior, indices, targets.did.to_numpy(), {"model": "baseline", "seed": int(seed)})


def predict_hidden_counts(posterior, data, target_ids, *, seed, posterior_draws_per_chain=500, nodes=400, race_batch=16):
    """Predict known-suppressed counts with fresh race effects and start-weighted donors.

    Computation streams one posterior draw and at most 64 finishers x ``nodes``
    at a time. No (draw, runner) array is materialized. Donor effects are read as
    float32, matching the shared trace derivation. ``target_probability`` is a
    small (chain, draw, did) auxiliary coordinate for convergence diagnostics.
    Probabilities at floating-point endpoints are clipped to [eps,1-eps], with
    eps=float64 machine epsilon; the number of clipped target/draws is recorded.
    """
    indices = _posterior(posterior, posterior_draws_per_chain)
    ids = np.sort(_ids(target_ids))
    if not isinstance(nodes, (int, np.integer)) or nodes < 1 or not isinstance(race_batch, (int, np.integer)) or race_batch < 1:
        raise ValueError("nodes and race_batch must be positive integers")
    if not set(ids) <= set(data.known_nonreporting_ids):
        raise ValueError("Targets must be explicitly known nonreporters")
    if not len(data.s_runner):
        raise ValueError("No reporting starts for donor population")
    if list(posterior.runner.values) != list(data.runners) or list(posterior.course.values) != list(data.courses):
        raise ValueError("Posterior runner/course coordinate ordering does not match inputs")
    if (not np.issubdtype(np.asarray(data.s_runner).dtype, np.integer)
            or np.any(data.s_runner < 0) or np.any(data.s_runner >= len(data.runners))):
        raise ValueError("Invalid starter runner indices")
    rows = {int(did): np.flatnonzero(np.asarray(data.edition_ids)[data.f_edition] == did) for did in ids}
    if any(not len(v) for v in rows.values()):
        raise ValueError("Every target needs F>0")
    shape = (posterior.sizes["chain"], len(indices), len(ids))
    counts, probabilities = np.empty(shape, dtype=np.int64), np.empty(shape)
    clipped = 0
    eps = np.finfo(float).eps
    for c in range(shape[0]):
        for j, d in enumerate(indices):
            draw = posterior.isel(chain=c, draw=int(d))
            tau = _scalar(draw, "tau_edition")
            if tau < 0:
                raise ValueError("tau_edition must be nonnegative")
            gap, a1, a2 = (_scalar(draw, name) for name in ("dnf_gender_gap", "dnf_age_linear", "dnf_age_quadratic"))
            levels = _finite(draw.course_dnf_level.values, "course DNF levels")
            # Read only ONE runner vector at a time, never chain x draw x runner.
            # Reusing this float32 vector avoids a lazy/zarr gather per target.
            runner_effects = _finite(np.asarray(draw.runner_dnf_effect.values, dtype=np.float32), "runner effects")
            for start in range(0, len(ids), race_batch):
                for k in range(start, min(start + race_batch, len(ids))):
                    did, r = int(ids[k]), rows[int(ids[k])]
                    chain, draw_id = posterior.chain.values[c], posterior.draw.values[d]
                    donor_rng = _rng(seed, chain, draw_id, did, "donors")
                    donor_indices = data.s_runner[donor_rng.integers(0, len(data.s_runner), nodes)]
                    donor = runner_effects[donor_indices]
                    edition = _rng(seed, chain, draw_id, did, "fresh-edition").normal(0, tau)
                    total = 0.0
                    for first in range(0, len(r), 64):
                        chunk = r[first:first + 64]
                        base = (levels[data.f_course[chunk]] + (data.f_gender[chunk] - 0.5) * gap
                                + a1 * data.f_age[chunk] + a2 * data.f_age2[chunk] + edition)
                        total += float(expit(base[:, None] + donor[None, :]).sum())
                    p = total / (len(r) * nodes)
                    if not np.isfinite(p) or p < 0 or p > 1:
                        raise ValueError("Invalid integrated DNF probability")
                    adjusted = float(np.clip(p, eps, 1 - eps))
                    clipped += int(adjusted != p)
                    probabilities[c, j, k] = adjusted
                    try:
                        counts[c, j, k] = _rng(seed, chain, draw_id, did, "hidden-count").negative_binomial(len(r), 1 - adjusted)
                    except ValueError as exc:
                        raise ValueError("Hidden predictive count exceeds numerical range after endpoint clipping") from exc
    result = _prediction_array(counts, posterior, indices, ids,
                               {"model": "joint", "seed": int(seed), "nodes": int(nodes), "race_batch": int(race_batch),
                                "endpoint_clip_count": clipped, "probability_epsilon": eps,
                                "max_finisher_batch": 64, "donor_population": "remaining_reporting_starts"})
    return result.assign_coords(target_probability=(("chain", "draw", "did"), probabilities))


def integrated_target_moments(posterior, data, target_ids, *, seed, posterior_draws_per_chain=500,
                              nodes=64, edition_nodes=8, donor_scope="all"):
    """Approximate fresh-effect-integrated means ONLY for posterior mixing diagnostics.

    Returns ``mean_probability`` = E_epsilon[p(epsilon)] and ``mean_count`` =
    E_epsilon[F*p(epsilon)/(1-p(epsilon))], retaining chain/draw/did. Each
    p(epsilon) averages over retained target finishers and a seeded sample of
    reporting *starts*, so repeated runners retain their start-frequency weight.
    With donor_scope="course", only starts on the target's course are eligible;
    the default "all" uses all remaining reporting starts. Donor indices are fixed
    per target across all chains/draws. Fresh
    Normal(0, tau_edition) effects use Gauss-Hermite quadrature, never fitted
    target effects. This common-donor/quadrature approximation removes predictive
    simulation noise for mean-mixing diagnostics; it is NOT a scored prediction.

    Streams one float32 runner vector per draw and at most 64 finishers x nodes.
    As in production prediction, endpoint probabilities are clipped to [eps,1-eps]
    before the count transform, with clipped quadrature evaluations recorded.
    """
    if donor_scope not in ("all", "course"):
        raise ValueError("donor_scope must be 'all' or 'course'")
    indices = _posterior(posterior, posterior_draws_per_chain)
    ids = np.sort(_ids(target_ids))
    for name, size in (("nodes", nodes), ("edition_nodes", edition_nodes)):
        if not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError(f"{name} must be a positive integer")
    for dim in ("chain", "draw", "runner", "course"):
        if (dim not in posterior.coords or posterior.coords[dim].dims != (dim,)
                or posterior.get_index(dim).has_duplicates or posterior.get_index(dim).isna().any()):
            raise ValueError(f"Missing, duplicate or invalid {dim} coordinates")
    if not set(ids) <= set(data.known_nonreporting_ids):
        raise ValueError("Targets must be explicitly known nonreporters")
    if not len(data.s_runner):
        raise ValueError("No reporting starts for donor population")
    if list(posterior.runner.values) != list(data.runners) or list(posterior.course.values) != list(data.courses):
        raise ValueError("Posterior runner/course coordinate ordering does not match inputs")
    if (not np.issubdtype(np.asarray(data.s_runner).dtype, np.integer)
            or np.any(data.s_runner < 0) or np.any(data.s_runner >= len(data.runners))):
        raise ValueError("Invalid starter runner indices")
    rows = {int(did): np.flatnonzero(np.asarray(data.edition_ids)[data.f_edition] == did) for did in ids}
    if any(not len(v) for v in rows.values()):
        raise ValueError("Every target needs F>0")
    pools = {int(did): data.s_runner for did in ids}
    if donor_scope == "course":
        s_course = np.asarray(data.s_course)
        f_course = np.asarray(data.f_course)
        if (s_course.shape != np.asarray(data.s_runner).shape or s_course.ndim != 1
                or not np.issubdtype(s_course.dtype, np.integer)
                or np.any(s_course < 0) or np.any(s_course >= len(data.courses))):
            raise ValueError("Invalid starter course indices")
        if (f_course.shape != np.asarray(data.f_edition).shape or f_course.ndim != 1
                or not np.issubdtype(f_course.dtype, np.integer)
                or np.any(f_course < 0) or np.any(f_course >= len(data.courses))):
            raise ValueError("Invalid finisher course indices")
        for did, r in rows.items():
            courses = np.unique(f_course[r])
            if len(courses) != 1:
                raise ValueError("Every target must have exactly one course")
            pools[did] = np.asarray(data.s_runner)[s_course == courses[0]]
            if not len(pools[did]):
                raise ValueError(f"No reporting starts for target {did} course")
    donors = {int(did): pools[int(did)][_rng(seed, "common", "common", did, "integrated-donors")
                                       .integers(0, len(pools[int(did)]), nodes)] for did in ids}
    abscissae, weights = np.polynomial.hermite.hermgauss(edition_nodes)
    abscissae, weights = np.sqrt(2) * abscissae, weights / np.sqrt(np.pi)
    shape = (posterior.sizes["chain"], len(indices), len(ids))
    probabilities, counts = np.empty(shape), np.empty(shape)
    eps, clipped = np.finfo(float).eps, 0
    for c in range(shape[0]):
        for j, d in enumerate(indices):
            draw = posterior.isel(chain=c, draw=int(d))
            tau = _scalar(draw, "tau_edition")
            if tau < 0:
                raise ValueError("tau_edition must be nonnegative")
            gap, a1, a2 = (_scalar(draw, name) for name in ("dnf_gender_gap", "dnf_age_linear", "dnf_age_quadratic"))
            levels = _finite(draw.course_dnf_level.values, "course DNF levels")
            runner_effects = _finite(np.asarray(draw.runner_dnf_effect.values, dtype=np.float32), "runner effects")
            for k, did in enumerate(ids):
                r, donor = rows[int(did)], runner_effects[donors[int(did)]]
                mean_p, mean_count = 0.0, 0.0
                for z, weight in zip(abscissae, weights):
                    total = 0.0
                    for first in range(0, len(r), 64):
                        chunk = r[first:first + 64]
                        base = (levels[data.f_course[chunk]] + (data.f_gender[chunk] - 0.5) * gap
                                + a1 * data.f_age[chunk] + a2 * data.f_age2[chunk] + tau * z)
                        total += float(expit(base[:, None] + donor[None, :]).sum())
                    p = total / (len(r) * nodes)
                    if not np.isfinite(p) or p < 0 or p > 1:
                        raise ValueError("Invalid integrated DNF probability")
                    adjusted = float(np.clip(p, eps, 1 - eps))
                    clipped += int(adjusted != p)
                    mean_p += weight * adjusted
                    # Average conditional means, NOT the odds of the average p.
                    mean_count += weight * len(r) * adjusted / (1 - adjusted)
                probabilities[c, j, k], counts[c, j, k] = mean_p, mean_count
    return xr.Dataset({"mean_probability": (("chain", "draw", "did"), probabilities),
                       "mean_count": (("chain", "draw", "did"), counts)},
                      coords={"chain": posterior.chain.values, "draw": posterior.draw.values[indices], "did": ids},
                      attrs={"purpose": "mixing diagnostics ONLY; not scored predictions",
                             "approximation": "Gauss-Hermite fresh race effects with common seeded training-start donors",
                             "seed": int(seed), "nodes": int(nodes), "edition_nodes": int(edition_nodes),
                             "donor_population": "remaining_reporting_starts", "donor_scope": donor_scope,
                             "common_donors_across_chain_draw": True,
                             "max_finisher_batch": 64, "endpoint_clip_count": clipped, "probability_epsilon": eps})


def target_probability_summary(predictions):
    """The integrated probabilities by chain/draw/target, ready for az.summary."""
    if "target_probability" not in predictions.coords:
        raise ValueError("Predictions have no target probability diagnostic")
    return predictions.coords["target_probability"].reset_coords(drop=True).rename("target_probability")


def _validate_predictions(predictions):
    if not isinstance(predictions, xr.DataArray) or set(predictions.dims) != {"chain", "draw", "did"}:
        raise ValueError("Predictions require chain, draw, did dimensions")
    for dim in predictions.dims:
        if predictions.sizes[dim] < 1 or predictions.get_index(dim).has_duplicates:
            raise ValueError(f"Empty or duplicate prediction {dim}")
    _ids(predictions.did.values)
    values = predictions.transpose("chain", "draw", "did").values
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Predictions must be finite nonnegative counts")
    if "target_probability" in predictions.coords:
        diagnostic = predictions.coords["target_probability"]
        p = diagnostic.values
        if (set(diagnostic.dims) != {"chain", "draw", "did"}
                or not np.isfinite(p).all() or (p <= 0).any() or (p >= 1).any()):
            raise ValueError("Invalid cached target probability diagnostic")
    return values


def score_predictions(predictions, truth, metadata):
    """Exact-ID-aligned central count intervals and WIS; signed error = mean-y."""
    values = _validate_predictions(predictions)
    _require(truth, ["did", "n_dnf"])
    _ids(truth.did)
    _require(metadata, ["did", "course_id"])
    _ids(metadata.did)
    ids = predictions.did.values
    if set(ids) != set(truth.did) or set(ids) != set(metadata.did):
        raise ValueError("Prediction, truth and metadata ID sets must match exactly")
    y = truth.set_index("did").loc[ids, "n_dnf"].to_numpy()
    if not np.isfinite(y).all() or (y < 0).any() or (y != np.floor(y)).any():
        raise ValueError("Truth counts must be finite nonnegative integers")
    out = metadata.set_index("did").loc[ids].copy()
    if out.course_id.isna().any():
        raise ValueError("Missing course IDs")
    sample = values.reshape(-1, len(ids))
    out["n_dnf"] = y
    out["mean"] = sample.mean(axis=0)
    out["median"] = np.quantile(sample, 0.5, axis=0, method="inverted_cdf")
    out["signed_mean_error"] = out["mean"] - y
    out["absolute_median_error"] = np.abs(out["median"] - y)
    wis = 0.5 * out.absolute_median_error.to_numpy()
    for level in (50, 80, 94):
        alpha = 1 - level / 100
        low, high = np.quantile(sample, [alpha / 2, 1 - alpha / 2], axis=0, method="inverted_cdf")
        width = high - low
        interval_score = width + 2 / alpha * (np.maximum(low - y, 0) + np.maximum(y - high, 0))
        out[f"lower_{level}"], out[f"upper_{level}"] = low, high
        out[f"coverage_{level}"] = (low <= y) & (y <= high)
        out[f"width_{level}"] = width
        out[f"interval_score_{level}"] = interval_score
        wis += alpha / 2 * interval_score
    out["wis"] = wis / 3.5
    if "model" in predictions.attrs:
        out["model"] = predictions.attrs["model"]
    return out.reset_index()


def sum_predictions(predictions):
    """Sum aligned targets within each posterior predictive draw, never endpoints."""
    _validate_predictions(predictions)
    # Remove probability coordinate: summing probabilities would not be meaningful.
    return predictions.reset_coords(drop=True).sum("did").rename("total_n_dnf")


def summarize_scores(scores, *, seed, bootstraps=2000):
    """Mean scores and course-clustered 94% evaluation (not posterior) intervals.

    Input is concatenated per-race scores with model='joint' (or 'v2') and
    model='baseline'. Positive paired WIS differences favor the baseline.
    Cluster resampling preserves every race in each sampled course, including
    repeated sampled clusters; equal-course results average within courses first.
    """
    _require(scores, ["did", "course_id", "model", "wis", "coverage_50", "coverage_80", "coverage_94"])
    if not isinstance(bootstraps, (int, np.integer)) or bootstraps < 1:
        raise ValueError("bootstraps must be positive")
    if scores.duplicated(["model", "did"]).any():
        raise ValueError("Duplicate model/race score rows")
    names = set(scores.model)
    joint = "joint" if "joint" in names else "v2"
    if names != {joint, "baseline"}:
        raise ValueError("Expected exactly joint/v2 and baseline scores")
    left, right = (scores[scores.model == name].set_index("did").sort_index() for name in (joint, "baseline"))
    if set(left.index) != set(right.index) or not left.course_id.equals(right.course_id):
        raise ValueError("Paired model targets/course IDs must match")
    for col in ("distance_bin", "finisher_bin", "stratum", "n_dnf", "n_finishers", "distance_km"):
        if col in left and not left[col].equals(right[col]):
            raise ValueError(f"Paired metadata mismatch: {col}")
    metrics = [c for c in ("wis", "coverage_50", "coverage_80", "coverage_94", "width_50", "width_80", "width_94",
                            "signed_mean_error", "absolute_median_error") if c in scores]
    if not np.isfinite(scores[metrics].to_numpy(dtype=float)).all() or scores.course_id.isna().any():
        raise ValueError("Nonfinite scores or missing course IDs")
    delta = left.wis - right.wis
    paired = pd.DataFrame({"did": left.index, "course_id": left.course_id.values, "wis_difference": delta.values})
    groups = list(paired.groupby("course_id", sort=True).indices.values())
    rng = np.random.default_rng(seed)
    cluster_n = np.array([len(g) for g in groups])
    delta_sums = np.array([delta.iloc[g].sum() for g in groups])
    cover_sums = {(m, level): np.array([table.iloc[g][f"coverage_{level}"].sum() for g in groups], dtype=float)
                  for m, table in ((joint, left), ("baseline", right)) for level in (50, 80, 94)}
    draws = {"race_weighted": np.empty(bootstraps), "equal_course": np.empty(bootstraps)}
    cover = {key: np.empty(bootstraps) for key in cover_sums}
    for b in range(bootstraps):
        sampled = rng.integers(len(groups), size=len(groups))
        n = cluster_n[sampled].sum()
        draws["race_weighted"][b] = delta_sums[sampled].sum() / n
        draws["equal_course"][b] = (delta_sums[sampled] / cluster_n[sampled]).mean()
        for key, sums in cover_sums.items():
            cover[key][b] = sums[sampled].sum() / n
    comparison = []
    for weighting, estimates in draws.items():
        point = delta.mean() if weighting == "race_weighted" else paired.groupby("course_id").wis_difference.mean().mean()
        low, high = np.quantile(estimates, [.03, .97])
        comparison.append(dict(weighting=weighting, mean_wis_difference=point, lower_94=low, upper_94=high,
                               interval_kind="course-clustered bootstrap evaluation interval", bootstraps=bootstraps))
    coverage = []
    for (model, level), estimates in cover.items():
        table = left if model == joint else right
        low, high = np.quantile(estimates, [.03, .97])
        coverage.append(dict(model=model, level=level, coverage=float(table[f"coverage_{level}"].mean()),
                             lower_94=low, upper_94=high, interval_kind="course-clustered bootstrap evaluation interval"))
    result = {"overall": scores.groupby("model", sort=True)[metrics].mean().reset_index(),
              "paired": paired, "comparison": pd.DataFrame(comparison), "coverage": pd.DataFrame(coverage),
              "equal_course": scores.groupby(["model", "course_id"])[metrics].mean().groupby("model").mean().reset_index()}
    for col in ("distance_bin", "finisher_bin", "stratum"):
        if col in scores:
            result[f"by_{col}"] = scores.groupby(["model", col], observed=True)[metrics].mean().reset_index()
            result[f"paired_by_{col}"] = (paired.assign(**{col: left[col].to_numpy()})
                                           .groupby(col, observed=True).wis_difference.agg(["mean", "size"]).reset_index())
    return result


def _json(value):
    def default(v):
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(f"Not JSON serializable: {type(v)}")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=default)


def _frame_digest(frame, columns=None):
    frame = frame.loc[:, columns] if columns is not None else frame
    frame = frame.reindex(sorted(frame.columns), axis=1)
    # Row order is irrelevant, but multiplicities, dtypes and column identities are not.
    hashed = np.sort(pd.util.hash_pandas_object(frame, index=False).to_numpy())
    digest = hashlib.sha256(_json([(str(c), str(frame[c].dtype)) for c in frame]).encode())
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def validation_fingerprint(visible, reporting_training, manifest, settings):
    """Training-only hash; full snapshot SHA belongs in separate provenance metadata.

    ``settings`` must include model/extraction/prediction versions, stage seeds,
    sampler settings and coordinate ordering (courses, runners). This is a protocol
    contract, checked explicitly instead of silently reusing an under-keyed cache.
    Extra settings are hashed too. Visible pass-through raw count metadata is ignored.
    """
    _require(visible, VISIBLE_FIELDS)
    required = {"model_version", "extraction_version", "prediction_version", "seeds", "sampler", "coordinates"}
    if not required <= set(settings):
        raise ValueError(f"Fingerprint settings missing {sorted(required - set(settings))}")
    _require(manifest, ["did", "selected"])
    _ids(manifest.did)
    _require(reporting_training, ["did"])
    if set(manifest.loc[manifest.selected, "did"]) & set(reporting_training.did):
        raise ValueError("Held-out IDs remain in reporting training")
    payload = {"validation_version": VALIDATION_VERSION, "visible": _frame_digest(visible, VISIBLE_FIELDS),
               "reporting_training": _frame_digest(reporting_training, [c for c in REPORTING_FIELDS if c in reporting_training]),
               "manifest": _frame_digest(manifest), "settings": settings}
    return hashlib.sha256(_json(payload).encode()).hexdigest()


def baseline_training_counts(visible, course_names):
    """Notebook-facing alias for training_counts (all years, including zero DNFs)."""
    return training_counts(visible, course_names)


def atomic_json(path, payload):
    """Notebook-facing alias for atomic_write_json."""
    return atomic_write_json(path, payload)


def save_trace_atomic(trace, path, metadata):
    """Save a configured complete trace; metadata needs chains and draws.

    An optional scalar_names list adds required variables (including small vector
    parameters) to the mandatory finite check of all scalar posterior variables.
    """
    names = metadata.get("scalar_names", [n for n, v in trace.posterior.data_vars.items()
                                        if set(v.dims) == {"chain", "draw"}])
    if not {"chains", "draws"} <= set(metadata):
        raise ValueError("Trace save metadata must declare expected chains and draws")
    return atomic_write_trace(path, trace, metadata=metadata, chains=metadata["chains"],
                              draws=metadata["draws"], scalar_names=names)


def load_cached_trace(path, metadata, *, chains, draws):
    """Exact-metadata cache load with explicit chain/draw completeness requirements."""
    if any(metadata.get(k, v) != v for k, v in (("chains", chains), ("draws", draws))):
        raise ValueError("Requested chains/draws disagree with metadata")
    names = metadata.get("scalar_names")
    if names is None and Path(path).exists():
        # Inspect variable shapes lazily; no giant runner-array materialization.
        import arviz as az
        trace = az.from_netcdf(path)
        try:
            names = [n for n, v in trace.posterior.data_vars.items() if set(v.dims) == {"chain", "draw"}]
        finally:
            trace.close()
    return load_trace_cache(path, metadata=metadata, chains=chains, draws=draws, scalar_names=names or [])


def source_snapshot_sha(path):
    """Full file SHA for provenance only, deliberately not a training cache key."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        writer(Path(temporary))
        with open(temporary, "rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_write_json(path, payload):
    """Write JSON via same-directory temporary file and atomic rename."""
    content = _json(payload)
    _atomic(path, lambda temporary: temporary.write_text(content + "\n"))


def atomic_write_frame(path, frame):
    """Atomically persist a parquet manifest, truth or score table."""
    _atomic(path, lambda temporary: frame.to_parquet(temporary, index=False))


def atomic_write_predictions(path, predictions, *, metadata):
    """Atomically save validated predictions and exact embedded cache metadata."""
    _validate_predictions(predictions)
    saved = predictions.copy(deep=False)
    saved.attrs = {**predictions.attrs, "validation_metadata": _json(metadata)}
    _atomic(path, saved.to_netcdf)


def load_predictions_cache(path, *, metadata):
    """None when absent; refuse metadata mismatch or corrupt count artifacts."""
    if not Path(path).exists():
        return None
    with xr.open_dataarray(path) as source:
        if source.attrs.get("validation_metadata") != _json(metadata):
            raise ValueError("Prediction cache metadata mismatch")
        result = source.load()
    _validate_predictions(result)
    return result


def validate_trace(trace, *, chains, draws, scalar_names):
    """Reject preallocated/interrupted traces; check every scalar in bounded chains."""
    post = trace.posterior
    if chains < 1 or draws < 1 or post.sizes.get("chain") != chains or post.sizes.get("draw") != draws:
        raise ValueError("Trace chain/draw count mismatch (interrupted or wrong configuration)")
    if post.get_index("chain").has_duplicates or post.get_index("draw").has_duplicates:
        raise ValueError("Trace coordinates must be unique")
    if not scalar_names or not set(scalar_names) <= set(post.data_vars):
        raise ValueError("Required scalar posterior variables missing")
    names = set(scalar_names) | {n for n, v in post.data_vars.items() if set(v.dims) == {"chain", "draw"}}
    for name in names:
        for chain in range(chains):
            _finite(post[name].isel(chain=chain).values, f"posterior {name}; interrupted trace?")
    if "sample_stats" not in trace.groups():
        raise ValueError("Trace lacks sample_stats completion evidence")
    stats = trace.sample_stats
    if stats.sizes.get("chain") != chains or stats.sizes.get("draw") != draws:
        raise ValueError("Sample stats chain/draw count mismatch")
    return trace


def atomic_write_trace(path, trace, *, metadata, chains, draws, scalar_names):
    """Validate and atomically save a complete trace using sampling.netcdf_safe."""
    from utils.sampling import netcdf_safe

    validate_trace(trace, chains=chains, draws=draws, scalar_names=scalar_names)
    # Do not mutate the caller's attrs while adding the cache protocol envelope.
    import arviz as az
    saved = az.InferenceData(**{group: trace[group].copy(deep=False) for group in trace.groups()})
    saved.posterior.attrs = {**saved.posterior.attrs, "validation_metadata": _json(metadata)}
    _atomic(path, netcdf_safe(saved).to_netcdf)


def load_trace_cache(path, *, metadata, chains, draws, scalar_names):
    """Load only an exact configured complete trace; missing returns None."""
    import arviz as az

    if not Path(path).exists():
        return None
    trace = az.from_netcdf(path)
    try:
        if trace.posterior.attrs.get("validation_metadata") != _json(metadata):
            raise ValueError("Trace cache metadata mismatch")
        validate_trace(trace, chains=chains, draws=draws, scalar_names=scalar_names)
    except Exception:
        trace.close()
        raise
    return trace


def prepare_trace_store(store_dir, *, metadata):
    """Reserve a NEW sample_lean store; never overwrite a failed/existing store.

    Call before sample_lean(..., store_dir=store_dir). Metadata is a sibling file,
    outside sample_lean's managed directory. Existing stores must be recovered or
    inspected explicitly, not deleted by an automatic sampling retry.
    """
    path = Path(store_dir)
    sidecar = path.with_name(path.name + ".metadata.json")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"Store or reservation exists at {path}; retain and inspect/recover it")
    atomic_write_json(sidecar, metadata)
    return path


def recover_trace_store(store_dir, *, metadata, chains, draws, scalar_names):
    """Recover a complete sample_lean zarr store; retain and explain failures."""
    from utils.sampling import load_store

    path = Path(store_dir)
    sidecar = path.with_name(path.name + ".metadata.json")
    if not path.exists() and not sidecar.exists():
        return None
    if not sidecar.exists() or _json(json.loads(sidecar.read_text())) != _json(metadata):
        raise ValueError(f"Store metadata mismatch at {path}; store retained")
    trace = load_store(path, draws)
    if trace is None:
        raise ValueError(f"Incomplete/unreadable store at {path}; retained for inspection, not reused")
    try:
        return validate_trace(trace, chains=chains, draws=draws, scalar_names=scalar_names)
    except ValueError as exc:
        trace.close()
        raise ValueError(f"Invalid store at {path}; retained: {exc}") from exc
