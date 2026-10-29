import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Unobserved DNFs

    2.2 learns DNF rates from editions that post their DNFs. Most editions do not: they list only finishers, so their DNFs are hidden. This model adds them, using 2.1's reporting model, to answer two questions: how many DNFs are hidden, and do the editions that post nothing change what 2.2 learned?

    For an edition that posts no DNF (or DNS) rows, two explanations compete:

    - it **reports, and had no DNFs**: its starters are exactly its finishers, each of whom finished; probability $r_e (1 - d_e)\, q_e$, with $q_e = \prod_{i \in \text{finishers}} (1 - p_i)$ from each finisher's own DNF probability under 2.2's model
    - it **does not report**: its DNFs, if any, are hidden; probability $1 - r_e$

    $r_e$ (reports, from year, series and distance) and $d_e$ (a reporting edition lists DNS rows) are 2.1's estimates before seeing the edition's zero, plugged in at their posterior means. The edition's likelihood term is $\log[\,r_e (1-d_e) q_e + (1 - r_e)\,]$. Its finishers' finish times enter the finish-time part as in 2.2; their outcomes do not enter the DNF part otherwise, because a runner who dropped out of a non-reporting edition is invisible, so a finish there says nothing about that runner's DNF tendency.

    The hidden DNFs are then predicted per edition: given $f_e$ finishers and the DNF probability $\bar p_e$ of a random starter there (course, the finishers' gender and age, a runner effect drawn from the starts in reporting editions, so frequent starters count as often as they start), starters until $f_e$ finishes give $k_e \sim \text{NegBinomial}(f_e, 1 - \bar p_e)$ hidden DNFs.

    What it leaves out: who starts (the hidden DNFers are assumed to be like the population of starters at that course), edition effects (so predictive intervals per edition are too narrow), and uncertainty in 2.1's $r_e$ and $d_e$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell(hide_code=True)
def _():
    import shutil
    import hashlib
    from datetime import datetime
    from pathlib import Path

    import arviz as az
    import numpy as np
    import pandas as pd
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt
    import xarray as xr
    import matplotlib.pyplot as plt
    from scipy.special import expit, log_expit

    from utils import finish_time_model as ftm
    from utils.sampling import check_memory, load_store, netcdf_safe, sample_lean
    from utils.data_processing import load_results, process_results

    CACHE = Path(__file__).resolve().parent.parent / 'data' / 'cache'
    OUT = CACHE / '2.3_dnf_unobserved'
    OUT.mkdir(parents=True, exist_ok=True)

    # Which 0.1 core to fit: 'core' is the dev core (3, 840), 28 courses, minutes;
    # 'core_3_500' is the final-fit core, 172 courses, hours (fewer draws to bound memory).
    CORE = 'core_3_500'
    DRAWS = 1000 if CORE == 'core' else 500
    return (
        CACHE,
        CORE,
        DRAWS,
        OUT,
        az,
        check_memory,
        datetime,
        expit,
        ftm,
        hashlib,
        load_results,
        load_store,
        netcdf_safe,
        np,
        pd,
        plt,
        pm,
        process_results,
        pt,
        pytensor,
        sample_lean,
        shutil,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data

    2.2's data (every finisher at the core courses for finish times; every start in reporting editions for DNFs), plus the editions at those courses that do not report, each with its finishers and 2.1's prior reporting probabilities. 2.2's course effects come along for comparison.
    """)
    return


@app.cell
def _(CACHE, CORE, ftm, load_results, np, pd, process_results):
    _core = pd.read_parquet(CACHE / '0.1_kcore' / f'{CORE}.parquet')
    _results = process_results(load_results())
    _at_core = _results[_results['name'].isin(_core['name'].unique())]
    finishers = _at_core[_at_core['finished']].reset_index(drop=True)
    starts = _at_core[_at_core['reports_dnfs']].reset_index(drop=True)
    course_effects_22 = pd.read_parquet(CACHE / '2.2_dnf_observed' / 'course_effects.parquet').set_index('course_id')
    _reporting = pd.read_parquet(CACHE / '2.1_dnf_reporting' / 'edition_reporting.parquet').set_index('did')

    courses = np.sort(_at_core['name'].unique())
    runners = pd.Index(pd.concat([finishers['participant_id'], starts['participant_id']]).unique())
    course_x = np.log(_at_core.groupby('name')['distance_km'].first().reindex(courses).to_numpy() / ftm.REFERENCE_KM)
    course_label = course_effects_22['label'].reindex(courses).to_numpy()
    runner_gender = (pd.concat([finishers, starts]).drop_duplicates('participant_id').set_index('participant_id')['gender']
                     .reindex(runners).eq('F').astype(int).to_numpy())

    # finish-time part (as in 1.2 and 2.2)
    _z, log_pace = ftm.design(finishers)
    f_course = courses.searchsorted(finishers['name'])
    f_runner = runners.get_indexer(finishers['participant_id'])
    f_gender = (finishers['gender'] == 'F').to_numpy().astype(int)
    f_x, f_age, f_age2 = _z[:, 1], _z[:, 2], _z[:, 3]

    # DNF part, reporting editions (as in 2.2)
    s_course = courses.searchsorted(starts['name'])
    s_runner = runners.get_indexer(starts['participant_id'])
    s_gender = (starts['gender'] == 'F').to_numpy().astype(int)
    s_age = ((starts['age'] - ftm.AGE_CENTER) / ftm.AGE_SCALE).to_numpy()
    dnf = (~starts['finished']).to_numpy().astype(int)

    # every edition at the core courses, with its finishers (for q_e and for predicting hidden DNFs)
    edition_ids, f_edition = np.unique(finishers['event_distance_id'].to_numpy(), return_inverse=True)
    editions = (finishers.groupby('event_distance_id')
                .agg(course_id=('name', 'first'), year=('year', 'first'), reports_dnfs=('reports_dnfs', 'first'),
                     n_finishers=('finished', 'size'))
                .reindex(edition_ids))
    editions['n_dnf'] = starts.groupby('event_distance_id')['finished'].apply(lambda f: int((~f).sum())).reindex(edition_ids).fillna(0).astype(int)
    editions = editions.join(_reporting[['kind', 'p_reports', 'p_reports_prior', 'p_lists_dns']])
    editions['course_idx'] = courses.searchsorted(editions['course_id'])
    unknown = (~editions['reports_dnfs']).to_numpy()
    # editions 2.1 did not cover (none expected at these courses) count as non-reporting
    editions['p_reports_prior'] = editions['p_reports_prior'].fillna(0.0)
    editions['p_lists_dns'] = editions['p_lists_dns'].fillna(0.5)

    _eds = pd.DataFrame({
        'editions': editions.groupby([editions['reports_dnfs'], editions['year'] >= 2010]).size(),
        'finishers': editions.groupby([editions['reports_dnfs'], editions['year'] >= 2010])['n_finishers'].sum(),
        'mean P(reports), 2.1 prior': editions.groupby([editions['reports_dnfs'], editions['year'] >= 2010])['p_reports_prior'].mean().round(3),
        'mean P(reports), 2.1 posterior': editions.groupby([editions['reports_dnfs'], editions['year'] >= 2010])['p_reports'].mean().round(3),
    })
    _eds.index = _eds.index.set_names(['reports DNFs', '2010 or later'])
    _eds
    return (
        course_label,
        course_x,
        courses,
        dnf,
        editions,
        f_age,
        f_age2,
        f_course,
        f_edition,
        f_gender,
        f_runner,
        f_x,
        log_pace,
        runner_gender,
        runners,
        s_age,
        s_course,
        s_gender,
        s_runner,
        unknown,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model

    2.2's full model (finish times, course and runner DNF tendencies correlated with pace, curvature in runner pace) with its priors, plus one likelihood term per non-reporting edition. That term adds no parameters: $r_e$ and $d_e$ are fixed from 2.1, and $q_e$ is a function of 2.2's parameters. `build_unobserved_model(unknown='naive')` builds the comparison model, which takes every non-reporting edition at its word (it reported, and had no DNFs: the term becomes $\log q_e$).
    """)
    return


@app.cell
def _(
    course_x,
    courses,
    dnf,
    editions,
    expit,
    f_age,
    f_age2,
    f_course,
    f_edition,
    f_gender,
    f_runner,
    f_x,
    ftm,
    log_pace,
    np,
    pm,
    pt,
    runner_gender,
    runners,
    s_age,
    s_course,
    s_gender,
    s_runner,
    unknown,
):
    _avg = np.eye(2)[runner_gender] / np.bincount(runner_gender, minlength=2)
    _f_half = np.array([-0.5, 0.5])[f_gender]
    _s_half = np.array([-0.5, 0.5])[s_gender]
    # finishers in non-reporting editions, for q_e
    _u_rows = np.flatnonzero(unknown[f_edition])
    _u_edition = np.unique(f_edition[_u_rows], return_inverse=True)[1]
    _u_ids = np.flatnonzero(unknown)
    _n_unknown = len(_u_ids)
    # 2.1's probabilities for each non-reporting edition, plugged in
    unknown_r = np.clip(editions['p_reports_prior'].to_numpy()[_u_ids], 1e-9, 1 - 1e-9)
    unknown_d = np.clip(editions['p_lists_dns'].to_numpy()[_u_ids], 1e-9, 1 - 1e-9)
    unknown_log_r_nodns = np.log(unknown_r) + np.log1p(-unknown_d)
    unknown_log_not_r = np.log1p(-unknown_r)


    def _zero_sum(raw):
        return raw - pm.math.dot(raw, _avg)[runner_gender]


    def build_unobserved_model(unknown='mixture'):
        """2.2's full model plus a term per non-reporting edition: 'mixture' (full) or 'naive' (all report, zero DNFs)."""
        with pm.Model(coords={'gender': list(ftm.GENDERS), 'course': courses, 'runner': runners}) as _model:
            _pace = pm.Normal('pace_50k_mean', np.log(8.0), 0.3)
            _pace_gap = pm.Normal('pace_50k_gap', 0.0, 0.2)
            _exp = pm.Normal('distance_exponent_mean', 0.1, 0.1)
            _exp_gap = pm.Normal('distance_exponent_gap', 0.0, 0.1)
            _a1 = pm.Normal('age_linear', 0.0, 0.1, dims='gender')
            _a2 = pm.Normal('age_quadratic', 0.0, 0.05, dims='gender')
            _tau = pm.HalfNormal('tau', 0.3)
            _omega = pm.HalfNormal('omega', 0.3)
            _sigma = pm.HalfNormal('sigma', 0.3, dims='gender')
            _course_mean = pm.Normal('course_mean', _pace + _exp * course_x, _tau, dims='course')
            _u = pm.Deterministic('course_effect', _course_mean - (_pace + _exp * course_x), dims='course')
            _z1 = _zero_sum(pm.Normal('runner_raw', 0.0, 1.0, dims='runner'))
            _v = pm.Deterministic('runner_effect', _omega * _z1, dims='runner')
            _mu = (_course_mean[f_course] + _f_half * (_pace_gap + _exp_gap * f_x)
                   + _a1[f_gender] * f_age + _a2[f_gender] * f_age2 + _v[f_runner])
            pm.Normal('log_pace', _mu, _sigma[f_gender], observed=log_pace)

            _d0 = pm.Normal('dnf_mean', -1.4, 1.0)
            _dd = pm.Normal('dnf_distance', 0.0, 1.0)
            _dg = pm.Normal('dnf_gender_gap', 0.0, 0.5)
            _da1 = pm.Normal('dnf_age_linear', 0.0, 0.5)
            _da2 = pm.Normal('dnf_age_quadratic', 0.0, 0.25)
            _tau_d = pm.HalfNormal('tau_dnf', 1.0)
            _rho_c = pm.Deterministic('rho_course', 2 * pm.Beta('rho_course_01', 2, 2) - 1)
            _level = pm.Normal('course_dnf_level', _d0 + _dd * course_x + _tau_d * _rho_c * _u / _tau,
                               _tau_d * pt.sqrt(1 - _rho_c**2), dims='course')
            pm.Deterministic('course_dnf_effect', _level - (_d0 + _dd * course_x), dims='course')
            _omega_d = pm.HalfNormal('omega_dnf', 1.0)
            _z2 = _zero_sum(pm.Normal('runner_dnf_raw', 0.0, 1.0, dims='runner'))
            _rho_r = pm.Deterministic('rho_runner', 2 * pm.Beta('rho_runner_01', 2, 2) - 1)
            _q = pm.Normal('dnf_runner_pace_quadratic', 0.0, 0.5)
            _w = pm.Deterministic('runner_dnf_effect', _omega_d * (_rho_r * _z1 + pt.sqrt(1 - _rho_r**2) * _z2) + _q * (_z1**2 - 1), dims='runner')
            _eta = _level[s_course] + _s_half * _dg + _da1 * s_age + _da2 * s_age**2 + _w[s_runner]
            pm.Bernoulli('dnf', logit_p=_eta, observed=dnf)

            # non-reporting editions: log q_e = sum over finishers of log(1 - p_i)
            _rows = _u_rows
            _eta_u = (_level[f_course[_rows]] + np.array([-0.5, 0.5])[f_gender[_rows]] * _dg
                      + _da1 * f_age[_rows] + _da2 * f_age2[_rows] + _w[f_runner[_rows]])
            _log_q = pt.inc_subtensor(pt.zeros(_n_unknown)[_u_edition], -pt.softplus(_eta_u))
            if unknown == 'mixture':
                pm.Potential('unreported', pm.math.logaddexp(unknown_log_r_nodns + _log_q, unknown_log_not_r).sum())
            else:
                pm.Potential('unreported', _log_q.sum())
        return _model


    FINISH_SCALARS = ['pace_50k_mean', 'pace_50k_gap', 'distance_exponent_mean', 'distance_exponent_gap', 'tau', 'omega', 'sigma']
    DNF_SCALARS = ['dnf_mean', 'dnf_distance', 'dnf_gender_gap', 'dnf_age_linear', 'dnf_age_quadratic', 'tau_dnf', 'omega_dnf',
                   'rho_course', 'rho_runner', 'dnf_runner_pace_quadratic']


    def flat_draws(posterior, name):
        return posterior[name].stack(s=('chain', 'draw')).transpose('s', ...).values


    def finisher_dnf_logit(posterior, rows):
        """DNF logit of the given finisher rows, with their own runner effects, as (draw, row)."""
        return (flat_draws(posterior, 'course_dnf_level')[:, f_course[rows]]
                + np.array([-0.5, 0.5])[f_gender[rows]] * flat_draws(posterior, 'dnf_gender_gap')[:, None]
                + flat_draws(posterior, 'dnf_age_linear')[:, None] * f_age[rows] + flat_draws(posterior, 'dnf_age_quadratic')[:, None] * f_age2[rows]
                + flat_draws(posterior, 'runner_dnf_effect')[:, f_runner[rows]])


    def unknown_terms(posterior):
        """Per non-reporting edition: log q_e and the edition's log-likelihood term, as (draw, edition)."""
        _lq = -np.logaddexp(0, finisher_dnf_logit(posterior, _u_rows))
        _log_q = np.stack([np.bincount(_u_edition, weights=_row, minlength=_n_unknown) for _row in _lq])
        return _log_q, np.logaddexp(unknown_log_r_nodns + _log_q, unknown_log_not_r)


    def starter_dnf_prob(posterior, n_nodes=400, seed=0):
        """DNF probability of a random starter in every edition, as (draw, edition): the course, the
        gender and age of the edition's finishers, and a runner effect drawn from the starts in
        reporting editions (so frequent starters count as often as they start; drawing from the
        runner population instead over-predicts DNFs, since frequent starters drop less)."""
        _rng = np.random.default_rng(seed)
        _base = (flat_draws(posterior, 'course_dnf_level')[:, f_course]
                 + np.array([-0.5, 0.5])[f_gender] * flat_draws(posterior, 'dnf_gender_gap')[:, None]
                 + flat_draws(posterior, 'dnf_age_linear')[:, None] * f_age + flat_draws(posterior, 'dnf_age_quadratic')[:, None] * f_age2)
        _wall = flat_draws(posterior, 'runner_dnf_effect')
        _n_ed = np.bincount(f_edition)
        _out = np.empty((len(_base), len(_n_ed)))
        for _s in range(len(_base)):
            _w = _wall[_s, s_runner[_rng.integers(0, len(s_runner), n_nodes)]]
            _p = expit(_base[_s][:, None] + _w[None, :]).mean(1)
            _out[_s] = np.bincount(f_edition, weights=_p) / _n_ed
        return _out


    unobserved_model = build_unobserved_model()
    unobserved_model
    return (
        DNF_SCALARS,
        FINISH_SCALARS,
        build_unobserved_model,
        flat_draws,
        starter_dnf_prob,
        unknown_d,
        unknown_log_r_nodns,
        unknown_r,
        unknown_terms,
        unobserved_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior predictive check

    The priors are 2.2's and the new term has no parameters of its own, so the check that matters is what the priors imply for the quantity this notebook predicts: the share of starters who DNF, as a random starter's DNF probability across editions, against the observed DNF rate of reporting editions (red).
    """)
    return


@app.cell
def _(editions, plt, pm, starter_dnf_prob, unknown, unobserved_model):
    with unobserved_model:
        _prior = pm.sample_prior_predictive(draws=100, var_names=['course_dnf_level', 'dnf_gender_gap', 'dnf_age_linear', 'dnf_age_quadratic',
                                                                  'omega_dnf', 'rho_runner', 'dnf_runner_pace_quadratic', 'runner_dnf_effect'], random_seed=1).prior
    _p_prior = starter_dnf_prob(_prior, n_nodes=16)
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _ax.hist(_p_prior.ravel(), bins=50, range=(0, 1), density=True, color='#999999', label='prior: random starter, per edition')
    _ax.hist(editions.loc[~unknown, 'n_dnf'] / (editions.loc[~unknown, 'n_dnf'] + editions.loc[~unknown, 'n_finishers']), bins=50, range=(0, 1),
             density=True, histtype='step', color='#E63946', lw=2, label='observed, reporting editions')
    _ax.set(xlabel='DNF probability', ylabel='Density', title='Prior predictive DNF probability per edition')
    _ax.legend()
    _ax.grid(alpha=0.3)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit

    As in 2.2, runner draws are stored as float32 without the raw effects.
    """)
    return


@app.cell
def _(
    DNF_SCALARS,
    DRAWS,
    FINISH_SCALARS,
    OUT,
    az,
    check_memory,
    datetime,
    dnf,
    f_course,
    f_runner,
    hashlib,
    load_store,
    log_pace,
    mo,
    netcdf_safe,
    np,
    runner_gender,
    s_age,
    s_course,
    s_runner,
    sample_lean,
    shutil,
    unknown_d,
    unknown_r,
    unobserved_model,
    xr,
):
    _key = hashlib.sha1(b''.join(np.ascontiguousarray(a).tobytes() for a in (log_pace, f_course, f_runner, dnf, s_course, s_runner, s_age, unknown_r, unknown_d))).hexdigest()[:12]
    _PER_RUNNER = ('runner_effect', 'runner_dnf_effect')


    def _derive_runner_effects(post):
        """Per-runner effects from the stored raw effects and scalars, as float32, one chain at a
        time (they are left out of sampling to keep its memory small). Mirrors the model."""
        _avg = np.eye(2)[runner_gender] / np.bincount(runner_gender, minlength=2)

        def _zero_sum(raw):
            return raw - (raw @ _avg)[..., runner_gender]

        _out = {}
        _z1 = np.stack([_zero_sum(post['runner_raw'].values[c]).astype(np.float32) for c in range(post.sizes['chain'])])
        _sc = lambda v: post[v].values[..., None].astype(np.float32)
        _out['runner_effect'] = _sc('omega') * _z1
        if 'runner_dnf_raw' in post:
            _z2 = np.stack([_zero_sum(post['runner_dnf_raw'].values[c]).astype(np.float32) for c in range(post.sizes['chain'])])
            if 'rho_runner' in post:
                _r = _sc('rho_runner')
                _w = _sc('omega_dnf') * (_r * _z1 + np.sqrt(1 - _r**2) * _z2)
                if 'dnf_runner_pace_quadratic' in post:
                    _w = _w + _sc('dnf_runner_pace_quadratic') * (_z1**2 - 1)
            else:
                _w = _sc('omega_dnf') * _z2
            _out['runner_dnf_effect'] = _w
        _dims = ('chain', 'draw', 'runner')
        _post = post.drop_vars([v for v in ('runner_raw', 'runner_dnf_raw') if v in post])
        for _k, _v in _out.items():
            _post[_k] = xr.DataArray(_v, dims=_dims, coords={'chain': post['chain'], 'draw': post['draw'], 'runner': post['runner']})
        return _post


    # Bump when the model changes, so a cached trace is never reused for a different model.
    _MODEL_VERSION = 1


    def fit_unobserved(model, label, minutes):
        """Sample `model` with nutpie, or load its cached trace (keyed by label, model version and data)."""
        _path = OUT / f'unobserved_{label}_v{_MODEL_VERSION}_{_key}.nc'
        if _path.exists():
            return az.from_netcdf(_path)
        _store = _path.with_suffix('.zarr')  # draws stream here, not into memory
        _trace = load_store(_store, DRAWS)    # a finished store whose save failed: no need to resample
        if _trace is None:
            _keep = [v.name for v in model.free_RVs + model.deterministics if v.name not in _PER_RUNNER]
            _gb = check_memory(model, _keep, chains=4, draws=DRAWS, on_disk=True)
            mo.output.replace(mo.callout(mo.md(f"Sampling the {label} model since {datetime.now():%H:%M} (about {minutes} minutes, about {_gb:.0f} GB)."), kind='info'))
            _trace = sample_lean(model, _keep, draws=DRAWS, random_seed=42, progressbar=True, store_dir=_store)
        _post = _derive_runner_effects(_trace.posterior)
        _trace = az.InferenceData(posterior=_post, sample_stats=_trace.sample_stats)
        # write then rename, so a failed save never leaves a file that looks like a cached trace
        _tmp = _path.with_suffix('.nc.tmp')
        netcdf_safe(_trace).to_netcdf(_tmp)
        _tmp.replace(_path)
        shutil.rmtree(_store)
        return az.from_netcdf(_path)


    unobserved_trace = fit_unobserved(unobserved_model, 'mixture', 15)
    unobserved_summary = az.summary(unobserved_trace, var_names=FINISH_SCALARS + DNF_SCALARS, round_to=4)
    print(f"Divergences: {int(unobserved_trace.sample_stats['diverging'].sum())}")
    _ess = az.ess(unobserved_trace, var_names=['course_mean', 'course_dnf_level', 'runner_effect', 'runner_dnf_effect'])
    print('Min bulk ESS: ' + ', '.join(f'{v} {float(_ess[v].min()):.0f}' for v in _ess.data_vars))
    unobserved_summary
    return fit_unobserved, unobserved_summary, unobserved_trace


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Convergence

    Rank plots for the DNF part's population parameters; R-hat and ESS for all are in the summary above.
    """)
    return


@app.cell
def _(DNF_SCALARS, az, np, unobserved_trace):
    _axes = az.plot_rank(unobserved_trace, var_names=DNF_SCALARS, kind='vlines', figsize=(14, 13))
    _fig = np.ravel(_axes)[0].figure
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posterior predictive checks

    The hidden-DNF prediction is checked where the answer is known: for every reporting edition from 2010 on, predict its DNF count from its finishers alone (the way hidden DNFs are predicted) and compare with the DNFs it posted. Left: predicted against posted DNFs per edition (94% predictive interval). Right: the share of editions whose posted count falls inside the 50% and 94% predictive intervals, which should be about 0.5 and 0.94; without edition effects, expect too few inside.
    """)
    return


@app.cell
def _(editions, np, pd, starter_dnf_prob, unknown, unobserved_trace):
    _sub = unobserved_trace.posterior.isel(draw=slice(None, None, 20))
    starter_p = starter_dnf_prob(_sub)
    _rng = np.random.default_rng(11)
    _f = editions['n_finishers'].to_numpy()
    hidden_dnf_draws = _rng.negative_binomial(np.maximum(_f, 1)[None, :], 1 - starter_p)

    _check = (~unknown) & (editions['year'].to_numpy() >= 2010)
    _pred = hidden_dnf_draws[:, _check]
    _obs = editions['n_dnf'].to_numpy()[_check]
    dnf_count_check = pd.DataFrame({
        'posted': _obs, 'predicted_mean': _pred.mean(0),
        'lo94': np.percentile(_pred, 3, axis=0), 'hi94': np.percentile(_pred, 97, axis=0),
        'lo50': np.percentile(_pred, 25, axis=0), 'hi50': np.percentile(_pred, 75, axis=0),
    })
    return dnf_count_check, hidden_dnf_draws


@app.cell
def _(dnf_count_check, plt):
    _c = dnf_count_check
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [3, 2]})
    _ax1.errorbar(_c['posted'], _c['predicted_mean'], yerr=[_c['predicted_mean'] - _c['lo94'], _c['hi94'] - _c['predicted_mean']],
                  fmt='o', ms=3, alpha=0.4, color='#2A9D8F')
    _lim = [0, max(_c['posted'].max(), _c['hi94'].max()) * 1.05]
    _ax1.plot(_lim, _lim, color='gray', lw=1)
    _ax1.set(xlabel='DNFs posted', ylabel='DNFs predicted from finishers (94% interval)', title=f'Reporting editions from 2010 ({len(_c)})', xlim=_lim, ylim=_lim)
    _cover = {'50%': ((_c['posted'] >= _c['lo50']) & (_c['posted'] <= _c['hi50'])).mean(),
              '94%': ((_c['posted'] >= _c['lo94']) & (_c['posted'] <= _c['hi94'])).mean()}
    _ax2.bar(list(_cover), list(_cover.values()), color='#2A9D8F')
    _ax2.plot([-0.4, 0.4], [0.5, 0.5], color='#E63946')
    _ax2.plot([0.6, 1.4], [0.94, 0.94], color='#E63946')
    _ax2.set(ylim=(0, 1), ylabel='Share of editions inside the interval', title='Predictive interval coverage (red: nominal)')
    for _ax in (_ax1, _ax2):
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model comparison

    PSIS-LOO over every DNF observation (each start in a reporting edition, and each non-reporting edition's term), for the mixture model against the naive one that takes every silent edition as reporting zero DNFs. Both models assign a probability to the same observed events, so their scores are comparable. 500 draws per model.

    On a final-fit core (`CORE` other than `'core'`) only the full model is fitted, so this table has one row; the comparison is the dev core's.
    """)
    return


@app.cell
def _(CORE, build_unobserved_model, fit_unobserved, unobserved_trace):
    # On the dev core every variant is fitted for the comparison; on a final-fit core only the full
    # model (hours each), and the comparison below is the dev core's (see its cached traces).
    if CORE == 'core':
        unobserved_comparison_traces = {
            'mixture': unobserved_trace,
            'naive': fit_unobserved(build_unobserved_model('naive'), 'naive', 15),
        }
    else:
        unobserved_comparison_traces = {'mixture': unobserved_trace}
    return (unobserved_comparison_traces,)


@app.cell
def _(
    az,
    dnf,
    flat_draws,
    np,
    s_age,
    s_course,
    s_gender,
    s_runner,
    unknown_terms,
    unobserved_comparison_traces,
    xr,
):
    def _with_loglik(trace, naive):
        _post = trace.posterior.isel(draw=slice(None, None, 8))
        _eta = (flat_draws(_post, 'course_dnf_level')[:, s_course] + np.array([-0.5, 0.5])[s_gender] * flat_draws(_post, 'dnf_gender_gap')[:, None]
                + flat_draws(_post, 'dnf_age_linear')[:, None] * s_age + flat_draws(_post, 'dnf_age_quadratic')[:, None] * s_age**2
                + flat_draws(_post, 'runner_dnf_effect')[:, s_runner])
        _ll_starts = np.where(dnf == 1, -np.logaddexp(0, -_eta), -np.logaddexp(0, _eta))
        _log_q, _term = unknown_terms(_post)
        _ll = np.concatenate([_ll_starts, _log_q if naive else _term], axis=1).astype(np.float32)
        _da = xr.DataArray(_ll.reshape(_post.sizes['chain'], _post.sizes['draw'], -1), dims=('chain', 'draw', 'observation'),
                           coords={'chain': _post['chain'], 'draw': _post['draw']})
        return az.InferenceData(posterior=_post, log_likelihood=xr.Dataset({'dnf': _da}))


    unobserved_loo_data = {_n: _with_loglik(_t, _n == 'naive') for _n, _t in unobserved_comparison_traces.items()}
    def _loo(idata):
        # Terms that barely vary across draws (pre-2010 editions, P(reports) near 0) give PSIS a
        # degenerate weight fit and a NaN; their LOO predictive is just the posterior mean.
        _l = az.loo(idata, pointwise=True)
        _ll = idata.log_likelihood['dnf'].stack(s=('chain', 'draw')).values
        _bad = np.isnan(_l.loo_i.values)
        if _bad.any():
            _l.loo_i.values[_bad] = np.logaddexp.reduce(_ll[_bad], axis=1) - np.log(_ll.shape[1])
            _l['elpd_loo'] = float(_l.loo_i.values.sum())
            _l['se'] = float(np.sqrt(len(_l.loo_i) * _l.loo_i.values.var()))
            _l['p_loo'] = float(np.nansum(np.logaddexp.reduce(_ll, axis=1) - np.log(_ll.shape[1]) - _l.loo_i.values))
        return _l


    unobserved_loos = {_n: _loo(_d) for _n, _d in unobserved_loo_data.items()}
    unobserved_comparison = az.compare(unobserved_loos)[['rank', 'elpd_loo', 'p_loo', 'elpd_diff', 'se', 'dse']].assign(
        observations_k_over_0_7=lambda d: [int((unobserved_loos[_m].pareto_k > 0.7).sum()) for _m in d.index]
    )
    unobserved_comparison.round(1)
    return (unobserved_loo_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior sensitivity

    Power-scaling sensitivity of the population parameters to their own priors, as in 2.2. Values above 0.05 mean the prior shapes that parameter.
    """)
    return


@app.cell
def _(
    DNF_SCALARS,
    FINISH_SCALARS,
    az,
    ftm,
    np,
    pd,
    pm,
    pytensor,
    unobserved_loo_data,
    unobserved_model,
    xr,
):
    _idata = unobserved_loo_data['mixture'].copy()
    _scalars = FINISH_SCALARS + DNF_SCALARS
    _chosen = ['pace_50k_mean', 'pace_50k_gap', 'distance_exponent_mean', 'distance_exponent_gap', 'age_linear', 'age_quadratic',
               'tau', 'omega', 'sigma', 'dnf_mean', 'dnf_distance', 'dnf_gender_gap', 'dnf_age_linear', 'dnf_age_quadratic',
               'tau_dnf', 'omega_dnf', 'rho_course_01', 'rho_runner_01', 'dnf_runner_pace_quadratic']
    # Log density of each chosen prior at the posterior draws (each has fixed parameters, so it is
    # evaluated on its own; the stored trace drops the raw runner effects compute_log_prior needs).
    _log_prior = {}
    for _v in _chosen:
        _rv = unobserved_model[_v]
        _val = _rv.type()
        _fn = pytensor.function([_val], pm.logp(_rv, _val))
        _post = _idata.posterior[_v]
        _log_prior[_v] = xr.DataArray(
            np.array([_fn(np.asarray(x, dtype=_rv.dtype)) for x in _post.values.reshape(-1, *_post.shape[2:])]).reshape(_post.shape),
            dims=_post.dims, coords=_post.coords)
    _idata.add_groups(log_prior=xr.Dataset(_log_prior))
    _s = az.psens(_idata, component='prior', component_var_names=_chosen, var_names=_scalars)
    _rows = []
    for _v in _scalars:
        _sel = [('', {})] if 'gender' not in _s[_v].dims else [(f'[{g}]', {'gender': g}) for g in ftm.GENDERS]
        _rows += [{'parameter': _v + _l, 'prior sensitivity': float(_s[_v].sel(_c))} for _l, _c in _sel]
    unobserved_sensitivity = pd.DataFrame(_rows).set_index('parameter')
    unobserved_sensitivity['diagnosis'] = np.where(unobserved_sensitivity['prior sensitivity'] >= 0.05, 'prior-sensitive', 'data-dominated')
    unobserved_sensitivity.round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results

    **Left**: for each non-reporting edition, the probability that it really reports (and had no DNFs) from this model against 2.1's, which used only distance for the DNF rate. **Middle**: DNFs per year at these courses, posted against hidden (94% band). **Right**: each course's DNF rate, from posted DNFs only (reporting editions) against all its editions including the hidden DNFs.
    """)
    return


@app.cell
def _(
    course_label,
    courses,
    editions,
    hidden_dnf_draws,
    np,
    pd,
    plt,
    unknown,
    unknown_log_r_nodns,
    unknown_terms,
    unobserved_trace,
):
    _sub = unobserved_trace.posterior.isel(draw=slice(None, None, 20))
    _log_q, _term = unknown_terms(_sub)
    _p_rep = np.exp(unknown_log_r_nodns + _log_q - _term)  # P(reports | no DNF rows), per draw
    _p_rep_mean = _p_rep.mean(0)
    editions['p_reports_23'] = 1.0
    editions.loc[unknown, 'p_reports_23'] = _p_rep_mean
    _hidden = hidden_dnf_draws.astype(float)
    _hidden[:, ~unknown] = 0.0
    _hidden[:, unknown] *= (1 - _p_rep)                   # expected hidden DNFs given the reporting uncertainty
    editions['hidden_dnf_mean'] = _hidden.mean(0)
    editions['hidden_dnf_lo94'] = np.where(unknown, np.percentile(hidden_dnf_draws, 3, axis=0), 0)
    editions['hidden_dnf_hi94'] = np.where(unknown, np.percentile(hidden_dnf_draws, 97, axis=0), 0)

    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
    _u = editions[unknown]
    _axes[0].scatter(_u['p_reports'], _u['p_reports_23'], s=8, alpha=0.5, color='#2A9D8F')
    _axes[0].plot([0, 1], [0, 1], color='gray', lw=1)
    _axes[0].set(xscale='symlog', yscale='symlog', xlim=(0, 1), ylim=(0, 1), xlabel='P(reports | nothing posted), 2.1',
                 ylabel='P(reports | nothing posted), 2.3', title=f'Non-reporting editions ({int(unknown.sum())})')
    _years = np.sort(editions['year'].unique())
    _yi = np.searchsorted(_years, editions['year'].to_numpy())
    _posted = np.bincount(_yi, weights=editions['n_dnf'].to_numpy(), minlength=len(_years))
    _hid = np.stack([np.bincount(_yi, weights=_h, minlength=len(_years)) for _h in _hidden])
    _axes[1].bar(_years, _posted, color='#264653', label='posted')
    _axes[1].bar(_years, np.median(_hid, axis=0), bottom=_posted, color='#E9C46A', label='hidden (median)')
    _axes[1].errorbar(_years, _posted + np.median(_hid, axis=0), yerr=np.abs(np.percentile(_hid, [3, 97], axis=0) - np.median(_hid, axis=0)),
                      fmt='none', color='#E76F51', lw=1)
    _axes[1].set(xlabel='Year', ylabel='DNFs at the core courses', title='Posted and hidden DNFs per year')
    _axes[1].legend()
    _rate = pd.DataFrame({
        'course': editions['course_idx'], 'posted': editions['n_dnf'], 'hidden': editions['hidden_dnf_mean'],
        'finishers': editions['n_finishers'], 'reports': ~unknown,
    })
    _g = _rate.groupby('course')
    _rep = _rate[_rate['reports']].groupby('course')
    _rate_posted = (_rep['posted'].sum() / (_rep['posted'].sum() + _rep['finishers'].sum())).reindex(range(len(courses)))
    _rate_all = (_g['posted'].sum() + _g['hidden'].sum()) / (_g['posted'].sum() + _g['hidden'].sum() + _g['finishers'].sum())
    course_dnf_rates = pd.DataFrame({'label': course_label, 'reporting_editions': _rate_posted.to_numpy(), 'all_editions': _rate_all.to_numpy(),
                                     'share_editions_reporting': _g['reports'].mean().to_numpy()}, index=courses)
    _axes[2].scatter(course_dnf_rates['reporting_editions'], course_dnf_rates['all_editions'], s=40, color='#2A9D8F')
    _axes[2].plot([0, 0.5], [0, 0.5], color='gray', lw=1)
    for _i, (_cid, _row) in enumerate(course_dnf_rates.iterrows()):
        if abs(_row['all_editions'] - _row['reporting_editions']) > 0.03:
            _axes[2].annotate(str(_row['label'])[:24], (_row['reporting_editions'], _row['all_editions']), fontsize=7, xytext=(4, 2), textcoords='offset points')
    _axes[2].set(xlabel='DNF rate, reporting editions (posted)', ylabel='DNF rate, all editions (posted + hidden)', title='Course DNF rates')
    for _ax in _axes:
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return (course_dnf_rates,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Do the silent editions change what 2.2 learned? The DNF population parameters, 2.2 against 2.3 (posterior mean and 94% interval).
    """)
    return


@app.cell
def _(CACHE, DNF_SCALARS, az, pd, unobserved_summary):
    _t22 = az.from_netcdf(max((CACHE / '2.2_dnf_observed').glob('dnf_quadratic_v1_*.nc'), key=lambda p: p.stat().st_mtime))  # the most recent 2.2 fit
    _a = az.summary(_t22, var_names=DNF_SCALARS, round_to=3)[['mean', 'hdi_3%', 'hdi_97%']].add_prefix('2.2 ')
    _b = unobserved_summary.loc[DNF_SCALARS, ['mean', 'hdi_3%', 'hdi_97%']].round(3).add_prefix('2.3 ')
    pd.concat([_a, _b], axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Handoff

    Saved to `data/cache/2.3_dnf_unobserved/`: per edition at the core courses (`edition_dnfs.parquet`: finishers, posted DNFs, P(reports) from 2.1 and from this model, expected hidden DNFs with a 94% interval) and per course (`course_dnf_rates.parquet`: DNF rate from posted DNFs and from all editions).
    """)
    return


@app.cell
def _(OUT, course_dnf_rates, editions):
    edition_dnfs = editions.reset_index().rename(columns={'event_distance_id': 'did'})[
        ['did', 'course_id', 'year', 'reports_dnfs', 'n_finishers', 'n_dnf', 'p_reports', 'p_reports_23', 'hidden_dnf_mean', 'hidden_dnf_lo94', 'hidden_dnf_hi94']]
    edition_dnfs.to_parquet(OUT / 'edition_dnfs.parquet', index=False)
    course_dnf_rates.rename_axis('course_id').reset_index().to_parquet(OUT / 'course_dnf_rates.parquet', index=False)
    course_dnf_rates.assign(change=lambda d: d['all_editions'] - d['reporting_editions']).sort_values('change').round(3)
    return


if __name__ == "__main__":
    app.run()
