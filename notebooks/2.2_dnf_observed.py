import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Observed DNFs

    Adds DNFs to the runner-effect finish-time model (1.2). Each start ends in a finish or a DNF; the probability of a DNF depends on distance, gender, age, the course and the runner. The course's DNF tendency is correlated with its pace effect, and so is the runner's: that is the question this model answers. Do slower runners drop more often, and is a hard course's high DNF rate partly a matter of who starts it?

    $$\text{logit}\, P(\text{DNF}_i) = \ell_c + \tfrac{\pm 1}{2}\delta + \kappa_1 A_i + \kappa_2 A_i^2 + w_r + q\,(v_r^2/\omega^2 - 1)$$

    $$\ell_c \sim N\!\big(\mu + \lambda x_c + \tau_d \rho_c\, u_c / \tau,\; \tau_d^2 (1 - \rho_c^2)\big), \qquad (v_r, w_r) \sim N\!\left(0, \begin{pmatrix}\omega^2 & \rho_r\, \omega\, \omega_d \\ \rho_r\, \omega\, \omega_d & \omega_d^2\end{pmatrix}\right)$$

    where $u_c$ is the course's pace effect and $v_r$ the runner's pace effect from the finish-time part, which is 1.2's model unchanged. The curvature $q$ lets both the fastest and the slowest runners drop more often than a straight line in pace allows: the posterior predictive check of the linear version showed exactly that. The DNF part uses only editions that report their DNFs; finish times come from every edition, since a finish time is valid whether or not DNFs were posted. Editions that do not report are 2.3's subject.

    What it leaves out: DNFs in editions that do not report them, edition effects (a hot year raises both pace and DNFs), and the finish times DNF runners would have run.
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
    from scipy.special import expit

    from utils import finish_time_model as ftm
    from utils.sampling import check_memory, sample_lean
    from utils.data_processing import load_results, process_results

    CACHE = Path(__file__).resolve().parent.parent / 'data' / 'cache'
    OUT = CACHE / '2.2_dnf_observed'
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

    The courses of 0.1's core (`CORE`). Finish times: every finisher (as in 1.2). DNFs: every start (finish or DNF) in editions that report their DNFs (`reports_dnfs`). 1.2's runner-adjusted course effects come along for comparison.
    """)
    return


@app.cell
def _(CACHE, CORE, ftm, load_results, np, pd, process_results):
    _core = pd.read_parquet(CACHE / '0.1_kcore' / f'{CORE}.parquet')
    _results = process_results(load_results())
    _at_core = _results[_results['name'].isin(_core['name'].unique())]
    finishers = _at_core[_at_core['finished']].reset_index(drop=True)
    starts = _at_core[_at_core['reports_dnfs']].reset_index(drop=True)
    course_effects_12 = pd.read_parquet(CACHE / '1.2_finish_runners' / 'course_effects.parquet').set_index('course_id')

    courses = np.sort(_at_core['name'].unique())
    runners = pd.Index(pd.concat([finishers['participant_id'], starts['participant_id']]).unique())
    course_x = np.log(_at_core.groupby('name')['distance_km'].first().reindex(courses).to_numpy() / ftm.REFERENCE_KM)
    course_label = course_effects_12['label'].reindex(courses).to_numpy()
    runner_gender = (pd.concat([finishers, starts]).drop_duplicates('participant_id').set_index('participant_id')['gender']
                     .reindex(runners).eq('F').astype(int).to_numpy())

    # finish-time part (as in 1.2)
    _z, log_pace = ftm.design(finishers)
    f_course = courses.searchsorted(finishers['name'])
    f_runner = runners.get_indexer(finishers['participant_id'])
    f_gender = (finishers['gender'] == 'F').to_numpy().astype(int)
    f_x, f_age, f_age2 = _z[:, 1], _z[:, 2], _z[:, 3]

    # DNF part
    s_course = courses.searchsorted(starts['name'])
    s_runner = runners.get_indexer(starts['participant_id'])
    s_gender = (starts['gender'] == 'F').to_numpy().astype(int)
    s_age = ((starts['age'] - ftm.AGE_CENTER) / ftm.AGE_SCALE).to_numpy()
    dnf = (~starts['finished']).to_numpy().astype(int)

    pd.DataFrame({'count': {
        'finishes (finish-time part)': len(finishers),
        'starts in reporting editions (DNF part)': len(starts),
        'DNFs': int(dnf.sum()),
        'reporting editions': starts['event_distance_id'].nunique(),
        'courses': len(courses),
        'runners': len(runners),
        'runners with 2+ reporting starts': int((np.bincount(s_runner, minlength=len(runners)) >= 2).sum()),
    }})
    return (
        course_effects_12,
        course_label,
        course_x,
        courses,
        dnf,
        f_age,
        f_age2,
        f_course,
        f_gender,
        f_runner,
        f_x,
        finishers,
        log_pace,
        runner_gender,
        runners,
        s_age,
        s_course,
        s_gender,
        s_runner,
        starts,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Do slower runners drop more?

    Left: DNF rate against the runner's pace at their *other* finishes (log pace relative to the median of that course and gender, averaged, in deciles), for starts whose runner finished somewhere else. A rising line is selection in the raw data, before any model. Right: each course's DNF rate (reporting editions) against its runner-adjusted pace effect from 1.2.
    """)
    return


@app.cell
def _(
    course_effects_12,
    course_x,
    courses,
    f_runner,
    finishers,
    ftm,
    log_pace,
    np,
    pd,
    plt,
    s_runner,
    starts,
):
    _rel = pd.Series(log_pace) - pd.Series(log_pace).groupby([finishers['name'].to_numpy(), finishers['gender'].to_numpy()]).transform('median')
    _by_runner_sum = _rel.groupby(f_runner).sum()
    _by_runner_n = _rel.groupby(f_runner).size()
    # leave out the start's own finish (if it finished, its pace is in the sum)
    _s = starts.assign(runner=s_runner)
    _own = pd.Series(np.nan, index=_s.index)
    _fin_key = pd.Series(_rel.to_numpy(), index=pd.MultiIndex.from_arrays([finishers['participant_id'], finishers['event_distance_id']]))
    _fin_key = _fin_key[~_fin_key.index.duplicated()]
    _own = _fin_key.reindex(pd.MultiIndex.from_arrays([_s['participant_id'], _s['event_distance_id']])).to_numpy()
    _tot = _by_runner_sum.reindex(s_runner).to_numpy() - np.nan_to_num(_own)
    _cnt = _by_runner_n.reindex(s_runner).fillna(0).to_numpy() - (~np.isnan(_own))
    _s['pace_elsewhere'] = np.where(_cnt > 0, _tot / np.maximum(_cnt, 1), np.nan)
    _s = _s.dropna(subset=['pace_elsewhere'])
    _s['decile'] = pd.qcut(_s['pace_elsewhere'], 10, labels=False)
    _by_dec = _s.groupby('decile').agg(pace=('pace_elsewhere', 'mean'), dnf=('finished', lambda f: 1 - f.mean()), n=('finished', 'size'))

    _course = starts.groupby('name').agg(dnf=('finished', lambda f: 1 - f.mean()), n=('finished', 'size'))
    _course['pace_effect'] = course_effects_12['runner_adjusted'].reindex(_course.index)
    _course['km'] = np.exp(course_x[courses.searchsorted(_course.index)]) * ftm.REFERENCE_KM

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    _ax1.plot(_by_dec['pace'], _by_dec['dnf'], 'o-', color='#2A9D8F')
    _ax1.set(xlabel='Relative log pace at the runner\'s other finishes (decile means)', ylabel='DNF rate',
             title=f'DNF rate by runner pace elsewhere ({len(_s):,} starts)', ylim=(0, None))
    _sc = _ax2.scatter(_course['pace_effect'], _course['dnf'], s=_course['n'] / 30, c=np.log(_course['km']), cmap='viridis', alpha=0.8)
    _ax2.set(xlabel='Course pace effect (1.2, runner-adjusted)', ylabel='DNF rate (reporting editions)', title='Course DNF rate vs pace effect (color: log distance)', ylim=(0, None))
    for _ax in (_ax1, _ax2):
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model

    The finish-time part and its priors are 1.2's. The DNF part's priors are fixed in advance and weakly informative on the logit scale:

    | Parameter | Prior | Meaning |
    |---|---|---|
    | `dnf_mean` $\mu$ | $N(-1.4, 1)$ | logit DNF probability at 50 km, age 40, average course and runner (about 20%) |
    | `dnf_distance` $\lambda$ | $N(0, 1)$ | change in logit per unit log distance |
    | `dnf_gender_gap` $\delta$ | $N(0, 0.5)$ | women minus men |
    | `dnf_age_linear`, `dnf_age_quadratic` | $N(0, 0.5)$, $N(0, 0.25)$ | per decade from 40 |
    | `tau_dnf` $\tau_d$, `omega_dnf` $\omega_d$ | $\text{HalfNormal}(1)$ | spread of course and runner DNF tendencies (logit) |
    | `rho_course`, `rho_runner` | $2\,\text{Beta}(2, 2) - 1$ | correlation of DNF with pace effect (the LKJ(2) prior for a 2x2 correlation) |
    | `dnf_runner_pace_quadratic` $q$ | $N(0, 0.5)$ | curvature of the DNF logit in the runner's standardized pace effect, $(v_r/\omega)^2 - 1$ |

    Parameterization, chosen for sampling: each course's DNF level is a direct parameter (centered, given its pace effect), like the course means; runner effects are non-centered and sum to zero within gender, as in 1.2. `build_dnf_model(runner_dnf=...)` also builds the comparison models: a linear pace relation only (`'correlated'`), runner DNF effects independent of pace (`'independent'`), or none (`'none'`).
    """)
    return


@app.cell
def _(
    course_x,
    courses,
    dnf,
    f_age,
    f_age2,
    f_course,
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
):
    _avg = np.eye(2)[runner_gender] / np.bincount(runner_gender, minlength=2)
    _f_half = np.array([-0.5, 0.5])[f_gender]
    _s_half = np.array([-0.5, 0.5])[s_gender]


    def _zero_sum(raw):
        return raw - pm.math.dot(raw, _avg)[runner_gender]


    def build_dnf_model(runner_dnf='quadratic'):
        """1.2's finish-time model plus the DNF part. runner_dnf: 'quadratic' (full), 'correlated', 'independent' or 'none'."""
        with pm.Model(coords={'gender': list(ftm.GENDERS), 'course': courses, 'runner': runners}) as _model:
            # finish times (1.2)
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
            _raw = pm.Normal('runner_raw', 0.0, 1.0, dims='runner')
            _z1 = _zero_sum(_raw)
            _v = pm.Deterministic('runner_effect', _omega * _z1, dims='runner')
            _mu = (_course_mean[f_course] + _f_half * (_pace_gap + _exp_gap * f_x)
                   + _a1[f_gender] * f_age + _a2[f_gender] * f_age2 + _v[f_runner])
            pm.Normal('log_pace', _mu, _sigma[f_gender], observed=log_pace)

            # DNFs
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
            _eta = _level[s_course] + _s_half * _dg + _da1 * s_age + _da2 * s_age**2
            if runner_dnf != 'none':
                _omega_d = pm.HalfNormal('omega_dnf', 1.0)
                _z2 = _zero_sum(pm.Normal('runner_dnf_raw', 0.0, 1.0, dims='runner'))
                if runner_dnf in ('correlated', 'quadratic'):
                    _rho_r = pm.Deterministic('rho_runner', 2 * pm.Beta('rho_runner_01', 2, 2) - 1)
                    _w = _omega_d * (_rho_r * _z1 + pt.sqrt(1 - _rho_r**2) * _z2)
                    if runner_dnf == 'quadratic':
                        # Curvature in the runner's standardized pace (centered so it does not shift
                        # the DNF level): both the fastest and the slowest runners can drop more.
                        _w = _w + pm.Normal('dnf_runner_pace_quadratic', 0.0, 0.5) * (_z1**2 - 1)
                else:
                    _w = _omega_d * _z2
                _eta = _eta + pm.Deterministic('runner_dnf_effect', _w, dims='runner')[s_runner]
            pm.Bernoulli('dnf', logit_p=_eta, observed=dnf)
        return _model


    FINISH_SCALARS = ['pace_50k_mean', 'pace_50k_gap', 'distance_exponent_mean', 'distance_exponent_gap', 'tau', 'omega', 'sigma']
    DNF_SCALARS = ['dnf_mean', 'dnf_distance', 'dnf_gender_gap', 'dnf_age_linear', 'dnf_age_quadratic', 'tau_dnf', 'omega_dnf', 'rho_course', 'rho_runner', 'dnf_runner_pace_quadratic']


    def flat_draws(posterior, name):
        return posterior[name].stack(s=('chain', 'draw')).transpose('s', ...).values


    def dnf_logit(posterior):
        """Logit DNF probability per start for every draw, as (draw, start)."""
        _eta = (flat_draws(posterior, 'course_dnf_level')[:, s_course] + _s_half * flat_draws(posterior, 'dnf_gender_gap')[:, None]
                + flat_draws(posterior, 'dnf_age_linear')[:, None] * s_age + flat_draws(posterior, 'dnf_age_quadratic')[:, None] * s_age**2)
        if 'runner_dnf_effect' in posterior:
            _eta = _eta + flat_draws(posterior, 'runner_dnf_effect')[:, s_runner]
        return _eta


    dnf_model = build_dnf_model()
    dnf_model
    return (
        DNF_SCALARS,
        FINISH_SCALARS,
        build_dnf_model,
        dnf_logit,
        dnf_model,
        flat_draws,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior predictive check

    What the priors imply for DNF rates (the finish-time part's prior check is in 1.2): the overall DNF rate, and the DNF rate per course, from 200 prior draws, against the observed values (red).
    """)
    return


@app.cell
def _(dnf, dnf_model, np, pd, plt, pm, s_course):
    with dnf_model:
        _prior = pm.sample_prior_predictive(draws=200, var_names=['dnf'], random_seed=1)
    _sim = _prior.prior_predictive['dnf'].values.reshape(200, -1)
    _obs_course = pd.Series(dnf).groupby(s_course).mean()
    _sim_course = np.stack([pd.Series(_d).groupby(s_course).mean().to_numpy() for _d in _sim])

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 3.5))
    _ax1.hist(_sim.mean(1), bins=30, range=(0, 1), color='#999999')
    _ax1.axvline(dnf.mean(), color='#E63946', ls='--')
    _ax1.set(title='Overall DNF rate', xlabel='DNF rate', ylabel='Prior draws')
    _ax2.hist(_sim_course.ravel(), bins=40, range=(0, 1), density=True, color='#999999', label='prior')
    _ax2.hist(_obs_course, bins=40, range=(0, 1), density=True, histtype='step', color='#E63946', lw=2, label='observed')
    _ax2.set(title='DNF rate per course', xlabel='DNF rate', ylabel='Density')
    _ax2.legend()
    for _ax in (_ax1, _ax2):
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit

    Runner-level draws are stored as float32 without the raw (standardized) effects, to keep the trace small.
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
    dnf_model,
    f_course,
    f_runner,
    hashlib,
    log_pace,
    mo,
    np,
    runner_gender,
    s_age,
    s_course,
    s_runner,
    sample_lean,
    shutil,
    xr,
):
    _key = hashlib.sha1(b''.join(np.ascontiguousarray(a).tobytes() for a in (log_pace, f_course, f_runner, dnf, s_course, s_runner, s_age))).hexdigest()[:12]
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


    def fit_dnf(model, label, minutes):
        """Sample `model` with nutpie, or load its cached trace (keyed by label, model version and data)."""
        _path = OUT / f'dnf_{label}_v{_MODEL_VERSION}_{_key}.nc'
        if _path.exists():
            return az.from_netcdf(_path)
        _keep = [v.name for v in model.free_RVs + model.deterministics if v.name not in _PER_RUNNER]
        _gb = check_memory(model, _keep, chains=4, draws=DRAWS, on_disk=True)
        mo.output.replace(mo.callout(mo.md(f"Sampling the {label} model since {datetime.now():%H:%M} (about {minutes} minutes, about {_gb:.0f} GB)."), kind='info'))
        _store = _path.with_suffix('.zarr')  # draws stream here, not into memory
        _trace = sample_lean(model, _keep, draws=DRAWS, random_seed=42, progressbar=True, store_dir=_store)
        _post = _derive_runner_effects(_trace.posterior)
        _trace = az.InferenceData(posterior=_post, sample_stats=_trace.sample_stats)
        _trace.to_netcdf(_path)
        shutil.rmtree(_store)
        return az.from_netcdf(_path)


    dnf_trace = fit_dnf(dnf_model, 'quadratic', 13)
    dnf_summary = az.summary(dnf_trace, var_names=FINISH_SCALARS + DNF_SCALARS, round_to=4)
    print(f"Divergences: {int(dnf_trace.sample_stats['diverging'].sum())}")
    _ess = az.ess(dnf_trace, var_names=['course_mean', 'course_dnf_level', 'runner_effect', 'runner_dnf_effect'])
    print('Min bulk ESS: ' + ', '.join(f'{v} {float(_ess[v].min()):.0f}' for v in _ess.data_vars))
    dnf_summary
    return dnf_trace, fit_dnf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Convergence

    Rank plots for the DNF part's population parameters (the finish-time part's match 1.2's); R-hat and ESS for all are in the summary above.
    """)
    return


@app.cell
def _(DNF_SCALARS, az, dnf_trace, np):
    _axes = az.plot_rank(dnf_trace, var_names=DNF_SCALARS, kind='vlines', figsize=(14, 12))
    _fig = np.ravel(_axes)[0].figure
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posterior predictive checks

    DNF outcomes replicated for the same starts from 200 posterior draws, compared with the observed:

    - **DNF rate per course**, which tests the course DNF levels
    - **DNF rate by runner pace** (deciles of the runner's posterior pace effect, for runners with a finish at some other start), which tests the pace-DNF correlation
    - **DNF rate by age and gender**
    - **how DNFs cluster in runners**: among runners with 3+ reporting starts, the share whose every start was a finish, against its replicated distribution, which tests the spread of runner DNF tendencies
    """)
    return


@app.cell
def _(dnf, dnf_logit, dnf_trace, expit, np):
    _sub = dnf_trace.posterior.isel(draw=slice(None, None, 20))
    _rng = np.random.default_rng(7)
    dnf_rep = (_rng.random((_sub.sizes['chain'] * _sub.sizes['draw'], len(dnf))) < expit(dnf_logit(_sub))).astype(np.int8)
    runner_pace_mean = dnf_trace.posterior['runner_effect'].mean(('chain', 'draw')).values
    return dnf_rep, runner_pace_mean


@app.cell
def _(
    course_label,
    dnf,
    dnf_rep,
    f_runner,
    np,
    pd,
    plt,
    runner_pace_mean,
    runners,
    s_course,
    s_runner,
    starts,
):
    def _rates(groups, values):
        _codes, _labels = pd.factorize(groups, sort=True)
        _cnt = np.bincount(_codes)
        if values.ndim == 1:
            return _labels, np.bincount(_codes, weights=values) / _cnt
        return _labels, np.stack([np.bincount(_codes, weights=_v) / _cnt for _v in values])


    def _panel(ax, groups, title, xlabel, xticks=None, mask=None):
        _m = np.ones(len(dnf), bool) if mask is None else mask
        _labels, _obs = _rates(groups[_m], dnf[_m])
        _, _rep = _rates(groups[_m], dnf_rep[:, _m])
        _x = np.arange(len(_labels))
        _lo, _hi = np.percentile(_rep, [3, 97], axis=0)
        ax.vlines(_x, _lo, _hi, color='#2A9D8F', lw=5, alpha=0.5, label='predicted 94%')
        ax.plot(_x, _obs, 'o', ms=4, color='#E63946', label='observed')
        ax.set(title=title, xlabel=xlabel, ylabel='DNF rate')
        if xticks is not None:
            ax.set_xticks(_x, xticks(_labels), rotation=90, fontsize=7)
        ax.legend(fontsize=8)


    _fig, _axes = plt.subplots(2, 2, figsize=(15, 10))
    _order = np.argsort(pd.Series(dnf).groupby(s_course).mean().to_numpy())
    _rank = np.empty_like(_order)
    _rank[_order] = np.arange(len(_order))
    _panel(_axes[0, 0], _rank[s_course], 'DNF rate per course (sorted by observed)', '', lambda l: [str(course_label[_order[i]])[:30] for i in l])
    # Only starts whose runner finished some *other* start: a runner with no finish has a pace
    # effect informed only through the correlation, and selecting on any finish would select on
    # this start's own outcome.
    _has_finish = np.bincount(f_runner, minlength=len(runners))[s_runner] - (1 - dnf) > 0
    _decile = np.full(len(dnf), -1)
    _decile[_has_finish] = pd.qcut(runner_pace_mean[s_runner][_has_finish], 10, labels=False)
    _panel(_axes[0, 1], _decile, 'DNF rate by runner pace effect (runners with another finish; deciles, fast to slow)', 'Decile of runner pace effect', mask=_has_finish)
    _age_band = np.clip((starts['age'].to_numpy() // 10) * 10, 20, 70)
    _panel(_axes[1, 0], np.char.add(starts['gender'].to_numpy().astype(str), _age_band.astype(str)), 'DNF rate by gender and age decade', '', lambda l: list(l))
    _multi = np.bincount(s_runner, minlength=len(runners)) >= 3
    _in = _multi[s_runner]


    def _share_clean(d):
        _dnfs = np.bincount(s_runner[_in], weights=d[_in], minlength=len(runners))[_multi]
        return (_dnfs == 0).mean()


    _rep_share = np.array([_share_clean(_d) for _d in dnf_rep])
    _axes[1, 1].hist(_rep_share, bins=30, color='#2A9D8F', alpha=0.6, label='predicted')
    _axes[1, 1].axvline(_share_clean(dnf), color='#E63946', lw=2, label='observed')
    _axes[1, 1].set(title=f'Runners with 3+ starts whose every start was a finish ({int(_multi.sum()):,} runners)', xlabel='Share of runners', ylabel='Replications')
    _axes[1, 1].legend(fontsize=8)
    for _ax in _axes.flat:
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the checks show.** Course DNF rates, age and gender, and how DNFs cluster within runners all fit. A linear relation between runner pace and DNF (the `correlated` model) missed a U-shape: the fastest runners drop more often than the next few deciles, and the slowest more than a straight line allows. The curvature term $q$ fixes both ends; the model still over-predicts DNFs by 1 to 3 points for runners just faster than average (deciles 2 to 5), so the true relation is flatter there than a quadratic. A spline in the pace effect would close that gap; it is small next to the effects themselves.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model comparison

    PSIS-LOO on the DNF outcomes (leave one start out; the finish-time part is identical in all four), 500 draws per model:

    - **quadratic** (full): runner DNF tendency correlated with runner pace, with curvature
    - **correlated**: the same, linear in pace
    - **independent**: runner DNF tendency, unrelated to pace
    - **none**: no runner DNF tendency

    On a final-fit core (`CORE` other than `'core'`) only the full model is fitted, so this table has one row; the comparison is the dev core's.
    """)
    return


@app.cell
def _(CORE, build_dnf_model, dnf_trace, fit_dnf):
    # On the dev core every variant is fitted for the comparison; on a final-fit core only the full
    # model (hours each), and the comparison below is the dev core's (see its cached traces).
    if CORE == 'core':
        dnf_comparison_traces = {
            'quadratic': dnf_trace,
            'correlated': fit_dnf(build_dnf_model('correlated'), 'correlated', 13),
            'independent': fit_dnf(build_dnf_model('independent'), 'independent', 13),
            'none': fit_dnf(build_dnf_model('none'), 'none', 8),
        }
    else:
        dnf_comparison_traces = {'quadratic': dnf_trace}
    return (dnf_comparison_traces,)


@app.cell
def _(az, dnf, dnf_comparison_traces, dnf_logit, np, xr):
    def _with_loglik(trace):
        _post = trace.posterior.isel(draw=slice(None, None, 8))
        _eta = dnf_logit(_post)
        _ll = np.where(dnf == 1, -np.logaddexp(0, -_eta), -np.logaddexp(0, _eta)).astype(np.float32)
        _da = xr.DataArray(_ll.reshape(_post.sizes['chain'], _post.sizes['draw'], -1), dims=('chain', 'draw', 'start'),
                           coords={'chain': _post['chain'], 'draw': _post['draw']})
        return az.InferenceData(posterior=_post, log_likelihood=xr.Dataset({'dnf': _da}))


    dnf_loo_data = {_name: _with_loglik(_t) for _name, _t in dnf_comparison_traces.items()}
    dnf_loos = {_name: az.loo(_d, pointwise=True) for _name, _d in dnf_loo_data.items()}
    dnf_comparison = az.compare(dnf_loos)[['rank', 'elpd_loo', 'p_loo', 'elpd_diff', 'se', 'dse']].assign(
        starts_k_over_0_7=lambda d: [int((dnf_loos[_m].pareto_k > 0.7).sum()) for _m in d.index]
    )
    dnf_comparison.round(1)
    return (dnf_loo_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior sensitivity

    Power-scaling sensitivity of the population parameters to their own priors (the priors we chose; the standardized effects belong to the hierarchy). Values above 0.05 mean the prior shapes that parameter. As in 1.2, expect the population pace curve to be prior-sensitive with 28 courses.
    """)
    return


@app.cell
def _(
    DNF_SCALARS,
    FINISH_SCALARS,
    az,
    dnf_loo_data,
    dnf_model,
    ftm,
    np,
    pd,
    pm,
    pytensor,
    xr,
):
    _idata = dnf_loo_data['quadratic'].copy()
    _scalars = FINISH_SCALARS + DNF_SCALARS
    _chosen = [v for v in ['pace_50k_mean', 'pace_50k_gap', 'distance_exponent_mean', 'distance_exponent_gap', 'age_linear', 'age_quadratic',
                           'tau', 'omega', 'sigma', 'dnf_mean', 'dnf_distance', 'dnf_gender_gap', 'dnf_age_linear', 'dnf_age_quadratic',
                           'tau_dnf', 'omega_dnf', 'rho_course_01', 'rho_runner_01', 'dnf_runner_pace_quadratic']]
    # Log density of each chosen prior at the posterior draws. pm.compute_log_prior needs every
    # free variable, and the stored trace drops the raw runner effects; these priors all have
    # fixed parameters, so each is evaluated on its own.
    _log_prior = {}
    for _v in _chosen:
        _rv = dnf_model[_v]
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
    dnf_sensitivity = pd.DataFrame(_rows).set_index('parameter')
    dnf_sensitivity['diagnosis'] = np.where(dnf_sensitivity['prior sensitivity'] >= 0.05, 'prior-sensitive', 'data-dominated')
    dnf_sensitivity.round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results

    **Left**: each course's DNF effect (logit, after distance) against its pace effect, with 94% intervals; the correlation $\rho_c$ is in the title. **Middle**: DNF probability against a runner's pace effect, for a 40-year-old at an average 50 km and 100 mile course, through the runner-level correlation and curvature (94% band). **Right**: DNF probability by distance and age for an average runner and course.
    """)
    return


@app.cell
def _(course_label, dnf_trace, expit, flat_draws, ftm, np, plt):
    _p = dnf_trace.posterior
    _rc = flat_draws(_p, 'rho_course')
    _rr = flat_draws(_p, 'rho_runner')
    _u = _p['course_effect']
    _w = _p['course_dnf_effect']
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5.5))
    _um, _wm = _u.mean(('chain', 'draw')).values, _w.mean(('chain', 'draw')).values
    _ulo, _uhi = np.percentile(_u.stack(s=('chain', 'draw')).values, [3, 97], axis=1)
    _wlo, _whi = np.percentile(_w.stack(s=('chain', 'draw')).values, [3, 97], axis=1)
    _axes[0].errorbar(_um, _wm, xerr=[_um - _ulo, _uhi - _um], yerr=[_wm - _wlo, _whi - _wm], fmt='o', color='#2A9D8F', alpha=0.7, ms=5)
    for _i in np.argsort(np.abs(_wm))[-8:]:
        _axes[0].annotate(str(course_label[_i])[:26], (_um[_i], _wm[_i]), fontsize=7, xytext=(4, 2), textcoords='offset points')
    _axes[0].axhline(0, color='gray', lw=1)
    _axes[0].axvline(0, color='gray', lw=1)
    _axes[0].set(xlabel='Course pace effect (log pace)', ylabel='Course DNF effect (logit)',
                 title=f'Courses: rho = {np.median(_rc):.2f} (94% {np.percentile(_rc, 3):.2f} to {np.percentile(_rc, 97):.2f})')

    _om, _omd = flat_draws(_p, 'omega'), flat_draws(_p, 'omega_dnf')
    _q = flat_draws(_p, 'dnf_runner_pace_quadratic')
    _v = np.linspace(-2, 2, 41)
    for _km, _color in [(50, '#2A9D8F'), (160.9, '#E76F51')]:
        _base = flat_draws(_p, 'dnf_mean') + flat_draws(_p, 'dnf_distance') * np.log(_km / ftm.REFERENCE_KM)
        # E[DNF logit | v] = rho * omega_d * v + q * (v^2 - 1), v in SDs of runner ability
        _prob = expit(_base[:, None] + (_rr * _omd)[:, None] * _v[None, :] + _q[:, None] * (_v[None, :] ** 2 - 1))
        _axes[1].fill_between(_v, *np.percentile(_prob, [3, 97], axis=0), color=_color, alpha=0.3)
        _axes[1].plot(_v, np.median(_prob, axis=0), color=_color, label=f'{_km:g} km')
    _axes[1].set(xlabel='Runner pace effect (SDs of runner ability; positive = slower)', ylabel='P(DNF)',
                 title=f'Runners: rho = {np.median(_rr):.2f}, curvature q = {np.median(_q):.2f} (94% {np.percentile(_q, 3):.2f} to {np.percentile(_q, 97):.2f})', ylim=(0, None))
    _axes[1].legend()

    _km = np.geomspace(42.2, 160.9, 40)
    for _age, _ls in [(30, ':'), (40, '-'), (60, '--')]:
        _a = (_age - 40) / 10
        for _gi, (_g, _color) in enumerate([('M', 'steelblue'), ('F', 'coral')]):
            _eta = (flat_draws(_p, 'dnf_mean')[:, None] + flat_draws(_p, 'dnf_distance')[:, None] * np.log(_km / ftm.REFERENCE_KM)
                    + (_gi - 0.5) * flat_draws(_p, 'dnf_gender_gap')[:, None] + flat_draws(_p, 'dnf_age_linear')[:, None] * _a
                    + flat_draws(_p, 'dnf_age_quadratic')[:, None] * _a**2)
            _axes[2].plot(_km, np.median(expit(_eta), axis=0), color=_color, ls=_ls, label=f'{_g}, age {_age}')
    _axes[2].set(xscale='log', xlabel='Distance', ylabel='P(DNF)', title='DNF probability, average runner and course', ylim=(0, None))
    _axes[2].set_xticks([42.2, 50, 80.5, 100, 160.9], ['marathon', '50K', '50 mi', '100K', '100 mi'])
    _axes[2].xaxis.set_minor_formatter(plt.NullFormatter())
    _axes[2].legend(fontsize=8, ncol=2)
    for _ax in _axes:
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Handoff

    Saved to `data/cache/2.2_dnf_observed/`: course effects (`course_effects.parquet`: pace and DNF effects, posterior mean and SD) and runner effects (`runner_effects.parquet`: pace and DNF effects, the DNF effect including the pace curvature, with each runner's finishes and reporting starts). 2.3 extends this model to editions that do not report DNFs.
    """)
    return


@app.cell
def _(
    OUT,
    course_label,
    course_x,
    courses,
    dnf,
    dnf_trace,
    f_runner,
    ftm,
    np,
    pd,
    runner_gender,
    runners,
    s_course,
    s_runner,
):
    _p = dnf_trace.posterior
    _m = lambda v: _p[v].mean(('chain', 'draw')).values
    _sd = lambda v: _p[v].std(('chain', 'draw')).values
    dnf_course_effects = pd.DataFrame({
        'course_id': courses, 'label': course_label, 'distance_km': np.exp(course_x) * ftm.REFERENCE_KM,
        'pace_effect_mean': _m('course_effect'), 'pace_effect_sd': _sd('course_effect'),
        'dnf_effect_mean': _m('course_dnf_effect'), 'dnf_effect_sd': _sd('course_dnf_effect'),
        'reporting_starts': np.bincount(s_course, minlength=len(courses)),
        'dnf_rate': np.bincount(s_course, weights=dnf, minlength=len(courses)) / np.maximum(np.bincount(s_course, minlength=len(courses)), 1),
    })
    dnf_runner_effects = pd.DataFrame({
        'participant_id': runners, 'gender': np.array(ftm.GENDERS)[runner_gender],
        'finishes': np.bincount(f_runner, minlength=len(runners)), 'reporting_starts': np.bincount(s_runner, minlength=len(runners)),
        'pace_effect_mean': _m('runner_effect'), 'pace_effect_sd': _sd('runner_effect'),
        'dnf_effect_mean': _m('runner_dnf_effect'), 'dnf_effect_sd': _sd('runner_dnf_effect'),
    })
    dnf_course_effects.to_parquet(OUT / 'course_effects.parquet', index=False)
    dnf_runner_effects.to_parquet(OUT / 'runner_effects.parquet', index=False)
    dnf_course_effects.sort_values('dnf_effect_mean').round(3)
    return


if __name__ == "__main__":
    app.run()
