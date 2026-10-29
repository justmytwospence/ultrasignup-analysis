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
    # Finish Times with Runner Effects

    The course-only model (1.1) cannot tell a hard course from a strong field: a course that draws fast runners looks easy. This model adds a runner effect, each runner's typical pace relative to others of the same gender and age, so a course effect means "harder than its runners would predict".

    $$\log \text{pace}_i = m_c + \tfrac{\pm 1}{2}\big(\delta_\alpha + \delta_\beta\, x_c\big) + \gamma_{1g} A_i + \gamma_{2g} A_i^2 + v_r + \varepsilon_i$$

    $$m_c \sim N(\alpha + \beta x_c, \tau^2), \qquad v_r \sim N(0, \omega^2), \qquad \varepsilon_i \sim N(0, \sigma_g^2)$$

    with $x_c = \log(d_c / 50\,\text{km})$, $A_i = (\text{age}_i - 40)/10$, and $\pm$ for women and men. Everything else matches 1.1, so the only new idea is $v_r$.

    Runner effects are only identified for runners who ran several courses, so the model fits 0.1's dense core: every finisher at its 28 courses (runners with one course still inform the course effects, pooled toward zero). Runners span courses, so unlike 1.1 the effects cannot be integrated out and are sampled.

    What it still leaves out: edition-to-edition variation, a runner's improvement or decline over time beyond age, and DNFs.
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
    import hashlib
    from datetime import datetime
    from pathlib import Path

    import arviz as az
    import numpy as np
    import pandas as pd
    import pymc as pm
    import pytensor
    import scipy.sparse as sp
    import xarray as xr
    import matplotlib.pyplot as plt

    from utils import finish_time_model as ftm
    from utils.data_processing import load_results, process_results

    CACHE = Path(__file__).resolve().parent.parent / 'data' / 'cache'

    # Which 0.1 core to fit: 'core' is the dev core (3, 840), 28 courses, minutes;
    # 'core_3_500' is the final-fit core, 172 courses, hours (fewer draws to bound memory).
    CORE = 'core_3_500'
    DRAWS = 1000 if CORE == 'core' else 500

    return (
        CACHE,
        CORE,
        DRAWS,
        az,
        datetime,
        ftm,
        hashlib,
        load_results,
        np,
        pd,
        plt,
        pm,
        process_results,
        pytensor,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data

    Every finisher at the courses of 0.1's core (`CORE`) (`0.1_kcore/core.parquet`), with 1.1's course effects (`1.1_finish_courses/course_effects.parquet`) for comparison.
    """)
    return


@app.cell
def _(CACHE, CORE, ftm, load_results, pd, process_results):
    _core = pd.read_parquet(CACHE / '0.1_kcore' / f'{CORE}.parquet')
    _results = process_results(load_results())
    finishers = _results[_results['finished'] & _results['name'].isin(_core['name'].unique())].reset_index(drop=True)
    nb1_effects = pd.read_parquet(CACHE / '1.1_finish_courses' / 'course_effects.parquet').set_index('course_id')

    _z, log_pace = ftm.design(finishers)
    x_finisher, age_z, age_z2 = _z[:, 1], _z[:, 2], _z[:, 3]
    gender_idx = (finishers['gender'] == 'F').to_numpy().astype(int)
    course_idx, courses = pd.factorize(finishers['name'], sort=True)
    runner_idx, runners = pd.factorize(finishers['participant_id'])
    course_x = pd.Series(x_finisher).groupby(course_idx).first().to_numpy()
    runner_gender = pd.Series(gender_idx).groupby(runner_idx).first().to_numpy()
    runner_courses = finishers.groupby(runner_idx)['name'].nunique().to_numpy()
    course_label = nb1_effects['label'].reindex(courses).to_numpy()

    pd.DataFrame({
        'finishers': [len(finishers)],
        'courses': [len(courses)],
        'runners': [len(runners)],
        'runners with 2+ courses': [int((runner_courses >= 2).sum())],
        'runners with 5+ courses': [int((runner_courses >= 5).sum())],
    }).T.rename(columns={0: 'count'})
    return (
        age_z,
        age_z2,
        course_idx,
        course_label,
        course_x,
        courses,
        finishers,
        gender_idx,
        log_pace,
        nb1_effects,
        runner_courses,
        runner_gender,
        runner_idx,
        runners,
        x_finisher,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Do runners have a consistent pace?

    For runners who finished at least three of these courses: their pace at one race relative to that course and gender's median, against their average relative pace at their other races. A strong relation means a runner's pace carries across courses, which is what the runner effect captures.
    """)
    return


@app.cell
def _(finishers, log_pace, np, pd, plt, runner_courses, runner_idx):
    _rel = pd.Series(log_pace)
    _rel = _rel - _rel.groupby([finishers['name'].to_numpy(), finishers['gender'].to_numpy()]).transform('median')
    _df = pd.DataFrame({'runner': runner_idx, 'rel': _rel.to_numpy()})
    _df = _df[runner_courses[_df['runner']] >= 3]
    _sum = _df.groupby('runner')['rel'].transform('sum')
    _cnt = _df.groupby('runner')['rel'].transform('size')
    _df['others'] = (_sum - _df['rel']) / (_cnt - 1)
    _sample = _df.sample(min(len(_df), 20000), random_state=1)
    _r = np.corrcoef(_df['rel'], _df['others'])[0, 1]

    _fig, _ax = plt.subplots(figsize=(6, 5))
    _ax.scatter(_sample['others'], _sample['rel'], s=2, alpha=0.15, color='#2A9D8F')
    _ax.plot([-0.6, 0.6], [-0.6, 0.6], color='gray', lw=1)
    _ax.set(xlim=(-0.6, 0.6), ylim=(-0.8, 0.8), xlabel='Mean relative log pace at their other races',
            ylabel='Relative log pace at this race', title=f'Runner consistency across courses (r = {_r:.2f})')
    _ax.grid(alpha=0.3)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model

    Priors match 1.1 where the parameter is the same, and are fixed in advance:

    | Parameter | Prior | Meaning |
    |---|---|---|
    | `pace_50k_mean` $\alpha$ | $N(\log 8, 0.3)$ | mean log min/km at 50 km, age 40, average course and runner |
    | `pace_50k_gap` $\delta_\alpha$ | $N(0, 0.2)$ | women minus men |
    | `distance_exponent_mean` $\beta$, `_gap` $\delta_\beta$ | $N(0.1, 0.1)$, $N(0, 0.1)$ | distance exponent and its gender gap |
    | `age_linear`, `age_quadratic` | $N(0, 0.1)$, $N(0, 0.05)$ | per decade from 40, per gender |
    | `tau` | $\text{HalfNormal}(0.3)$ | spread of course difficulty |
    | `omega` | $\text{HalfNormal}(0.3)$ | spread of runner ability |
    | `sigma` | $\text{HalfNormal}(0.3)$ | spread of a runner's results around their own level, per gender |

    Parameterization, chosen for sampling: course means are centered (each course has thousands of finishers), runner effects are non-centered and sum to zero within each gender, so they cannot trade off against the gender gap or the course means (with about 40,000 runners this barely changes the prior). `build_runner_model(runner_effects=False)` is the same model without $v_r$, on the same data, for the comparison.
    """)
    return


@app.cell
def _(
    age_z,
    age_z2,
    course_idx,
    course_x,
    courses,
    ftm,
    gender_idx,
    log_pace,
    np,
    pm,
    runner_gender,
    runner_idx,
    runners,
    x_finisher,
):
    def build_runner_model(runner_effects=True):
        """Finish-time model on the core data, with or without runner effects."""
        _half = np.array([-0.5, 0.5])[gender_idx]
        _coords = {'gender': list(ftm.GENDERS), 'course': courses}
        if runner_effects:
            _coords['runner'] = runners
        with pm.Model(coords=_coords) as _model:
            _pace = pm.Normal('pace_50k_mean', np.log(8.0), 0.3)
            _pace_gap = pm.Normal('pace_50k_gap', 0.0, 0.2)
            _exp = pm.Normal('distance_exponent_mean', 0.1, 0.1)
            _exp_gap = pm.Normal('distance_exponent_gap', 0.0, 0.1)
            _a1 = pm.Normal('age_linear', 0.0, 0.1, dims='gender')
            _a2 = pm.Normal('age_quadratic', 0.0, 0.05, dims='gender')
            _tau = pm.HalfNormal('tau', 0.3)
            _sigma = pm.HalfNormal('sigma', 0.3, dims='gender')
            _course_mean = pm.Normal('course_mean', _pace + _exp * course_x, _tau, dims='course')
            pm.Deterministic('course_effect', _course_mean - (_pace + _exp * course_x), dims='course')
            _mu = (_course_mean[course_idx] + _half * (_pace_gap + _exp_gap * x_finisher)
                   + _a1[gender_idx] * age_z + _a2[gender_idx] * age_z2)
            if runner_effects:
                _omega = pm.HalfNormal('omega', 0.3)
                _raw = pm.Normal('runner_raw', 0.0, 1.0, dims='runner')
                _avg = np.eye(2)[runner_gender] / np.bincount(runner_gender, minlength=2)
                _v = pm.Deterministic('runner_effect', _omega * (_raw - pm.math.dot(_raw, _avg)[runner_gender]), dims='runner')
                _mu = _mu + _v[runner_idx]
            pm.Normal('log_pace', _mu, _sigma[gender_idx], observed=log_pace)
        return _model


    SCALARS = ['pace_50k_mean', 'pace_50k_gap', 'distance_exponent_mean', 'distance_exponent_gap',
               'age_linear', 'age_quadratic', 'tau', 'omega', 'sigma']


    def predicted_mean(posterior):
        """Expected log pace per finisher for every draw in `posterior`, as (draw, finisher)."""
        _f = lambda v: posterior[v].stack(s=('chain', 'draw')).transpose('s', ...).values
        _half = np.array([-0.5, 0.5])[gender_idx]
        _mu = (_f('course_mean')[:, course_idx] + _half * (_f('pace_50k_gap')[:, None] + _f('distance_exponent_gap')[:, None] * x_finisher)
               + _f('age_linear')[:, gender_idx] * age_z + _f('age_quadratic')[:, gender_idx] * age_z2)
        if 'runner_effect' in posterior:
            _mu = _mu + _f('runner_effect')[:, runner_idx]
        return _mu, _f('sigma')[:, gender_idx]


    runner_model = build_runner_model()
    runner_model
    return SCALARS, build_runner_model, predicted_mean, runner_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior predictive check

    Finish times the priors imply for these finishers (gray: 1st to 99th percentile across 200 prior draws, per distance band), against the observed range (red). The priors should cover the data without concentrating on it.
    """)
    return


@app.cell
def _(finishers, np, pd, plt, pm, runner_model):
    with runner_model:
        _prior = pm.sample_prior_predictive(draws=200, random_seed=1)
    _hours = np.exp(_prior.prior_predictive['log_pace'].values.reshape(200, -1)) * finishers['distance_km'].to_numpy() / 60
    _bands = pd.cut(finishers['distance_km'], [42, 46, 60, 85, 110, 165], labels=['marathon', '50K', '50 mi', '100K', '100 mi'], include_lowest=True)
    _obs_hours = finishers['time_ms'].to_numpy() / 3.6e6

    _fig, _ax = plt.subplots(figsize=(9, 3.5))
    for _i, _b in enumerate(_bands.cat.categories):
        _m = (_bands == _b).to_numpy()
        if not _m.any():
            continue
        _lo, _hi = np.percentile(_hours[:, _m], [1, 99])
        _ax.plot([_i - 0.1, _i - 0.1], [_lo, _hi], color='#999999', lw=6, solid_capstyle='butt', label='prior 1-99%' if _i == 0 else None)
        _olo, _ohi = np.percentile(_obs_hours[_m], [1, 99])
        _ax.plot([_i + 0.1, _i + 0.1], [_olo, _ohi], color='#E63946', lw=6, solid_capstyle='butt', label='observed 1-99%' if _i == 0 else None)
    _ax.set(yscale='log', xticks=range(5), xticklabels=list(_bands.cat.categories), ylabel='Finish time (hours)', title='Prior predictive finish times')
    _ax.legend()
    _ax.grid(alpha=0.3, axis='y')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit
    """)
    return


@app.cell
def _(
    CACHE,
    DRAWS,
    SCALARS,
    age_z,
    az,
    course_idx,
    datetime,
    gender_idx,
    hashlib,
    log_pace,
    mo,
    np,
    pm,
    runner_idx,
    runner_model,
):
    _key = hashlib.sha1(b''.join(np.ascontiguousarray(a).tobytes() for a in (log_pace, course_idx, runner_idx, gender_idx, age_z))).hexdigest()[:12]
    _cache_dir = CACHE / '1.2_finish_runners'
    _cache_dir.mkdir(parents=True, exist_ok=True)
    # Bump when the model changes, so a cached trace is never reused for a different model.
    _MODEL_VERSION = 1


    def fit_runner(model, label, minutes):
        """Sample `model` with nutpie, or load its cached trace (keyed by label, model version and data)."""
        _path = _cache_dir / f'runner_{label}_v{_MODEL_VERSION}_{_key}.nc'
        if _path.exists():
            return az.from_netcdf(_path)
        mo.output.replace(mo.callout(mo.md(f"Sampling the {label} model since {datetime.now():%H:%M} (about {minutes} minutes)."), kind='info'))
        with model:
            _trace = pm.sample(draws=DRAWS, tune=1000, chains=4, nuts_sampler='nutpie', random_seed=42, progressbar=True)
        # store runner effects as float32 without the raw (standardized) ones, to keep the trace small
        _post = _trace.posterior.drop_vars([v for v in ('runner_raw',) if v in _trace.posterior])
        if 'runner_effect' in _post:
            _post['runner_effect'] = _post['runner_effect'].astype(np.float32)
        _trace = az.InferenceData(posterior=_post, sample_stats=_trace.sample_stats)
        _trace.to_netcdf(_path)
        return _trace


    runner_trace = fit_runner(runner_model, 'full', 5)
    runner_summary = az.summary(runner_trace, var_names=SCALARS, round_to=4)
    print(f"Divergences: {int(runner_trace.sample_stats['diverging'].sum())}")
    _ess = az.ess(runner_trace, var_names=['course_mean', 'runner_effect'])
    print(f"Min bulk ESS: course means {float(_ess['course_mean'].min()):.0f}, runner effects {float(_ess['runner_effect'].min()):.0f}")
    runner_summary
    return fit_runner, runner_trace


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Convergence

    Rank plots for the population parameters; R-hat and ESS are in the summary above.
    """)
    return


@app.cell
def _(SCALARS, az, np, runner_trace):
    _axes = az.plot_rank(runner_trace, var_names=SCALARS, kind='vlines', figsize=(14, 12))
    _fig = np.ravel(_axes)[0].figure
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posterior predictive checks

    From 200 posterior draws, finish times replicated for the same finishers:

    - **distribution of log pace by distance band** (quantiles), which tests the population curve and the noise
    - **within-runner spread**: for runners with 3+ finishes, the SD of their residuals around their own fitted level, observed against replicated; this is the direct test of a constant runner effect
    - **residual by year** (not in the model)
    """)
    return


@app.cell
def _(
    log_pace,
    np,
    pd,
    predicted_mean,
    runner_courses,
    runner_idx,
    runner_trace,
):
    _sub = runner_trace.posterior.isel(draw=slice(None, None, 20))  # 4 chains x 50 draws
    _mu, _sd = predicted_mean(_sub)
    _rng = np.random.default_rng(5)
    ppc_rep = _mu + _rng.normal(0, _sd)
    ppc_resid_obs = log_pace[None, :] - _mu
    ppc_resid_rep = ppc_rep - _mu
    _multi = runner_courses[runner_idx] >= 3
    _grp = pd.Series(runner_idx[_multi])


    def _within_sd(resid):
        return np.stack([pd.Series(r[_multi]).groupby(_grp.to_numpy()).std().to_numpy() for r in resid])


    within_sd_obs = _within_sd(ppc_resid_obs)
    within_sd_rep = _within_sd(ppc_resid_rep)
    return ppc_rep, ppc_resid_obs, within_sd_obs, within_sd_rep


@app.cell
def _(
    finishers,
    log_pace,
    np,
    pd,
    plt,
    ppc_rep,
    ppc_resid_obs,
    within_sd_obs,
    within_sd_rep,
):
    _bands = pd.cut(finishers['distance_km'], [42, 46, 60, 85, 110, 165], labels=['marathon', '50K', '50 mi', '100K', '100 mi'], include_lowest=True)
    _qs = [5, 25, 50, 75, 95]
    _fig, _axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for _i, _b in enumerate(_bands.cat.categories):
        _m = (_bands == _b).to_numpy()
        if not _m.any():
            continue
        _rep_q = np.percentile(ppc_rep[:, _m], _qs, axis=1)
        _obs_q = np.percentile(log_pace[_m], _qs)
        for _j, _q in enumerate(_qs):
            _axes[0].plot([_i] * 2, np.percentile(_rep_q[_j], [3, 97]), color='#2A9D8F', lw=6, alpha=0.4, solid_capstyle='butt')
            _axes[0].plot(_i, _obs_q[_j], 'o', ms=4, color='#E63946')
    _axes[0].set(xticks=range(5), xticklabels=list(_bands.cat.categories), ylabel='Log pace (min/km)',
                 title='Log pace quantiles (5, 25, 50, 75, 95%)\npredicted 94% (bars) vs observed (dots)')
    _bins = np.linspace(0, 0.4, 41)
    _x = (_bins[1:] + _bins[:-1]) / 2
    _h_rep = np.array([np.histogram(d[~np.isnan(d)], bins=_bins, density=True)[0] for d in within_sd_rep])
    _h_obs = np.median([np.histogram(d[~np.isnan(d)], bins=_bins, density=True)[0] for d in within_sd_obs], axis=0)
    _axes[1].fill_between(_x, *np.percentile(_h_rep, [3, 97], axis=0), color='#2A9D8F', alpha=0.3, label='predicted 94%')
    _axes[1].plot(_x, _h_obs, 'o', ms=3, color='#E63946', label='observed')
    _axes[1].set(title='Within-runner residual SD (runners with 3+ finishes)', xlabel='SD of residual log pace', ylabel='Density')
    _axes[1].legend()
    _mean_resid = ppc_resid_obs.mean(0)
    for _gname, _color in [('M', 'steelblue'), ('F', 'coral')]:
        _m = (finishers['gender'] == _gname).to_numpy()
        _s = pd.Series(_mean_resid[_m]).groupby(finishers.loc[_m, 'year'].to_numpy())
        _mean = _s.mean()[_s.size() >= 100]
        _axes[2].plot(_mean.index, _mean.values, marker='o', ms=3, color=_color, label=_gname)
    _axes[2].axhline(0, color='gray', lw=1)
    _axes[2].set(title='Mean residual by year (not in the model)', xlabel='Year', ylabel='Mean residual (log pace)')
    _axes[2].legend()
    for _ax in _axes:
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model comparison

    PSIS-LOO per finish (leave one result out, predict it from the runner's other results and the course), for the model with and without runner effects on the same data, 500 draws per model. For a runner with a single finish, leaving it out leaves nothing to estimate their effect from, so that finish's predictive is computed exactly with the runner effect integrated out; PSIS cannot handle those (each one alone determines a parameter).
    """)
    return


@app.cell
def _(build_runner_model, fit_runner, runner_trace):
    runner_comparison_traces = {
        'runner effects': runner_trace,
        'no runner effects': fit_runner(build_runner_model(runner_effects=False), 'no_runner', 1),
    }
    return (runner_comparison_traces,)


@app.cell
def _(
    az,
    log_pace,
    np,
    predicted_mean,
    runner_comparison_traces,
    runner_idx,
    xr,
):
    def _with_loglik(trace):
        _post = trace.posterior.isel(draw=slice(None, None, 8))
        _mu, _sd = predicted_mean(_post)
        if 'runner_effect' in _post:
            # A runner with a single finish: leaving it out leaves their effect at its prior, so
            # its LOO predictive is exact with the effect integrated out (mean without v_r,
            # variance sigma^2 + omega^2). PSIS is used for everything else.
            _v = _post['runner_effect'].stack(s=('chain', 'draw')).transpose('s', ...).values[:, runner_idx]
            _om = _post['omega'].stack(s=('chain', 'draw')).values[:, None]
            _mu = np.where(single_finish, _mu - _v, _mu)
            _sd = np.where(single_finish, np.sqrt(_sd**2 + _om**2), _sd)
        _ll = (-0.5 * ((log_pace - _mu) / _sd) ** 2 - np.log(_sd) - 0.5 * np.log(2 * np.pi)).astype(np.float32)
        _da = xr.DataArray(_ll.reshape(_post.sizes['chain'], _post.sizes['draw'], -1), dims=('chain', 'draw', 'finish'),
                           coords={'chain': _post['chain'], 'draw': _post['draw']})
        return az.InferenceData(posterior=_post, log_likelihood=xr.Dataset({'log_pace': _da}))


    single_finish = np.bincount(runner_idx)[runner_idx] == 1
    runner_loo_data = {_name: _with_loglik(_t) for _name, _t in runner_comparison_traces.items()}
    runner_loos = {_name: az.loo(_d, pointwise=True) for _name, _d in runner_loo_data.items()}
    runner_comparison = az.compare(runner_loos)[['rank', 'elpd_loo', 'p_loo', 'elpd_diff', 'se', 'dse']].assign(
        finishes_k_over_0_7=lambda d: [int((runner_loos[_m].pareto_k > 0.7).sum()) for _m in d.index]
    )
    runner_comparison.round(1)
    return (runner_loo_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior sensitivity

    Power-scaling sensitivity of the population parameters to their own priors (the priors we chose; the standardized runner effects belong to the hierarchy). Values above 0.05 would mean the prior shapes that parameter.
    """)
    return


@app.cell
def _(
    SCALARS,
    az,
    ftm,
    np,
    pd,
    pm,
    pytensor,
    runner_loo_data,
    runner_model,
    xr,
):
    _idata = runner_loo_data['runner effects'].copy()
    _chosen = [v for v in SCALARS if v in _idata.posterior]
    # Log density of each chosen prior at the posterior draws (each has fixed parameters, so it is
    # evaluated on its own; the stored trace drops the raw runner effects compute_log_prior needs).
    _log_prior = {}
    for _v in _chosen:
        _rv = runner_model[_v]
        _val = _rv.type()
        _fn = pytensor.function([_val], pm.logp(_rv, _val))
        _post = _idata.posterior[_v]
        _log_prior[_v] = xr.DataArray(
            np.array([_fn(np.asarray(x, dtype=_rv.dtype)) for x in _post.values.reshape(-1, *_post.shape[2:])]).reshape(_post.shape),
            dims=_post.dims, coords=_post.coords)
    _idata.add_groups(log_prior=xr.Dataset(_log_prior))
    _s = az.psens(_idata, component='prior', component_var_names=_chosen, var_names=SCALARS)
    _rows = []
    for _v in SCALARS:
        _sel = [('', {})] if 'gender' not in _s[_v].dims else [(f'[{g}]', {'gender': g}) for g in ftm.GENDERS]
        _rows += [{'parameter': _v + _l, 'prior sensitivity': float(_s[_v].sel(_c))} for _l, _c in _sel]
    runner_sensitivity = pd.DataFrame(_rows).set_index('parameter')
    runner_sensitivity['diagnosis'] = np.where(runner_sensitivity['prior sensitivity'] >= 0.05, 'prior-sensitive', 'data-dominated')
    runner_sensitivity.round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results

    **Left**: each course's effect with runner effects (vertical) against 1.1's course-only effect (horizontal). Points below the diagonal drew weaker fields than these courses on average, which made them look harder in 1.1 than they are; points above drew stronger fields and are harder than 1.1 said. **Right**: the spread of runner ability against the spread of course difficulty and of a runner's day-to-day variation.
    """)
    return


@app.cell
def _(course_label, courses, nb1_effects, np, pd, plt, runner_trace):
    _eff = runner_trace.posterior['course_effect']
    _new = _eff.mean(('chain', 'draw')).values
    _new_lo, _new_hi = np.percentile(_eff.stack(s=('chain', 'draw')).values, [3, 97], axis=1)
    _old = nb1_effects['effect_mean'].reindex(courses).to_numpy()
    _old = _old - _old.mean()
    _new_c = _new - _new.mean()
    course_comparison = pd.DataFrame({'label': course_label, 'nb1_effect': _old, 'runner_adjusted': _new_c,
                                      'shift': _new_c - _old}, index=courses).sort_values('shift')

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [3, 2]})
    _ax1.errorbar(_old, _new_c, yerr=[_new_c - (_new_lo - _new.mean()), (_new_hi - _new.mean()) - _new_c], fmt='o', color='#2A9D8F', ms=5, alpha=0.8)
    _lim = [min(_old.min(), _new_c.min()) - 0.05, max(_old.max(), _new_c.max()) + 0.05]
    _ax1.plot(_lim, _lim, color='gray', lw=1)
    for _i in np.argsort(np.abs(_new_c - _old))[-10:]:
        _ax1.annotate(str(course_label[_i])[:28], (_old[_i], _new_c[_i]), fontsize=7, xytext=(4, 2), textcoords='offset points')
    _ax1.set(xlabel='1.1 course effect (course only)', ylabel='Course effect with runner effects', title='Course effects, centered over these 28 courses')
    _draws = lambda v: runner_trace.posterior[v].values.reshape(-1, *runner_trace.posterior[v].shape[2:])
    _parts = {'courses (tau)': _draws('tau'), 'runners (omega)': _draws('omega'),
              'day-to-day, M (sigma)': _draws('sigma')[:, 0], 'day-to-day, F (sigma)': _draws('sigma')[:, 1]}
    _ax2.violinplot(list(_parts.values()), showmedians=True)
    _ax2.set(xticks=range(1, 5), xticklabels=list(_parts), ylabel='SD of log pace', title='Where pace variation comes from')
    for _ax in (_ax1, _ax2):
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return (course_comparison,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Handoff

    Saved to `data/cache/1.2_finish_runners/`: runner-adjusted course effects (`course_effects.parquet`, with 1.1's for comparison) and runner effects (`runner_effects.parquet`, posterior mean and SD, with each runner's number of courses here). The table lists the courses whose effect moved most.
    """)
    return


@app.cell
def _(
    CACHE,
    course_comparison,
    ftm,
    np,
    pd,
    runner_courses,
    runner_gender,
    runner_trace,
    runners,
):
    _rp = runner_trace.posterior['runner_effect']
    runner_effects = pd.DataFrame({
        'participant_id': runners,
        'gender': np.array(ftm.GENDERS)[runner_gender],
        'n_courses': runner_courses,
        'effect_mean': _rp.mean(('chain', 'draw')).values,
        'effect_sd': _rp.std(('chain', 'draw')).values,
    })
    runner_effects.to_parquet(CACHE / '1.2_finish_runners' / 'runner_effects.parquet', index=False)
    course_comparison.rename_axis('course_id').reset_index().to_parquet(CACHE / '1.2_finish_runners' / 'course_effects.parquet', index=False)
    pd.concat([course_comparison.head(6), course_comparison.tail(6)]).round(3)
    return


if __name__ == "__main__":
    app.run()
