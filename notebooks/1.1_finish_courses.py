import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Finish Time Model

    The simplest worthwhile model of finish times: how pace depends on distance, gender and age, and how much courses differ in difficulty. It is the baseline the later models (runner effects, DNFs) build on.

    Log pace (min/km) of a finisher is a per-gender intercept at 50 km, a distance exponent, a quadratic in age, a course effect shared by everyone at that course, and noise:

    $$\log \text{pace}_i = \alpha_g + \beta_g \log\frac{d_c}{50\,\text{km}} + \gamma_{1g} A_i + \gamma_{2g} A_i^2 + u_c + \varepsilon_i, \qquad u_c \sim N(0, \tau^2),\; \varepsilon_i \sim N(0, \sigma_g^2)$$

    with $A_i = (\text{age}_i - 40)/10$. Each course has one distance, so the course effects integrate out exactly and the model fits every finisher at every course in seconds; the course effects are then drawn from their exact conditional posterior. No k-core subset and no second-stage point estimates are needed.

    What it leaves out on purpose, for later notebooks: who runs each course (no runner effect, so a course drawing a strong field looks easier), edition-to-edition variation (weather, reroutes), and DNFs (only finishers are modeled).

    Sections follow the Bayesian workflow: data, model and prior predictive check, fit and convergence, posterior predictive checks, model comparison (LOO), prior sensitivity, results, and the handoff to later notebooks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    Load and process the whole dataset
    """)
    return


@app.cell
def _():
    import hashlib
    import time
    from datetime import datetime
    from pathlib import Path

    import arviz as az
    import numpy as np
    import pandas as pd
    import pymc as pm
    import pytensor.tensor as pt
    import xarray as xr
    import matplotlib.pyplot as plt

    from utils import finish_time_model as ftm
    from utils.data_processing import load_results, process_results

    return (
        Path,
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
        pt,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data

    Every finisher from marathon to 100 miles, after `process_results` (running races in distance format, not virtual or cancelled, plausible ages and paces, a known course). Whether a race posts its DNFs does not matter for finish times, so no reporting filter.
    """)
    return


@app.cell
def _(ftm, load_results, pd, process_results):
    _results = process_results(load_results())
    finishers = _results[_results['finished'] & (_results['distance_km'] >= 42)].reset_index(drop=True)
    finishers['pace_min_km'] = finishers['time_ms'] / 60000 / finishers['distance_km']
    course_stats = ftm.course_stats(finishers)
    course_labels = (
        finishers.sort_values('race_date').groupby('name')['race_name'].last()
        .str.replace(r' \d{4} - ', ' - ', regex=True)
    )
    distance_bands = pd.cut(finishers['distance_km'], [42, 46, 60, 85, 110, 165],
                            labels=['marathon', '50K', '50 mi', '100K', '100 mi'], include_lowest=True)

    finisher_summary = (
        finishers.groupby([distance_bands, 'gender'], observed=True)
        .agg(finishers=('participant_id', 'size'), courses=('name', 'nunique'), median_pace_min_km=('pace_min_km', 'median'))
        .unstack('gender').round(2)
    )
    finisher_summary
    return course_labels, course_stats, distance_bands, finishers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pace by distance and age

    Left: each course's median pace by distance (one point per course and gender). Right: pace relative to the median of the same course and gender, by age, which removes differences in which courses each age group runs.
    """)
    return


@app.cell
def _(finishers, ftm, np, plt):
    _cell = finishers.groupby(['name', 'gender']).agg(km=('distance_km', 'first'), pace=('pace_min_km', 'median'), n=('pace_min_km', 'size')).reset_index()
    _rel = finishers['pace_min_km'] / finishers.groupby(['name', 'gender'])['pace_min_km'].transform('median')
    _age = (finishers['age'] // 5) * 5
    _by_age = _rel.groupby([_age, finishers['gender']]).median().unstack()
    _colors = {'M': 'steelblue', 'F': 'coral'}

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for _g in ftm.GENDERS:
        _c = _cell[_cell['gender'] == _g]
        _ax1.scatter(_c['km'], _c['pace'], s=np.clip(_c['n'] / 20, 2, 40), alpha=0.25, color=_colors[_g], label=_g)
        _ax2.plot(_by_age.index, _by_age[_g], marker='o', ms=3, color=_colors[_g], label=_g)
    _ax1.set(xscale='log', yscale='log', xlabel='Distance (km)', ylabel='Median pace (min/km)', title='Course median pace by distance')
    _ax1.set_xticks([42.2, 50, 80.5, 100, 160.9], ['marathon', '50K', '50 mi', '100K', '100 mi'])
    _ax1.xaxis.set_minor_formatter(plt.NullFormatter())
    _ax1.set_yticks([4, 6, 8, 10, 15, 20, 30], ['4', '6', '8', '10', '15', '20', '30'])
    _ax1.yaxis.set_minor_formatter(plt.NullFormatter())
    _ax1.legend()
    _ax1.grid(alpha=0.3)
    _ax2.axhline(1, color='gray', lw=1)
    _ax2.set(xlabel='Age (5-year bins)', ylabel='Pace / course median', title='Pace relative to course median, by age')
    _ax2.legend()
    _ax2.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model

    Priors are fixed in advance, not computed from the data:

    | Parameter | Prior | Meaning |
    |---|---|---|
    | `pace_50k_mean` | $N(\log 8, 0.3)$ | mean of the two genders' $\alpha_g$, log min/km at 50 km, age 40, average course: 8 min/km, 95% between 4.4 and 14.5 |
    | `pace_50k_gap` | $N(0, 0.2)$ | women minus men, $\alpha_F - \alpha_M$ |
    | `distance_exponent_mean` | $N(0.1, 0.1)$ | mean of the genders' $\beta_g$: pace slows 0 to 20% per doubling of distance (Riegel's 1.06 on time is 0.06 on pace) |
    | `distance_exponent_gap` | $N(0, 0.1)$ | women minus men, $\beta_F - \beta_M$ |
    | `age_linear` $\gamma_{1g}$ | $N(0, 0.1)$ | per decade from 40, up to about ±20% |
    | `age_quadratic` $\gamma_{2g}$ | $N(0, 0.05)$ | curvature per decade squared |
    | `sigma` $\sigma_g$ | $\text{HalfNormal}(0.3)$ | spread of finishers within a course |
    | `tau` $\tau$ | $\text{HalfNormal}(0.3)$ | spread of course difficulty |

    `build_finish_model` takes the number of age terms (0, 1 or 2) so the same code builds the simpler models compared later. The likelihood is `utils/finish_time_model.py`, tested against a brute-force multivariate normal.
    """)
    return


@app.cell
def _(course_stats, ftm, np, pm, pt):
    COEFFICIENTS = ['pace_50k', 'distance_exponent', 'age_linear', 'age_quadratic']


    def build_finish_model(age_terms=2):
        """The finish-time model with 0, 1 or 2 age terms (2 is the full model)."""
        _stats = course_stats.features(2 + age_terms)
        with pm.Model(coords={'gender': list(ftm.GENDERS)}) as _model:
            # Each gender pair as a shared mean and a female-minus-male gap: the two genders'
            # levels move together (they share the course-effect mean), so separate
            # parameters would be ~0.99 correlated and sample poorly.
            _half = np.array([-0.5, 0.5])
            _pace = pm.Deterministic('pace_50k', pm.Normal('pace_50k_mean', np.log(8.0), 0.3)
                                     + _half * pm.Normal('pace_50k_gap', 0.0, 0.2), dims='gender')
            _exponent = pm.Deterministic('distance_exponent', pm.Normal('distance_exponent_mean', 0.1, 0.1)
                                         + _half * pm.Normal('distance_exponent_gap', 0.0, 0.1), dims='gender')
            _cols = [_pace, _exponent]
            if age_terms >= 1:
                _cols.append(pm.Normal('age_linear', 0.0, 0.1, dims='gender'))
            if age_terms >= 2:
                _cols.append(pm.Normal('age_quadratic', 0.0, 0.05, dims='gender'))
            _sigma = pm.HalfNormal('sigma', 0.3, dims='gender')
            _tau = pm.HalfNormal('tau', 0.3)
            pm.Potential('finish_times', ftm.course_loglik_pt(_stats, pt.stack(_cols, axis=1), _sigma, _tau).sum())
        return _model


    def draws_of(group):
        """theta (S, 2, K), sigma (S, 2), tau (S,) from a posterior or prior group."""
        _flat = lambda v: group[v].stack(sample=('chain', 'draw')).transpose('sample', ...).values
        _theta = np.stack([_flat(v) for v in COEFFICIENTS if v in group], axis=-1)
        return _theta, _flat('sigma'), _flat('tau')


    finish_model = build_finish_model()
    finish_model
    return COEFFICIENTS, build_finish_model, draws_of, finish_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior predictive check

    Finish times the priors alone imply for a 40-year-old at an average-difficulty draw of course, against the observed range (shaded: 1st to 99th percentile, dashed: median). The priors should cover the data comfortably without putting much mass on absurd times.
    """)
    return


@app.cell
def _(draws_of, finish_model, finishers, np, plt, pm):
    with finish_model:
        _prior = pm.sample_prior_predictive(draws=2000, random_seed=1).prior
    _theta, _sigma, _tau = draws_of(_prior)
    _rng = np.random.default_rng(1)
    _targets = [('marathon', 42.2), ('50 mi', 80.5), ('100 mi', 160.9)]

    _fig, _axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for _ax, (_label, _km) in zip(_axes, _targets):
        _g = _rng.integers(0, 2, len(_tau))
        _log_pace = (_theta[np.arange(len(_tau)), _g, 0] + _theta[np.arange(len(_tau)), _g, 1] * np.log(_km / 50)
                     + _rng.normal(0, _tau) + _rng.normal(0, _sigma[np.arange(len(_tau)), _g]))
        _hours = np.exp(_log_pace) * _km / 60
        _obs = finishers.loc[(finishers['distance_km'] - _km).abs() < 2, 'time_ms'] / 3.6e6
        _ax.hist(np.clip(_hours, 0.5, 200), bins=np.geomspace(0.5, 200, 50), color='#999999')
        _ax.axvspan(*np.percentile(_obs, [1, 99]), color='#E63946', alpha=0.15)
        _ax.axvline(np.median(_obs), color='#E63946', ls='--')
        _ax.set(xscale='log', title=f'{_label}: prior finish time', xlabel='Hours')
    _axes[0].set_ylabel('Prior draws')
    _fig.tight_layout()
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
    COEFFICIENTS,
    Path,
    az,
    course_stats,
    datetime,
    finish_model,
    hashlib,
    mo,
    pm,
):
    _key = hashlib.sha1(b''.join(a.tobytes() for a in (course_stats.zz, course_stats.zy, course_stats.yy))).hexdigest()[:12]
    _cache_dir = Path(__file__).resolve().parent.parent / 'data' / 'cache' / 'model_1'
    _cache_dir.mkdir(parents=True, exist_ok=True)
    # Bump when the model changes, so a cached trace is never reused for a different model.
    _MODEL_VERSION = 2


    def fit_finish(model, label):
        """Sample `model` with nutpie, or load its cached trace (keyed by label, model version and data)."""
        _path = _cache_dir / f'finish_{label}_v{_MODEL_VERSION}_{_key}.nc'
        if _path.exists():
            return az.from_netcdf(_path)
        mo.output.replace(mo.callout(mo.md(f"Sampling the {label} finish-time model since {datetime.now():%H:%M} (under a minute)."), kind='info'))
        with model:
            _trace = pm.sample(draws=1000, tune=1000, chains=4, nuts_sampler='nutpie', random_seed=42, progressbar=True)
        _trace.to_netcdf(_path)
        return _trace


    finish_trace = fit_finish(finish_model, 'full')
    finish_summary = az.summary(finish_trace, var_names=['pace_50k_mean', 'pace_50k_gap', 'distance_exponent_mean', 'distance_exponent_gap'] + COEFFICIENTS + ['sigma', 'tau'], round_to=4)
    print(f"Divergences: {int(finish_trace.sample_stats['diverging'].sum())}")
    finish_summary
    return finish_trace, fit_finish


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Convergence

    Rank plots: with good mixing every chain's ranks are uniform. R-hat and ESS are in the summary above.
    """)
    return


@app.cell
def _(COEFFICIENTS, az, finish_trace, np):
    _axes = az.plot_rank(finish_trace, var_names=COEFFICIENTS + ['sigma', 'tau'], kind='vlines', figsize=(14, 10))
    _fig = np.ravel(_axes)[0].figure
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posterior predictive checks

    The course effects are integrated out, so the checks use course-level summaries whose predictive distribution is exact, plus finisher residuals:

    - **course means**: each course's mean residual (after gender, distance and age) against its predictive distribution $N(0, \tau^2 + \bar\sigma^2/n)$, which tests the course-effect distribution
    - **within-course spread**: each course and gender's residual SD against $\sigma_g \sqrt{\chi^2_{n-1}/(n-1)}$, which tests one $\sigma_g$ for all courses
    - **finisher tails**: standardized residuals (after the posterior mean course effect) against a standard normal, which tests the Gaussian noise
    - **age, distance and year**: mean residual by age, distance band and year, which should be flat at zero; year is not in the model, so a trend there is a missing effect
    """)
    return


@app.cell
def _(course_stats, draws_of, finish_trace, finishers, ftm, np, pd):
    _rng = np.random.default_rng(3)
    _pick = _rng.choice(len(finish_trace.posterior.chain) * len(finish_trace.posterior.draw), 200, replace=False)
    _theta, _sigma, _tau = (a[_pick] for a in draws_of(finish_trace.posterior))
    _n = course_stats.n
    _nc = _n.sum(1)

    # course mean residual, observed (per draw) and replicated
    _r1 = np.stack([course_stats.zy[:, :, 0] - (course_stats.zz[:, :, 0, :] * t[None]).sum(-1) for t in _theta])  # (S, C, 2)
    course_mean_obs = _r1.sum(-1) / _nc
    _sd_rep = np.sqrt(_tau[:, None] ** 2 + (_n[None] * _sigma[:, None, :] ** 2).sum(-1) / _nc**2)
    course_mean_rep = _rng.normal(0, _sd_rep)

    # within-cell SD for cells with at least 5 finishers
    _cells = _n >= 5
    _ss = np.stack([ftm.within_course_ss(course_stats, t) for t in _theta])
    cell_sd_obs = np.sqrt(_ss[:, _cells] / (_n[_cells] - 1))
    _g_of_cell = np.broadcast_to(np.arange(2), _n.shape)[_cells]
    cell_sd_rep = _sigma[:, _g_of_cell] * np.sqrt(_rng.chisquare(_n[_cells] - 1, size=(len(_tau), _cells.sum())) / (_n[_cells] - 1))

    # finisher residuals at the posterior mean
    _th_all, _sg_all, _tu_all = draws_of(finish_trace.posterior)
    _u_mean, _ = ftm.course_effect_posterior(course_stats, _th_all[::8], _sg_all[::8], _tu_all[::8])
    course_effect_mean = pd.Series(_u_mean.mean(0), index=course_stats.courses)
    _z, _y = ftm.design(finishers)
    _g = (finishers['gender'] == 'F').to_numpy().astype(int)
    _th_mean = _th_all.mean(0)
    finisher_resid = _y - (_z * _th_mean[_g]).sum(1) - course_effect_mean.reindex(finishers['name']).to_numpy()
    finisher_std_resid = finisher_resid / _sg_all.mean(0)[_g]
    return (
        cell_sd_obs,
        cell_sd_rep,
        course_effect_mean,
        course_mean_obs,
        course_mean_rep,
        finisher_resid,
        finisher_std_resid,
    )


@app.cell
def _(
    cell_sd_obs,
    cell_sd_rep,
    course_mean_obs,
    course_mean_rep,
    distance_bands,
    finisher_resid,
    finisher_std_resid,
    finishers,
    np,
    pd,
    plt,
):
    def _band(ax, data_rep, data_obs, bins, color='#2A9D8F'):
        _h_rep = np.array([np.histogram(d, bins=bins, density=True)[0] for d in data_rep])
        _h_obs = np.median([np.histogram(d, bins=bins, density=True)[0] for d in data_obs], axis=0)
        _x = (bins[1:] + bins[:-1]) / 2
        ax.fill_between(_x, *np.percentile(_h_rep, [3, 97], axis=0), color=color, alpha=0.3, label='predicted 94%')
        ax.plot(_x, _h_obs, 'o', ms=3, color='#E63946', label='observed')


    def _mean_resid(ax, groups, xlabel, title, categorical=False):
        for _gname, _color in [('M', 'steelblue'), ('F', 'coral')]:
            _m = (finishers['gender'] == _gname).to_numpy()
            _s = pd.Series(finisher_resid[_m]).groupby(np.asarray(groups)[_m])
            _mean, _n = _s.mean(), _s.size()
            _mean = _mean[_n >= 100]
            _x = np.arange(len(_mean)) if categorical else _mean.index
            ax.plot(_x, _mean.values, marker='o', ms=3, color=_color, label=_gname)
            if categorical:
                ax.set_xticks(_x, _mean.index)
        ax.axhline(0, color='gray', lw=1)
        ax.set(title=title, xlabel=xlabel, ylabel='Mean residual (log pace)')


    _fig, _axes = plt.subplots(2, 3, figsize=(17, 8))
    _band(_axes[0, 0], course_mean_rep, course_mean_obs, np.linspace(-0.8, 0.8, 41))
    _axes[0, 0].set(title='Course mean residual', xlabel='Residual (log pace)', ylabel='Density')
    _band(_axes[0, 1], cell_sd_rep, cell_sd_obs, np.linspace(0, 0.45, 46))
    _axes[0, 1].set(title='Within-course residual SD (cells with 5+ finishers)', xlabel='SD of log pace')
    _bins = np.linspace(-6, 6, 61)
    _x = (_bins[1:] + _bins[:-1]) / 2
    _axes[0, 2].plot(_x, np.histogram(finisher_std_resid, bins=_bins, density=True)[0], 'o', ms=3, color='#E63946', label='observed')
    _axes[0, 2].plot(_x, np.exp(-_x**2 / 2) / np.sqrt(2 * np.pi), color='#2A9D8F', label='standard normal')
    _axes[0, 2].set(yscale='log', ylim=(1e-7, 1), title='Finisher standardized residuals', xlabel='Residual / sigma')
    _mean_resid(_axes[1, 0], (finishers['age'] // 5) * 5, 'Age (5-year bins, 100+ finishers)', 'Mean residual by age')
    _mean_resid(_axes[1, 1], distance_bands.astype(str).to_numpy(), 'Distance band', 'Mean residual by distance band', categorical=True)
    _axes[1, 1].set_xticks(range(5), ['marathon', '50K', '50 mi', '100K', '100 mi'])
    _mean_resid(_axes[1, 2], finishers['year'], 'Year', 'Mean residual by year (not in the model)')
    for _ax in _axes.flat:
        _ax.legend(fontsize=8)
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model comparison

    PSIS-LOO over courses (leave one course out), which is the unit the model generalizes to. The full model (quadratic age) against linear age and no age. The effective number of parameters (`p_loo`) exceeds the parameter count because leaving out a course with thousands of finishers (Western States has 11,000) moves the estimates noticeably; Pareto $\hat k$ shows whether the estimate is still reliable.
    """)
    return


@app.cell
def _(build_finish_model, finish_trace, fit_finish):
    finish_comparison_traces = {
        'quadratic age (full)': finish_trace,
        'linear age': fit_finish(build_finish_model(1), 'linear_age'),
        'no age': fit_finish(build_finish_model(0), 'no_age'),
    }
    return (finish_comparison_traces,)


@app.cell
def _(az, course_stats, draws_of, finish_comparison_traces, ftm, xr):
    def _with_loglik(trace):
        _theta, _sigma, _tau = draws_of(trace.posterior)
        _stats = course_stats.features(_theta.shape[-1])
        _ll = ftm.course_loglik(_stats, _theta, _sigma, _tau)
        _chains, _draws = trace.posterior.sizes['chain'], trace.posterior.sizes['draw']
        _da = xr.DataArray(_ll.reshape(_chains, _draws, -1), dims=('chain', 'draw', 'course'),
                           coords={'chain': trace.posterior['chain'], 'draw': trace.posterior['draw']})
        return az.InferenceData(posterior=trace.posterior, log_likelihood=xr.Dataset({'finish_times': _da}))


    finish_loo_data = {_name: _with_loglik(_t) for _name, _t in finish_comparison_traces.items()}
    finish_loos = {_name: az.loo(_d, pointwise=True) for _name, _d in finish_loo_data.items()}
    # Stacking weights fail numerically at differences this large, so they are left out.
    finish_comparison = az.compare(finish_loos)[['rank', 'elpd_loo', 'p_loo', 'elpd_diff', 'se', 'dse']].assign(
        courses_k_over_0_7=lambda d: [int((finish_loos[_m].pareto_k > 0.7).sum()) for _m in d.index]
    )
    finish_comparison.round(1)
    return (finish_loo_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior sensitivity

    Power-scaling sensitivity (priorsense) for every parameter, scaling the prior and, separately, the likelihood. Values above 0.05 flag a parameter: prior-sensitive only means the prior dominates it; sensitive to both means prior-data conflict.
    """)
    return


@app.cell
def _(COEFFICIENTS, az, finish_loo_data, finish_model, ftm, np, pd, pm):
    _idata = finish_loo_data['quadratic age (full)'].copy()
    pm.compute_log_prior(_idata, model=finish_model)
    _vars = COEFFICIENTS + ['sigma', 'tau']
    _prior_s = az.psens(_idata, component='prior', var_names=_vars)
    _lik_s = az.psens(_idata, component='likelihood', var_names=_vars)
    _rows = []
    for _v in _vars:
        for _label, _sel in ([('', {})] if 'gender' not in _prior_s[_v].dims else [(f'[{g}]', {'gender': g}) for g in ftm.GENDERS]):
            _rows.append({'parameter': _v + _label, 'prior': float(_prior_s[_v].sel(_sel)), 'likelihood': float(_lik_s[_v].sel(_sel))})
    finish_sensitivity = pd.DataFrame(_rows).set_index('parameter')
    finish_sensitivity['diagnosis'] = np.select(
        [(finish_sensitivity['prior'] >= 0.05) & (finish_sensitivity['likelihood'] >= 0.05), finish_sensitivity['prior'] >= 0.05],
        ['prior-data conflict', 'prior-dominated'], 'fine')
    finish_sensitivity.round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results

    Pace against distance for a 40-year-old on an average course (94% band), pace against age relative to 40, and the distribution of course effects. A course effect of +0.1 means pace about 10% slower than an average course of the same distance, for the same gender and age; it still includes who runs the course.
    """)
    return


@app.cell
def _(course_effect_mean, draws_of, finish_trace, ftm, np, plt):
    _theta, _sigma, _tau = draws_of(finish_trace.posterior)
    _km = np.geomspace(42.2, 160.9, 60)
    _ages = np.arange(15, 81)
    _colors = {'M': 'steelblue', 'F': 'coral'}

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for _gi, _g in enumerate(ftm.GENDERS):
        _pace = np.exp(_theta[:, _gi, 0:1] + _theta[:, _gi, 1:2] * np.log(_km / 50))
        _axes[0].fill_between(_km, *np.percentile(_pace, [3, 97], axis=0), color=_colors[_g], alpha=0.3)
        _axes[0].plot(_km, np.median(_pace, axis=0), color=_colors[_g], label=_g)
        _a = (_ages - 40) / 10
        _mult = np.exp(_theta[:, _gi, 2:3] * _a + _theta[:, _gi, 3:4] * _a**2)
        _axes[1].fill_between(_ages, *np.percentile(_mult, [3, 97], axis=0), color=_colors[_g], alpha=0.3)
        _axes[1].plot(_ages, np.median(_mult, axis=0), color=_colors[_g], label=_g)
    _axes[0].set(xscale='log', xlabel='Distance', ylabel='Pace (min/km)', title='Pace by distance, age 40, average course')
    _axes[0].set_xticks([42.2, 50, 80.5, 100, 160.9], ['marathon', '50K', '50 mi', '100K', '100 mi'])
    _axes[0].xaxis.set_minor_formatter(plt.NullFormatter())
    _axes[1].axhline(1, color='gray', lw=1)
    _axes[1].set(xlabel='Age', ylabel='Pace relative to age 40', title='Age effect on pace')
    _axes[2].hist(course_effect_mean, bins=60, color='#2A9D8F')
    _axes[2].set(xlabel='Course effect (log pace), posterior mean', ylabel='Courses', title=f'Course effects ({len(course_effect_mean):,} courses)')
    for _ax in _axes:
        _ax.grid(alpha=0.3)
    _axes[0].legend()
    _axes[1].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Handoff

    Each course's effect (posterior mean and SD, integrating over the hyperparameters) is saved for later notebooks, with the course's label, distance and number of finishers. The table below lists the 15 hardest and 15 easiest courses with at least 200 finishers.
    """)
    return


@app.cell
def _(Path, course_labels, course_stats, draws_of, finish_trace, ftm, np, pd):
    _theta, _sigma, _tau = draws_of(finish_trace.posterior)
    _means, _sds = ftm.course_effect_posterior(course_stats, _theta[::4], _sigma[::4], _tau[::4])
    course_effects = pd.DataFrame({
        'course_id': course_stats.courses,
        'label': course_labels.reindex(course_stats.courses).to_numpy(),
        'distance_km': np.exp(course_stats.log_distance) * ftm.REFERENCE_KM,
        'n_finishers': course_stats.n.sum(1).astype(int),
        'effect_mean': _means.mean(0),
        'effect_sd': np.sqrt((_sds**2).mean(0) + _means.var(0)),
    })
    course_effects_path = Path(__file__).resolve().parent.parent / 'data' / 'cache' / 'model_1' / 'course_effects.parquet'
    course_effects.to_parquet(course_effects_path, index=False)

    _big = course_effects[course_effects['n_finishers'] >= 200].sort_values('effect_mean')
    pd.concat([_big.tail(15).iloc[::-1], _big.head(15)]).round(3)
    return


if __name__ == "__main__":
    app.run()
