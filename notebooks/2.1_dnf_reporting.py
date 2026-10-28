import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DNF Reporting Model

    Many race editions post only their finishers. For those, zero DNF rows does not mean zero DNFs, and every DNF model downstream has to know which editions it can trust.

    The mirror settles most of this directly. Each edition falls into one of three kinds:

    - **reports**: it posts DNF rows, so its DNF count is known
    - **reports, zero DNFs**: it posts DNS rows but no DNFs, so it lists non-finishers and the zero is real
    - **unknown**: it posts neither, so it either had no DNFs or does not report them

    The model infers, for the unknown editions, the probability that they report. Reporting depends on era and on the organizer (series); the DNF rate depends on distance. An unknown edition whose zero DNFs would be very surprising for its distance and field size is probably not reporting. 2.3 reuses the reporting part.

    The notebook follows the Bayesian workflow: data, model and prior predictive check, fit and convergence, posterior predictive checks, model comparison with LOO, prior sensitivity, and finally the results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    import hashlib
    import os
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
    from scipy.special import betaln, expit, gammaln, log_expit

    from utils import mirror
    from utils.data_processing import is_running_race

    return (
        Path,
        az,
        betaln,
        datetime,
        expit,
        gammaln,
        hashlib,
        is_running_race,
        log_expit,
        mirror,
        np,
        pd,
        plt,
        pm,
        pt,
        time,
        xr,
    )


@app.cell
def _(is_running_race, mirror, np):
    # One row per edition (a distance at a dated event) from main.races: public results,
    # distance format, marathon to 100 miles, running races, not virtual or cancelled.
    with mirror.connect() as _con:
        editions = _con.execute("""
            SELECT did, course_id, series_id, series_name, race_name, year, distance_km,
                   n_starters, n_finishers, coalesce(n_dnf, 0) AS n_dnf, coalesce(n_dns, 0) AS n_dns,
                   reports_dnfs
            FROM races
            WHERE results_state = 'public' AND race_format = 'distance'
              AND NOT virtual AND NOT cancelled
              AND distance_km BETWEEN 42 AND 165
              AND year IS NOT NULL AND n_starters > 0
        """).df()
    editions = editions[editions['race_name'].map(is_running_race)].reset_index(drop=True)

    editions['kind'] = np.select(
        [editions['n_dnf'] > 0, editions['n_dns'] > 0],
        ['reports', 'reports, zero DNFs'],
        default='unknown',
    )
    editions['known_reporting'] = editions['kind'] != 'unknown'
    editions['log_distance'] = np.log(editions['distance_km'] / 50)
    editions
    return (editions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data

    ### Edition kinds

    How many editions of each kind, and how big they are. Starters here are finishers plus DNFs; for unknown editions that is just finishers.
    """)
    return


@app.cell
def _(editions):
    edition_kinds = (
        editions.groupby('kind')
        .agg(editions=('did', 'size'), series=('series_id', 'nunique'),
             median_starters=('n_starters', 'median'), dnf_rate=('n_dnf', 'sum'), starters=('n_starters', 'sum'))
        .assign(dnf_rate=lambda d: (d['dnf_rate'] / d['starters']).round(3))
        .drop(columns='starters')
    )
    edition_kinds
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reporting by era and distance

    Share of editions known to report (either reporting kind), by year and by distance. Reporting is mostly a matter of era: before about 2010 almost nothing posts DNFs.
    """)
    return


@app.cell
def _(editions, pd, plt):
    _by_year = editions.groupby('year').agg(share=('known_reporting', 'mean'), n=('did', 'size'))
    _by_year = _by_year[_by_year['n'] >= 20]
    _bands = pd.cut(editions['distance_km'], [42, 46, 60, 85, 110, 165],
                    labels=['marathon', '50K', '50 mi', '100K', '100 mi'])
    _recent = editions['year'] >= 2010
    _by_distance = editions[_recent].groupby(_bands[_recent], observed=True)['known_reporting'].mean()

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1]})
    _ax1.plot(_by_year.index, _by_year['share'], marker='o', ms=3, color='#2A9D8F')
    _ax1.set(title='Share of editions known to report DNFs, by year', xlabel='Year', ylabel='Share', ylim=(0, 1))
    _ax1.grid(alpha=0.3)
    _ax2.bar(_by_distance.index.astype(str), _by_distance.values, color='#2A9D8F')
    _ax2.set(title='By distance, 2010 onward', ylabel='Share', ylim=(0, 1))
    _ax2.grid(alpha=0.3, axis='y')
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reporting by series

    For series with at least five editions since 2010, the share of those editions known to report. If organizers are consistent, this piles up near 0 and 1; a series effect in the model is worth it only if it does.
    """)
    return


@app.cell
def _(editions, plt):
    _recent_eds = editions[editions['year'] >= 2010]
    _series = _recent_eds.groupby('series_id').agg(share=('known_reporting', 'mean'), n=('did', 'size'))
    series_reporting = _series[_series['n'] >= 5]

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(series_reporting['share'], bins=20, color='#2A9D8F', edgecolor='white')
    _ax.set(title=f'Share of editions known to report, per series ({len(series_reporting):,} series, 2010 onward)',
            xlabel='Share of the series\' editions known to report', ylabel='Series')
    _ax.grid(alpha=0.3, axis='y')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model

    Three parts, each a probability per edition:

    - **reporting** $r$: logit is a free effect per year (years before 2005 pooled), plus a series effect, plus a slope on log distance
    - **DNF rate** $p$: beta-binomial with logit mean linear in log distance (relative to 50 km) and overdispersion $\kappa$
    - **listing DNS rows** $d$, given that the edition reports: logit linear in log field size. Only an edition that lists non-starters can show that its zero DNFs are real; one that does not lands among the unknowns.

    With $n$ starters, $k$ DNFs, $f$ finishers and $q_0 = \text{BetaBinomial}(0 \mid f, p, \kappa)$, the likelihood of each kind is:

    | Kind | Likelihood |
    |---|---|
    | reports | $r \cdot \text{BetaBinomial}(k \mid n, p, \kappa) \cdot d^{g}(1-d)^{1-g}$, $g$ = lists DNS rows |
    | reports, zero DNFs | $r \cdot q_0 \cdot d$ |
    | unknown | $r \cdot q_0 \cdot (1-d) + (1 - r)$ |

    The unknown term sums over the two explanations, so there is no latent reporting flag to sample.
    """)
    return


@app.cell
def _(betaln, editions, expit, gammaln, log_expit, np, pd, pm, pt, xr):
    series_idx, series_ids = pd.factorize(editions['series_id'])
    # Almost nothing reports before 2010, so years before 2005 share one level
    # (2005 stands for '2005 and earlier').
    years = np.arange(2005, editions['year'].max() + 1)
    year_idx = (editions['year'].clip(lower=2005) - years[0]).to_numpy()
    log_distance = editions['log_distance'].to_numpy()
    log_field = np.log(editions['n_starters'].to_numpy() / 50)
    _n = editions['n_starters'].to_numpy()
    _k = editions['n_dnf'].to_numpy()
    _f = editions['n_finishers'].to_numpy()
    _lists_dns = (editions['n_dns'] > 0).to_numpy()
    _kind = editions['kind'].to_numpy()
    _rep = np.flatnonzero(_kind == 'reports')
    _zero = np.flatnonzero(_kind == 'reports, zero DNFs')
    _unk = np.flatnonzero(_kind == 'unknown')


    def _log_sigmoid(x):
        return -pt.softplus(-x)


    def _betaln(a, b):
        return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


    def build_reporting_model(series_effect=True, distance_on_reporting=True):
        """The reporting model; the flags drop terms for the model comparison."""
        with pm.Model(coords={'year': years, 'series': series_ids}) as _model:
            # Reporting: a free effect per year (reporting jumps around 2010 rather than
            # drifting, and every year has hundreds of editions) + series effect + distance slope
            _year_effect = pm.Normal('year_effect', 0.0, 5.0, dims='year')
            _logit_r = _year_effect[year_idx]
            if series_effect:
                _series_sd = pm.HalfNormal('series_sd', 2.0)
                _series_raw = pm.Normal('series_raw', 0.0, 1.0, dims='series')
                _series_effect = pm.Deterministic('series_effect', _series_sd * _series_raw, dims='series')
                _logit_r = _logit_r + _series_effect[series_idx]
            if distance_on_reporting:
                _report_distance = pm.Normal('report_distance', 0.0, 1.0)
                _logit_r = _logit_r + _report_distance * log_distance

            # DNF rate: beta-binomial on log distance
            _dnf_intercept = pm.Normal('dnf_intercept', -1.5, 1.0)
            _dnf_distance = pm.Normal('dnf_distance', 0.0, 1.0)
            _kappa = pm.LogNormal('kappa', np.log(10.0), 1.0)
            _p = pm.math.invlogit(_dnf_intercept + _dnf_distance * log_distance)
            _a, _b = _p * _kappa, (1 - _p) * _kappa

            # Listing DNS rows, given reporting: logit linear in log field size
            _dns_intercept = pm.Normal('dns_intercept', 0.0, 1.5)
            _dns_field = pm.Normal('dns_field', 0.0, 1.0)
            _logit_d = _dns_intercept + _dns_field * log_field

            _log_r, _log_not_r = _log_sigmoid(_logit_r), _log_sigmoid(-_logit_r)
            _log_d, _log_not_d = _log_sigmoid(_logit_d), _log_sigmoid(-_logit_d)
            _log_q0 = _betaln(_a, _b + _f) - _betaln(_a, _b)

            _bb = pm.logp(pm.BetaBinomial.dist(alpha=_a[_rep], beta=_b[_rep], n=_n[_rep]), _k[_rep])
            pm.Potential('reports', (_log_r[_rep] + _bb + pt.switch(_lists_dns[_rep], _log_d[_rep], _log_not_d[_rep])).sum())
            pm.Potential('reports_zero', (_log_r[_zero] + _log_q0[_zero] + _log_d[_zero]).sum())
            pm.Potential('unknown', pm.math.logaddexp(_log_r[_unk] + _log_q0[_unk] + _log_not_d[_unk], _log_not_r[_unk]).sum())
        return _model


    def edition_loglik(posterior):
        """Log-likelihood of each edition for every draw, as (chain, draw, edition): the
        Potentials' three terms edition by edition, for LOO and power-scaling. Terms a model
        variant lacks (series, distance on reporting) count as zero."""
        _get = lambda v: posterior[v].values if v in posterior else None
        _ye, _se, _rd = _get('year_effect'), _get('series_effect'), _get('report_distance')
        _out = np.empty((posterior.sizes['chain'], posterior.sizes['draw'], len(editions)))
        for _c in range(posterior.sizes['chain']):
            _col = lambda v: posterior[v].values[_c][:, None]
            _logit_r = _ye[_c][:, year_idx]
            if _se is not None:
                _logit_r = _logit_r + _se[_c][:, series_idx]
            if _rd is not None:
                _logit_r = _logit_r + _rd[_c][:, None] * log_distance
            _p = expit(_col('dnf_intercept') + _col('dnf_distance') * log_distance)
            _a, _b = _p * _col('kappa'), (1 - _p) * _col('kappa')
            _logit_d = _col('dns_intercept') + _col('dns_field') * log_field
            _log_r, _log_not_r = log_expit(_logit_r), log_expit(-_logit_r)
            _log_d, _log_not_d = log_expit(_logit_d), log_expit(-_logit_d)
            _log_q0 = betaln(_a, _b + _f) - betaln(_a, _b)
            _log_bb = gammaln(_n + 1) - gammaln(_k + 1) - gammaln(_n - _k + 1) + betaln(_k + _a, _n - _k + _b) - betaln(_a, _b)
            _out[_c] = np.select(
                [_kind == 'reports', _kind == 'reports, zero DNFs'],
                [_log_r + _log_bb + np.where(_lists_dns, _log_d, _log_not_d), _log_r + _log_q0 + _log_d],
                np.logaddexp(_log_r + _log_q0 + _log_not_d, _log_not_r),
            )
        return xr.DataArray(_out, dims=('chain', 'draw', 'edition'),
                            coords={'chain': posterior['chain'], 'draw': posterior['draw']})


    reporting_model = build_reporting_model()
    reporting_model
    return (
        build_reporting_model,
        edition_loglik,
        log_distance,
        log_field,
        reporting_model,
        series_idx,
        year_idx,
        years,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulator

    `simulate_editions` runs the model forward from one set of parameters per draw: whether each edition reports, its DNF rate, its DNF count given its finishers, and whether it lists DNS rows. That yields a simulated kind per edition, comparable with the observed one. The same simulator serves the prior and posterior checks.
    """)
    return


@app.cell
def _(editions, log_distance, np, pd, series_idx, year_idx):
    KINDS = ['reports', 'reports, zero DNFs', 'unknown']
    PARAMS = ['year_effect', 'series_effect', 'report_distance', 'dnf_intercept', 'dnf_distance', 'kappa', 'dns_intercept', 'dns_field']


    def draws_of(group, n_draws, seed=0):
        """Parameters from an InferenceData group, draws first, n_draws chosen at random."""
        _stacked = {v: group[v].stack(sample=('chain', 'draw')).transpose('sample', ...).values for v in PARAMS}
        _pick = np.random.default_rng(seed).choice(len(_stacked['kappa']), n_draws, replace=False)
        return {v: a[_pick] for v, a in _stacked.items()}


    def simulate_editions(params, seed=0):
        """Simulated kind (index into KINDS) and DNF count per draw and edition, given finishers."""
        _rng = np.random.default_rng(seed)
        _expit = lambda x: 1 / (1 + np.exp(-x))
        _f = editions['n_finishers'].to_numpy()
        _S = len(params['kappa'])
        kinds = np.empty((_S, len(editions)), dtype=np.int8)
        dnfs = np.empty((_S, len(editions)), dtype=np.int64)
        for s in range(_S):
            _report = _rng.random(len(_f)) < _expit(
                params['year_effect'][s][year_idx] + params['series_effect'][s][series_idx] + params['report_distance'][s] * log_distance
            )
            _p = _expit(params['dnf_intercept'][s] + params['dnf_distance'][s] * log_distance)
            _theta = np.clip(_rng.beta(_p * params['kappa'][s], (1 - _p) * params['kappa'][s]), 1e-9, 1 - 1e-9)
            _k = _rng.negative_binomial(np.maximum(_f, 1), 1 - _theta)  # DNFs before f finishes
            _lists = _rng.random(len(_f)) < _expit(params['dns_intercept'][s] + params['dns_field'][s] * np.log((_f + _k) / 50))
            kinds[s] = np.where(_report & (_k > 0), 0, np.where(_report & _lists, 1, 2))
            dnfs[s] = _k
        return kinds, dnfs


    observed_kind = pd.Categorical(editions['kind'], categories=KINDS).codes
    return KINDS, PARAMS, draws_of, observed_kind, simulate_editions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior predictive check

    What the priors alone imply for the share of each kind and the DNF rate of reporting editions, against the observed values (dashed). The priors should allow the data without concentrating on it.
    """)
    return


@app.cell
def _(
    KINDS,
    PARAMS,
    draws_of,
    editions,
    np,
    observed_kind,
    plt,
    pm,
    reporting_model,
    simulate_editions,
):
    with reporting_model:
        _prior = pm.sample_prior_predictive(draws=200, var_names=PARAMS, random_seed=1)
    prior_kinds, prior_dnfs = simulate_editions(draws_of(_prior.prior, 200), seed=1)

    _n_obs = editions['n_starters'].to_numpy()
    _obs_rate = editions.loc[editions['kind'] == 'reports', 'n_dnf'].sum() / _n_obs[editions['kind'] == 'reports'].sum()
    _f = editions['n_finishers'].to_numpy()
    _sim_rate = np.array([
        prior_dnfs[s][prior_kinds[s] == 0].sum() / (_f + prior_dnfs[s])[prior_kinds[s] == 0].sum() if (prior_kinds[s] == 0).any() else np.nan
        for s in range(len(prior_kinds))
    ])

    _fig, _axes = plt.subplots(1, 4, figsize=(14, 3))
    for _i, _kind in enumerate(KINDS):
        _axes[_i].hist((prior_kinds == _i).mean(axis=1), bins=20, range=(0, 1), color='#999999')
        _axes[_i].axvline((observed_kind == _i).mean(), color='#E63946', ls='--')
        _axes[_i].set(title=f'Share {_kind}', xlabel='Share of editions')
    _axes[3].hist(_sim_rate[~np.isnan(_sim_rate)], bins=20, range=(0, 1), color='#999999')
    _axes[3].axvline(_obs_rate, color='#E63946', ls='--')
    _axes[3].set(title='DNF rate, reporting editions', xlabel='DNFs / starters')
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
    Path,
    az,
    datetime,
    editions,
    hashlib,
    mo,
    pd,
    pm,
    reporting_model,
    time,
):
    _key = hashlib.sha1(pd.util.hash_pandas_object(
        editions[['did', 'series_id', 'year', 'distance_km', 'n_starters', 'n_dnf', 'n_dns', 'kind']], index=False
    ).to_numpy().tobytes()).hexdigest()[:12]
    _cache_dir = Path(__file__).resolve().parent.parent / 'data' / 'cache' / '2.1_dnf_reporting'
    _cache_dir.mkdir(parents=True, exist_ok=True)
    # Bump when the model changes, so a cached trace is never reused for a different model.
    _MODEL_VERSION = 3


    def fit_reporting(model, label):
        """Sample `model` with nutpie, or load its cached trace (keyed by label, model version and data)."""
        _path = _cache_dir / f'reporting_{label}_v{_MODEL_VERSION}_{_key}.nc'
        if _path.exists():
            return az.from_netcdf(_path)
        mo.output.replace(mo.callout(mo.md(f"Sampling the {label} reporting model since {datetime.now():%H:%M} (about 5 minutes)."), kind='info'))
        _start = time.time()
        with model:
            _trace = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.95, nuts_sampler='nutpie', random_seed=42, progressbar=True)
        _trace.to_netcdf(_path)
        print(f'{label}: sampled in {(time.time() - _start) / 60:.1f} min')
        return _trace


    reporting_trace = fit_reporting(reporting_model, 'full')
    reporting_summary = az.summary(
        reporting_trace,
        var_names=['series_sd', 'report_distance', 'dnf_intercept', 'dnf_distance', 'kappa', 'dns_intercept', 'dns_field'],
        round_to=3,
    )
    print(f"Divergences: {int(reporting_trace.sample_stats['diverging'].sum())}")
    reporting_summary
    return fit_reporting, reporting_trace


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Convergence

    Rank plots for the scalar parameters: with good mixing every chain's ranks are uniform (flat bars). Divergences, R-hat and ESS are in the fit summary above; the 8,000-odd year and series effects have R-hat at most about 1.01.
    """)
    return


@app.cell
def _(az, np, reporting_trace):
    _axes = az.plot_rank(reporting_trace, var_names=['series_sd', 'report_distance', 'dnf_intercept', 'dnf_distance', 'kappa', 'dns_intercept', 'dns_field'], kind='vlines', figsize=(14, 6))
    _fig = np.ravel(_axes)[0].figure
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posterior predictive checks

    The same simulation from 200 posterior draws. Each panel compares the observed share of a kind (dots) with the predicted median and 94% band, by year (top) and by finishers (bottom). Field size is the axis the model leans on to call an unknown edition non-reporting, so the bottom row is the sharpest test.
    """)
    return


@app.cell
def _(
    KINDS,
    draws_of,
    editions,
    np,
    observed_kind,
    pd,
    plt,
    reporting_trace,
    simulate_editions,
):
    post_kinds, post_dnfs = simulate_editions(draws_of(reporting_trace.posterior, 200), seed=2)


    def _share_by(groups, kinds):
        _codes, _labels = pd.factorize(groups, sort=True)
        _onehot = np.zeros((len(_labels), len(_codes)))
        _onehot[_codes, np.arange(len(_codes))] = 1
        _counts = _onehot.sum(axis=1)
        return _labels, np.stack([(_onehot @ (kinds == _i).T).T / _counts for _i in range(3)], axis=-1)  # (draw, group, kind)


    _year = editions['year'].clip(lower=2000).to_numpy()
    _field = pd.cut(editions['n_finishers'], [-1, 10, 20, 40, 80, 160, 10_000],
                    labels=['0-10', '11-20', '21-40', '41-80', '81-160', '161+']).astype(str).to_numpy()
    _order = ['0-10', '11-20', '21-40', '41-80', '81-160', '161+']

    _fig, _axes = plt.subplots(2, 3, figsize=(15, 7), sharey='row')
    for _row, (_groups, _xlabel) in enumerate([(_year, 'Year (2000 = 2000 and earlier)'), (_field, 'Finishers')]):
        _labels, _sim = _share_by(_groups, post_kinds)
        _, _obs = _share_by(_groups, observed_kind[None, :])
        _pos = np.array([_order.index(l) for l in _labels]) if _row else _labels.astype(int)
        _sort = np.argsort(_pos)
        for _i, _kind in enumerate(KINDS):
            _ax = _axes[_row, _i]
            _lo, _mid, _hi = np.percentile(_sim[:, _sort, _i], [3, 50, 97], axis=0)
            _ax.fill_between(_pos[_sort], _lo, _hi, color='#2A9D8F', alpha=0.3, label='predicted 94%')
            _ax.plot(_pos[_sort], _mid, color='#2A9D8F')
            _ax.plot(_pos[_sort], _obs[0, _sort, _i], 'o', ms=4, color='#E63946', label='observed')
            _ax.set(title=_kind, xlabel=_xlabel, ylim=(0, 1))
            if _row:
                _ax.set_xticks(range(len(_order)), _order)
            _ax.grid(alpha=0.3)
    _axes[0, 0].set_ylabel('Share of editions')
    _axes[1, 0].set_ylabel('Share of editions')
    _axes[0, 0].legend()
    _fig.tight_layout()
    _fig
    return post_dnfs, post_kinds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    DNF rates among editions that report DNFs: the distribution across editions (left) and the mean by distance (right), observed against the posterior predictive. This checks the beta-binomial part, including its overdispersion.
    """)
    return


@app.cell
def _(editions, np, pd, plt, post_dnfs, post_kinds):
    _f = editions['n_finishers'].to_numpy()
    _rep_obs = editions['kind'].to_numpy() == 'reports'
    _obs_rate = editions['n_dnf'].to_numpy()[_rep_obs] / editions['n_starters'].to_numpy()[_rep_obs]
    _bins = np.linspace(0, 1, 26)
    _hists = np.array([np.histogram(post_dnfs[s][post_kinds[s] == 0] / (_f + post_dnfs[s])[post_kinds[s] == 0], bins=_bins, density=True)[0]
                       for s in range(len(post_kinds))])
    _centers = (_bins[1:] + _bins[:-1]) / 2

    _bands = pd.cut(editions['distance_km'], [42, 46, 60, 85, 110, 165], labels=['marathon', '50K', '50 mi', '100K', '100 mi'])
    _band_codes = _bands.cat.codes.to_numpy()
    _sim_mean = np.array([[np.mean((post_dnfs[s] / (_f + post_dnfs[s]))[(post_kinds[s] == 0) & (_band_codes == b)]) for b in range(5)]
                          for s in range(len(post_kinds))])
    _obs_mean = [np.mean(_obs_rate[_band_codes[_rep_obs] == b]) for b in range(5)]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4))
    _ax1.fill_between(_centers, *np.percentile(_hists, [3, 97], axis=0), color='#2A9D8F', alpha=0.3, label='predicted 94%')
    _ax1.plot(_centers, np.median(_hists, axis=0), color='#2A9D8F')
    _ax1.plot(_centers, np.histogram(_obs_rate, bins=_bins, density=True)[0], 'o', ms=4, color='#E63946', label='observed')
    _ax1.set(title='DNF rate per reporting edition', xlabel='DNFs / starters', ylabel='Density')
    _ax1.legend()
    _ax1.grid(alpha=0.3)
    _ax2.errorbar(range(5), np.median(_sim_mean, axis=0),
                  yerr=np.abs(np.percentile(_sim_mean, [3, 97], axis=0) - np.median(_sim_mean, axis=0)),
                  fmt='s', color='#2A9D8F', capsize=4, label='predicted 94%')
    _ax2.plot(range(5), _obs_mean, 'o', color='#E63946', label='observed')
    _ax2.set_xticks(range(5), _bands.cat.categories)
    _ax2.set(title='Mean DNF rate of reporting editions, by distance', ylabel='DNFs / starters', ylim=(0, None))
    _ax2.legend()
    _ax2.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model comparison

    PSIS-LOO estimates how well each model predicts an edition it has not seen (expected log predictive density, higher is better). The full model is compared with two simpler ones, each dropping one reporting term:

    - **no series effect**: every organizer reports at the same rate for a given year and distance
    - **no distance on reporting**: reporting does not depend on distance (DNF rates still do)

    LOO uses every 4th draw (1,000 per model). Pareto $\hat k$ above 0.7 marks editions whose LOO estimate is unreliable, typically ones that alone pin down their series' effect.
    """)
    return


@app.cell
def _(build_reporting_model, fit_reporting, reporting_trace):
    comparison_traces = {
        'full': reporting_trace,
        'no series effect': fit_reporting(build_reporting_model(series_effect=False), 'no_series'),
        'no distance on reporting': fit_reporting(build_reporting_model(distance_on_reporting=False), 'no_report_distance'),
    }
    return (comparison_traces,)


@app.cell
def _(az, comparison_traces, edition_loglik, xr):
    def _with_loglik(trace):
        _posterior = trace.posterior.sel(draw=slice(None, None, 4))
        return az.InferenceData(posterior=_posterior, log_likelihood=xr.Dataset({'reporting': edition_loglik(_posterior)}))


    loo_data = {_name: _with_loglik(_t) for _name, _t in comparison_traces.items()}
    reporting_loos = {_name: az.loo(_d, pointwise=True) for _name, _d in loo_data.items()}
    reporting_comparison = az.compare(reporting_loos).assign(
        share_k_over_0_7=lambda d: [float((reporting_loos[_m].pareto_k > 0.7).mean()) for _m in d.index]
    )
    reporting_comparison.round(3)
    return loo_data, reporting_comparison


@app.cell
def _(az, reporting_comparison):
    _ax = az.plot_compare(reporting_comparison, figsize=(10, 3))
    _ax.set_title('Expected log predictive density (PSIS-LOO), with standard errors')
    _ax.figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior sensitivity

    Power-scaling sensitivity (priorsense): how far each posterior moves when its prior is raised to a power near 1. Values above 0.05 would mean the prior, not the data, shapes that parameter.

    Scaled are the priors we chose: the top-level parameters and the year effects. The standardized series effects (`series_raw`, 2,372 unit normals) belong to the hierarchy, not to a choice of prior; scaling them too, like scaling the whole likelihood, makes the importance weights collapse with this many parameters, so those columns are not shown. Uses the same thinned full-model draws as LOO.
    """)
    return


@app.cell
def _(az, loo_data, np, pd, pm, reporting_model):
    _idata = loo_data['full'].copy()
    pm.compute_log_prior(_idata, model=reporting_model)
    _scalars = ['series_sd', 'report_distance', 'dnf_intercept', 'dnf_distance', 'kappa', 'dns_intercept', 'dns_field']
    _sens = az.psens(_idata, component='prior', component_var_names=_scalars + ['year_effect'], var_names=_scalars)
    prior_sensitivity = pd.DataFrame({'prior sensitivity': [float(_sens[_v]) for _v in _scalars]}, index=_scalars)
    prior_sensitivity['diagnosis'] = np.where(prior_sensitivity['prior sensitivity'] >= 0.05, 'prior-sensitive', 'data-dominated')
    prior_sensitivity.round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results

    Left: probability of reporting by year (2005 stands for 2005 and earlier) for a typical series at 50 km (series effect zero), with a 94% band. Right: for the unknown editions, the posterior probability that they do report (so their zero DNFs are real), by era.
    """)
    return


@app.cell
def _(
    az,
    editions,
    log_distance,
    log_field,
    np,
    plt,
    reporting_trace,
    series_idx,
    year_idx,
    years,
):
    _post = az.extract(reporting_trace, num_samples=400, rng=1)
    _ye = _post['year_effect'].values                           # (year, draw)
    _curve = 1 / (1 + np.exp(-_ye))

    # P(reports | unknown) per edition, averaged over draws
    _logit_r = _ye[year_idx] + _post['series_effect'].values[series_idx] + _post['report_distance'].values * log_distance[:, None]
    _p = 1 / (1 + np.exp(-(_post['dnf_intercept'].values + _post['dnf_distance'].values * log_distance[:, None])))
    _kap = _post['kappa'].values
    _a, _b = _p * _kap, (1 - _p) * _kap
    from scipy.special import betaln as _betaln_np, expit as _expit
    _f = editions['n_finishers'].to_numpy()[:, None]
    _q0 = np.exp(_betaln_np(_a, _b + _f) - _betaln_np(_a, _b))
    _d = _expit(_post['dns_intercept'].values + _post['dns_field'].values * log_field[:, None])
    _r = _expit(_logit_r)
    _p_rep = _r * _q0 * (1 - _d) / (_r * _q0 * (1 - _d) + 1 - _r)
    edition_reporting = editions[['did', 'course_id', 'series_id', 'year', 'distance_km', 'n_finishers', 'kind']].assign(
        p_reports=np.where(editions['kind'] == 'unknown', _p_rep.mean(axis=1), 1.0)
    )

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _ax1.plot(years, np.median(_curve, axis=1), color='#2A9D8F')
    _ax1.fill_between(years, *np.percentile(_curve, [3, 97], axis=1), color='#2A9D8F', alpha=0.25)
    _ax1.set(title='P(reports DNFs) by year, typical series at 50 km', xlabel='Year', ylabel='Probability', ylim=(0, 1))
    _ax1.grid(alpha=0.3)
    _unknown = edition_reporting[edition_reporting['kind'] == 'unknown']
    for _label, _mask, _color in [('before 2010', _unknown['year'] < 2010, '#999999'), ('2010 onward', _unknown['year'] >= 2010, '#2A9D8F')]:
        _ax2.hist(_unknown.loc[_mask, 'p_reports'], bins=25, range=(0, 1), alpha=0.6, color=_color, label=f'{_label} ({_mask.sum():,})')
    _ax2.set(title='P(reports | nothing posted), unknown editions', xlabel='Probability', ylabel='Editions')
    _ax2.legend()
    _ax2.grid(alpha=0.3, axis='y')
    _fig.tight_layout()
    _fig
    return (edition_reporting,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Handoff

    Each edition's probability of reporting DNFs is saved as `data/cache/2.1_dnf_reporting/edition_reporting.parquet` (1 for editions known to report, the posterior mean for unknown ones), with its course, series, year, distance and finishers. 2.3 uses it as the weight on an edition's zero-DNF count.
    """)
    return


@app.cell
def _(Path, edition_reporting):
    edition_reporting_path = Path(__file__).resolve().parent.parent / 'data' / 'cache' / '2.1_dnf_reporting' / 'edition_reporting.parquet'
    edition_reporting.to_parquet(edition_reporting_path, index=False)
    edition_reporting.groupby('kind').agg(editions=('did', 'size'), mean_p_reports=('p_reports', 'mean')).round(3)
    return


if __name__ == "__main__":
    app.run()
