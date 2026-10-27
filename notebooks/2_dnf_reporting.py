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

    The model infers, for the unknown editions, the probability that they report. Reporting depends on era and on the organizer (series); the DNF rate depends on distance. An unknown edition whose zero DNFs would be very surprising for its distance and field size is probably not reporting.

    This notebook first looks at the editions, then (next) fits that model. nb 4 reuses its reporting part.
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
    import matplotlib.pyplot as plt

    from utils import mirror
    from utils.data_processing import is_running_race

    return (
        Path,
        az,
        datetime,
        hashlib,
        is_running_race,
        mirror,
        np,
        pd,
        plt,
        pm,
        pt,
        time,
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
    editions['log_distance'] = np.log(editions['distance_km'] / 50)
    editions
    return (editions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Editions

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
    ## Reporting by era and distance

    Share of editions known to report (either reporting kind), by year and by distance. Reporting is mostly a matter of era: before about 2010 almost nothing posts DNFs.
    """)
    return


@app.cell
def _(editions, pd, plt):
    editions['known_reporting'] = editions['kind'] != 'unknown'
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
    ## Reporting by series

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
def _(editions, np, pd, pm, pt):
    series_idx, series_ids = pd.factorize(editions['series_id'])
    # Almost nothing reports before 2010, so years before 2005 share one level; the random
    # walk runs from 2005 (2005 stands for '2005 and earlier').
    years = np.arange(2005, editions['year'].max() + 1)
    year_idx = (editions['year'].clip(lower=2005) - years[0]).to_numpy()
    log_distance = editions['log_distance'].to_numpy()
    log_field = np.log(editions['n_starters'].to_numpy() / 50)
    _n = editions['n_starters'].to_numpy()
    _k = editions['n_dnf'].to_numpy()
    _f = editions['n_finishers'].to_numpy()
    _lists_dns = (editions['n_dns'] > 0).to_numpy()
    _rep = np.flatnonzero(editions['kind'] == 'reports')
    _zero = np.flatnonzero(editions['kind'] == 'reports, zero DNFs')
    _unk = np.flatnonzero(editions['kind'] == 'unknown')


    def _log_sigmoid(x):
        return -pt.softplus(-x)


    def _betaln(a, b):
        return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


    _coords = {'year': years, 'series': series_ids}
    with pm.Model(coords=_coords) as reporting_model:
        # Reporting: a free effect per year (reporting jumps around 2010 rather than drifting,
        # and every year has hundreds of editions) + series effect + distance slope
        year_effect = pm.Normal('year_effect', 0.0, 5.0, dims='year')
        series_sd = pm.HalfNormal('series_sd', 2.0)
        _series_raw = pm.Normal('series_raw', 0.0, 1.0, dims='series')
        series_effect = pm.Deterministic('series_effect', series_sd * _series_raw, dims='series')
        report_distance = pm.Normal('report_distance', 0.0, 1.0)
        _logit_r = year_effect[year_idx] + series_effect[series_idx] + report_distance * log_distance

        # DNF rate: beta-binomial on log distance
        dnf_intercept = pm.Normal('dnf_intercept', -1.5, 1.0)
        dnf_distance = pm.Normal('dnf_distance', 0.0, 1.0)
        kappa = pm.LogNormal('kappa', np.log(10.0), 1.0)
        _p = pm.math.invlogit(dnf_intercept + dnf_distance * log_distance)
        _a, _b = _p * kappa, (1 - _p) * kappa

        # Listing DNS rows, given reporting: logit linear in log field size
        dns_intercept = pm.Normal('dns_intercept', 0.0, 1.5)
        dns_field = pm.Normal('dns_field', 0.0, 1.0)
        _logit_d = dns_intercept + dns_field * log_field

        _log_r, _log_not_r = _log_sigmoid(_logit_r), _log_sigmoid(-_logit_r)
        _log_d, _log_not_d = _log_sigmoid(_logit_d), _log_sigmoid(-_logit_d)
        _log_q0 = _betaln(_a, _b + _f) - _betaln(_a, _b)

        _bb = pm.logp(pm.BetaBinomial.dist(alpha=_a[_rep], beta=_b[_rep], n=_n[_rep]), _k[_rep])
        pm.Potential('reports', (_log_r[_rep] + _bb + pt.switch(_lists_dns[_rep], _log_d[_rep], _log_not_d[_rep])).sum())
        pm.Potential('reports_zero', (_log_r[_zero] + _log_q0[_zero] + _log_d[_zero]).sum())
        pm.Potential('unknown', pm.math.logaddexp(_log_r[_unk] + _log_q0[_unk] + _log_not_d[_unk], _log_not_r[_unk]).sum())

    reporting_model
    return (
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
    # Bump when the model changes, so a cached trace is never reused for a different model.
    _MODEL_VERSION = 3
    reporting_cache = Path(__file__).resolve().parent.parent / 'data' / 'cache' / 'model_2' / f'reporting_v{_MODEL_VERSION}_{_key}.nc'
    reporting_cache.parent.mkdir(parents=True, exist_ok=True)

    if reporting_cache.exists():
        reporting_trace = az.from_netcdf(reporting_cache)
    else:
        mo.output.replace(mo.callout(mo.md(f"Sampling the reporting model since {datetime.now():%H:%M} (a few minutes)."), kind='info'))
        _start = time.time()
        with reporting_model:
            reporting_trace = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.95, nuts_sampler='nutpie', random_seed=42, progressbar=True)
        reporting_trace.to_netcdf(reporting_cache)
        print(f'Sampled in {(time.time() - _start) / 60:.1f} min')

    reporting_summary = az.summary(
        reporting_trace,
        var_names=['series_sd', 'report_distance', 'dnf_intercept', 'dnf_distance', 'kappa', 'dns_intercept', 'dns_field'],
        round_to=3,
    )
    print(f"Divergences: {int(reporting_trace.sample_stats['diverging'].sum())}")
    reporting_summary
    return (reporting_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reporting over time and per edition

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
    return


if __name__ == "__main__":
    app.run()
