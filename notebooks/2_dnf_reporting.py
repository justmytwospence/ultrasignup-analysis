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
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from utils import mirror
    from utils.data_processing import is_running_race

    return is_running_race, mirror, np, pd, plt


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


if __name__ == "__main__":
    app.run()
