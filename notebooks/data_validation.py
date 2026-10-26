import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # UltraSignup data validation

    Sanity checks against `analysis/data/ultrasignup.duckdb`. We're following along live; cells will fill in as we go.
    """)
    return


@app.cell
def _():
    import duckdb
    from pathlib import Path

    DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ultrasignup.duckdb"
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    return (conn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Row counts
    """)
    return


@app.cell
def _(conn):
    counts = conn.execute(
        """
        SELECT 'races' AS tbl, COUNT(*) AS n FROM races
        UNION ALL SELECT 'results', COUNT(*) FROM results
        UNION ALL SELECT 'participants', COUNT(*) FROM participants
        """
    ).fetchall()
    counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Race date distribution

    Pre-1990 races are mostly legitimate historical entries (JFK 50 etc., back to 1963).
    The earliest record (1931 virtual race) is the only obvious data-quality oddity.
    """)
    return


@app.cell
def _(conn):
    yearly = conn.execute(
        """
        SELECT EXTRACT(YEAR FROM race_date)::INT AS yr, COUNT(*) AS n_races
        FROM races
        WHERE race_date IS NOT NULL
        GROUP BY 1 ORDER BY 1
        """
    ).fetchdf()
    yearly
    return


@app.cell
def _(conn):
    pre_1990 = conn.execute(
        """
        SELECT event_distance_id, name, race_date, location, distance_km
        FROM races
        WHERE race_date < '1990-01-01'
        ORDER BY race_date
        """
    ).fetchdf()
    pre_1990
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. DNF rate by distance bucket

    Sanity check: monotone increasing with distance (longer races have higher
    DNF rates). Confirms the data is plausible for ultramarathon modeling.
    """)
    return


@app.cell
def _(conn):
    dnf_by_dist = conn.execute(
        """
        SELECT
          CASE
            WHEN distance_km < 50 THEN '<50K'
            WHEN distance_km < 80 THEN '50K-80K (50mi)'
            WHEN distance_km < 100 THEN '80K-100K'
            WHEN distance_km < 161 THEN '100K-100mi'
            ELSE '>100mi'
          END AS bucket,
          COUNT(*) AS n_results,
          ROUND(100.0 * AVG(CASE WHEN status != 1 THEN 1.0 ELSE 0.0 END), 2) AS dnf_pct
        FROM results r JOIN races ra USING (event_distance_id)
        WHERE distance_km IS NOT NULL
        GROUP BY 1
        ORDER BY MIN(distance_km)
        """
    ).fetchdf()
    dnf_by_dist
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Time and field outliers

    - `under_1hr` / `over_100hr`: extreme time_ms values (note: `time_ms < 1hr` is
      mostly road-race results scraped along with the ultras)
    - `bad_gender`: rows with gender other than M/F (analysis filters to M/F only)
    """)
    return


@app.cell
def _(conn):
    outliers = conn.execute(
        """
        SELECT
          COUNT(*) FILTER (WHERE time_ms IS NULL) AS null_time_ms,
          COUNT(*) FILTER (WHERE time_ms < 3600000) AS under_1hr,
          COUNT(*) FILTER (WHERE time_ms > 360000000) AS over_100hr,
          COUNT(*) FILTER (WHERE gender NOT IN ('M', 'F') OR gender IS NULL) AS bad_gender,
          COUNT(*) AS total
        FROM results
        """
    ).fetchdf()
    outliers
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. last_scraped_at reliability

    Bug surfaced during validation: ~52K races have `last_scraped_at IS NULL`
    but DO have results in the results table. The scraper inserted results
    without updating the timestamp. Tracked as a follow-up; for incremental
    scraping decisions, treat any race with `EXISTS(results)` as scraped
    regardless of timestamp.
    """)
    return


@app.cell
def _(conn):
    scrape_status = conn.execute(
        """
        SELECT
          CASE WHEN last_scraped_at IS NULL THEN 'never_scraped' ELSE 'scraped' END AS scrape_status,
          CASE WHEN res_count > 0 THEN 'has_results' ELSE 'no_results' END AS result_status,
          CASE
            WHEN race_date > CURRENT_DATE THEN 'future'
            WHEN race_date IS NULL THEN 'no_date'
            ELSE 'past'
          END AS date_status,
          COUNT(*) AS n_races
        FROM (
          SELECT r.event_distance_id, r.last_scraped_at, r.race_date,
                 (SELECT COUNT(*) FROM results res
                    WHERE res.event_distance_id = r.event_distance_id) AS res_count
          FROM races r
        )
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """
    ).fetchdf()
    scrape_status
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Distance parsing sanity (post-backfill)

    After the 2026-05 fix to `DistanceExtractor` (lowercase 'm' no longer
    matches as miles), the obviously-bogus mappings are gone. A small
    residual issue with multi-distance race-name suffixes is tracked as
    UltraSignup-a3d.
    """)
    return


@app.cell
def _(conn):
    # races where the name says <half-marathon but distance_km is ultra (>30km)
    suspect_short = conn.execute(
        """
        SELECT name, distance_km
        FROM races
        WHERE distance_km > 30
          AND (LOWER(name) LIKE '%5k%' OR LOWER(name) LIKE '%10k%' OR LOWER(name) LIKE '%half marathon%')
          AND LOWER(name) NOT LIKE '%50k%' AND LOWER(name) NOT LIKE '%100k%' AND LOWER(name) NOT LIKE '%150k%'
        ORDER BY distance_km DESC
        LIMIT 20
        """
    ).fetchdf()
    suspect_short
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Findings rolled up:

    | Check                                                   | Status         |
    |---------------------------------------------------------|----------------|
    | Row counts: 77K races, 3.2M results, 1.7M participants  | OK             |
    | Pre-1990 race dates are real historical events          | OK             |
    | DNF rate increases monotonically with distance          | OK             |
    | `time_ms < 1hr` rows are short road races (not ultras)  | known          |
    | `bad_gender` rows: ~1.2K (M/F filter handles)           | tolerable      |
    | **`last_scraped_at` lies for ~52K races**               | bug — see plan |
    | **`distance_km` parser fixed (M-vs-m bug)**             | fixed in 55305b9 |
    | Multi-distance suffix-priority bug                      | UltraSignup-a3d |
    """)
    return


if __name__ == "__main__":
    app.run()
