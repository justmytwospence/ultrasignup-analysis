"""
Data loading and the model frame for the UltraSignup Bayesian notebooks.

Data comes from the mirror's analyst schema (see utils/mirror.py): `main.results` carries
one row per runner per race with the race context (course, date, distance, format) already
joined in, statuses as strings, times in seconds and NULL for unknowns.

`load_results()` returns those rows with the mirror's column names. `process_results()`
turns them into the model frame every notebook consumes. Its column contract is:

    event_distance_id   race id (mirror `did`)
    participant_id      canonical runner id (mirror `runner_id`), as a string like the
                        old scraper's VARCHAR column, which the k-core code relies on
    name                course identity (mirror `course_id`, stable across years)
    distance_miles      course distance
    time_ms             finish time in milliseconds (NaN for DNFs)
    finished            True for finishers, False for DNFs
    pace_min_mi         finisher pace
    gender, age
plus pass-through context: course_id, race_name, race_date, year, distance_km, status, source.
"""

from __future__ import annotations

from collections.abc import Sequence

import duckdb
import pandas as pd

from utils import mirror

KM_PER_MILE = 1.609344

# Statuses that count as a start in the DNF models. dns (never started) and dq are dropped.
FINISHED_STATUSES = ("finished", "finished_small_field", "unofficial")
DNF_STATUSES = ("dnf",)

DEFAULT_SOURCES = ("results_api", "legacy_2025")

RESULT_COLUMNS = (
    "did",
    "runner_id",
    "course_id",
    "race_name",
    "race_date",
    "year",
    "distance_km",
    "race_format",
    "gender",
    "age",
    "status",
    "finish_time_s",
    "source",
)


def load_results(con: duckdb.DuckDBPyConnection | None = None, sources: Sequence[str] = DEFAULT_SOURCES) -> pd.DataFrame:
    """
    Load results from the mirror's `main.results`.

    Parameters
    ----------
    con : DuckDB connection, optional
        A connection from utils.mirror.connect(). Defaults to a live connection.
    sources : sequence of str
        Which result sources to include. Default: the public results API plus the rows
        preserved from the 2025 scrape for races since retracted. Add
        "participant_history" for results only visible on runners' pages (those races are
        usually incomplete, so they are opt-in).
    """
    own = con is None
    con = con or mirror.connect()
    try:
        # Literals, not bound parameters: parameters do not travel through CONNECT to the
        # remote (DuckDB 2.0.0 raises "Values were not provided").
        source_list = ", ".join("'" + s.replace("'", "''") + "'" for s in sources)
        return con.execute(f"SELECT {', '.join(RESULT_COLUMNS)} FROM results WHERE source IN ({source_list})").df()
    finally:
        if own:
            con.close()


def is_running_race(name: str) -> bool:
    """
    Determine if a race name indicates a running race.

    UltraSignup also hosts cycling, swimming, paddling and other events; this filters them
    out by keyword.
    """
    if pd.isna(name):
        return False

    name_lower = str(name).lower()

    non_running_keywords = [
        'bike', 'bicycle', 'cycling', 'duathlon', 'triathlon', 'swim',
        'swimming', 'paddle', 'kayak', 'canoe', 'ski', 'snowshoe', 'hike',
        'walk', 'horse', 'equestrian', 'roller', 'skate', 'scooter', 'mtb',
        'gravel', 'ride', 'cycle'
    ]

    return not any(keyword in name_lower for keyword in non_running_keywords)


def process_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    Build the model frame from load_results() output.

    Steps:
    1. Keep M/F runners with a known age.
    2. Keep running races with a fixed distance (drops fixed-time races and relays),
       between 10K and 100 miles.
    3. Keep starts that ended in a finish or a DNF (dns and dq are dropped).
    4. Compute pace and drop finishers faster than 5 min/mile (data errors).
    """
    results = results[results['gender'].isin(['M', 'F']) & results['age'].notna()].copy()
    results = results[results['race_name'].map(is_running_race)]
    results = results[results['race_format'] == 'distance']
    results = results[results['status'].isin(FINISHED_STATUSES + DNF_STATUSES)]

    results['distance_miles'] = results['distance_km'] / KM_PER_MILE
    results = results[(results['distance_miles'] >= 6.2) & (results['distance_miles'] <= 100)]

    results['finished'] = results['status'].isin(FINISHED_STATUSES) & (results['finish_time_s'] > 0)
    # A finish status without a time is unusable for either model.
    results = results[results['finished'] | results['status'].isin(DNF_STATUSES)]

    results['time_ms'] = (results['finish_time_s'] * 1000).where(results['finished'])
    results['pace_min_mi'] = (results['time_ms'] / 60000 / results['distance_miles']).where(results['finished'])
    results = results[~results['finished'] | (results['pace_min_mi'] >= 5)]

    results = results.rename(columns={'did': 'event_distance_id', 'runner_id': 'participant_id'})
    results['participant_id'] = results['participant_id'].astype(str)
    results['name'] = results['course_id']
    results['age'] = results['age'].astype(int)
    return results.reset_index(drop=True)


def filter_races_with_dnfs(results: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only courses with at least one DNF.

    Courses with no DNFs at all may be courses whose results never report DNFs, which
    would bias the DNF models.
    """
    dnf_counts = (~results['finished']).groupby(results['name']).sum()
    courses_with_dnf = dnf_counts[dnf_counts > 0].index
    return results[results['name'].isin(courses_with_dnf)].copy()
