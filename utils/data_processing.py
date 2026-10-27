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
plus pass-through context: first_name, last_name, course_id, race_name, race_date, year, distance_km, status, source.
"""

from __future__ import annotations

from collections.abc import Sequence

import re

import duckdb
import pandas as pd

from utils import mirror

KM_PER_MILE = 1.609344

# Statuses that count as a start in the DNF models. dns (never started) and dq are dropped.
FINISHED_STATUSES = ("finished", "finished_small_field", "unofficial")
DNF_STATUSES = ("dnf",)

# Plausibility bounds for the model frame (the mirror does not clean posted values).
MIN_AGE, MAX_AGE = 10, 95
# min/mile: faster than a 5:00 mile is a timing error; slower than 60 min/mile (100 miles
# in 100 hours) is a mistyped time, e.g. the 470 posted finishes over 14 days.
MIN_PACE, MAX_PACE = 5, 60

# The mirror serves only the public results: hidden results (about a third of editions,
# shown only on runner pages) are not mirrored, and the 2025 scrape is not imported.
DEFAULT_SOURCES = ("results_api",)

RESULT_COLUMNS = (
    "did",
    "runner_id",
    "course_id",
    "race_name",
    "race_date",
    "year",
    "distance_km",
    "race_format",
    "first_name",
    "last_name",
    "gender",
    "age",
    "status",
    "finish_time_s",
    "source",
)


def load_results(con: duckdb.DuckDBPyConnection | None = None, sources: Sequence[str] = DEFAULT_SOURCES) -> pd.DataFrame:
    """
    Load results from the mirror's `main.results`, plus the `virtual` and `cancelled` flags
    of their races (fetched separately and merged here: Quack cannot stream two remote
    tables in one query).

    Parameters
    ----------
    con : DuckDB connection, optional
        A connection from utils.mirror.connect(). Defaults to a live connection.
    sources : sequence of str
        Which result sources to include. The mirror only has "results_api" today; the
        parameter stays so a later source can be opted into.
    """
    own = con is None
    con = con or mirror.connect()
    try:
        # Literals, not bound parameters: parameters do not travel through CONNECT to the
        # remote (DuckDB 2.0.0 raises "Values were not provided").
        source_list = ", ".join("'" + s.replace("'", "''") + "'" for s in sources)
        results = con.execute(f"SELECT {', '.join(RESULT_COLUMNS)} FROM results WHERE source IN ({source_list})").df()
        flags = con.execute("SELECT did, virtual, cancelled FROM races").df()
        results = results.merge(flags, on="did", how="left")
        for flag in ("virtual", "cancelled"):
            results[flag] = results[flag].fillna(False).astype(bool)
        return results
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

    return _NON_RUNNING.search(str(name)) is None


# Whole words only (with common inflections), so "Whiskey" and "Pride" are not caught by
# "ski" and "ride".
_NON_RUNNING = re.compile(
    r"\b(?:bikes?|biker?s?|biking|bicycles?|cycl(?:e|es|ing|ists?)|duathlons?|triathlons?"
    r"|swim(?:s|mers?|ming)?|paddl(?:e|es|ers?|ing)|kayak(?:s|ing)?|canoe(?:s|ing)?"
    r"|ski(?:s|ers?|ing)?|snowshoe(?:s|ing)?|hik(?:e|es|ers?|ing)|walk(?:s|ers?|ing)?"
    r"|skimo|skijor\w*|fatbikes?|handcycl(?:e|es|ing)|horses?|equestrian|roller|skat(?:e|es|ers?|ing)|scooters?|mtb|gravel"
    r"|rid(?:e|es|ers?|ing))\b",
    re.IGNORECASE,
)


def process_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    Build the model frame from load_results() output.

    Steps:
    1. Keep M/F runners with a plausible known age (MIN_AGE to MAX_AGE).
    2. Keep running races (not virtual or cancelled) with a fixed distance (drops fixed-time races; relays are not in
       main.results), between 10K and 100 miles. A NULL distance_km (kids' runs,
       memberships, unparsed labels) drops the race.
    3. Keep starts that ended in a finish or a DNF (dns and dq are dropped).
    4. Compute pace and drop finishers outside MIN_PACE to MAX_PACE min/mile (data errors).

    The mirror posts values as UltraSignup has them, never corrected (ages under 5 or
    negative, finish times over 14 days), so these filters are the cleaning step.
    """
    # course_id is NULL until the mirror has fetched the race's series (a handful of new
    # editions); such races have no course identity to pool on yet, so they are dropped.
    results = results[
        results['gender'].isin(['M', 'F'])
        & results['age'].between(MIN_AGE, MAX_AGE)
        & results['course_id'].notna()
    ].copy()
    # Virtual and cancelled-but-posted races are not comparable to the real event.
    results = results[~results['virtual'] & ~results['cancelled']]
    # Classify each distinct name once (about 60k names vs millions of rows).
    names = results['race_name'].drop_duplicates()
    running = dict(zip(names, names.map(is_running_race)))
    results = results[results['race_name'].map(running)]
    results = results[results['race_format'] == 'distance']
    results = results[results['status'].isin(FINISHED_STATUSES + DNF_STATUSES)]

    results['distance_miles'] = results['distance_km'] / KM_PER_MILE
    results = results[(results['distance_miles'] >= 6.2) & (results['distance_miles'] <= 100)]

    results['finished'] = results['status'].isin(FINISHED_STATUSES) & (results['finish_time_s'] > 0)
    # A finish status without a time is unusable for either model.
    results = results[results['finished'] | results['status'].isin(DNF_STATUSES)]

    results['time_ms'] = (results['finish_time_s'] * 1000).where(results['finished'])
    results['pace_min_mi'] = (results['time_ms'] / 60000 / results['distance_miles']).where(results['finished'])
    results = results[~results['finished'] | results['pace_min_mi'].between(MIN_PACE, MAX_PACE)]

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
