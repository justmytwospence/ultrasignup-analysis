"""A tiny stand-in for the mirror's analyst schema (`main`), built as a local DuckDB file.

Only the columns this repo reads are modeled; the real schema is documented in the
mirror's docs/schema.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCHEMA = """
CREATE TABLE about (schema_version VARCHAR, snapshot_at TIMESTAMP);
CREATE TABLE courses (course_id VARCHAR, series_name VARCHAR, distance_km DOUBLE);
CREATE TABLE races (did INTEGER, course_id VARCHAR, race_name VARCHAR, race_date DATE, distance_km DOUBLE, race_format VARCHAR, virtual BOOLEAN, cancelled BOOLEAN, reports_dnfs BOOLEAN);
CREATE TABLE runners (runner_id INTEGER, first_name VARCHAR, last_name VARCHAR);
CREATE TABLE events (event_id INTEGER, series_id INTEGER, event_name VARCHAR);
CREATE TABLE relay_results (did INTEGER, team_name VARCHAR);
CREATE TABLE results (
    did INTEGER, runner_id INTEGER, course_id VARCHAR, race_name VARCHAR, race_date DATE,
    year INTEGER, distance_km DOUBLE, race_format VARCHAR, first_name VARCHAR, last_name VARCHAR,
    gender VARCHAR, age INTEGER,
    status VARCHAR, finish_time_s DOUBLE, source VARCHAR
);
"""

# (did, runner_id, course_id, race_name, distance_km, race_format, gender, age, status, finish_time_s, source)
RESULTS = [
    # A 50K with every status
    (1, 10, "100-50k", "Canyon Run - 50K", 50.0, "distance", "M", 30, "finished", 5 * 3600, "results_api"),
    (1, 11, "100-50k", "Canyon Run - 50K", 50.0, "distance", "F", 41, "finished_small_field", 6 * 3600, "results_api"),
    (1, 12, "100-50k", "Canyon Run - 50K", 50.0, "distance", "M", 25, "unofficial", 7 * 3600, "results_api"),
    (1, 13, "100-50k", "Canyon Run - 50K", 50.0, "distance", "F", 35, "dnf", None, "results_api"),
    (1, 14, "100-50k", "Canyon Run - 50K", 50.0, "distance", "M", 50, "dns", None, "results_api"),
    (1, 15, "100-50k", "Canyon Run - 50K", 50.0, "distance", "M", 28, "dq", 4 * 3600, "results_api"),
    # Filtered out: unknown age, gender X, implausibly fast pace
    (1, 16, "100-50k", "Canyon Run - 50K", 50.0, "distance", "M", None, "finished", 5 * 3600, "results_api"),
    (1, 17, "100-50k", "Canyon Run - 50K", 50.0, "distance", "X", 33, "finished", 5 * 3600, "results_api"),
    (1, 18, "100-50k", "Canyon Run - 50K", 50.0, "distance", "M", 22, "finished", 3600, "results_api"),
    # Filtered out: implausible ages, a mistyped 15-day finish
    (1, 19, "100-50k", "Canyon Run - 50K", 50.0, "distance", "M", -2, "finished", 5 * 3600, "results_api"),
    (1, 20, "100-50k", "Canyon Run - 50K", 50.0, "distance", "F", 3, "dnf", None, "results_api"),
    (1, 21, "100-50k", "Canyon Run - 50K", 50.0, "distance", "F", 99, "finished", 6 * 3600, "results_api"),
    (1, 22, "100-50k", "Canyon Run - 50K", 50.0, "distance", "M", 40, "finished", 15 * 86400, "results_api"),
    # A 100-miler with no DNFs
    (2, 10, "200-100mi", "Mountain 100 - 100 Miler", 160.934, "distance", "M", 30, "finished", 24 * 3600, "results_api"),
    (2, 11, "200-100mi", "Mountain 100 - 100 Miler", 160.934, "distance", "F", 41, "finished", 28 * 3600, "results_api"),
    # Dropped by format, sport and distance
    (3, 10, "300-24h", "Loop Day - 24 Hour", None, "fixed_time", "M", 30, "finished", 86400, "results_api"),
    (4, 10, "400-100k", "Gravel Grinder - 100K Ride", 100.0, "distance", "M", 30, "finished", 4 * 3600, "results_api"),
    (5, 10, "500-5k", "Fun Run - 5K", 5.0, "distance", "M", 30, "finished", 1500, "results_api"),
    (8, 10, "800-kids", "Trail Fest - Kids Run", None, "distance", "M", 30, "finished", 1500, "results_api"),
    # A virtual edition (races.virtual)
    (9, 10, "900-50k", "Lockdown 50K - 50K", 50.0, "distance", "M", 30, "finished", 5 * 3600, "results_api"),
    # No course identity yet (series unknown): dropped
    (7, 10, None, "Mystery Run - 50K", 50.0, "distance", "M", 30, "finished", 5 * 3600, "results_api"),
    # A non-default source: excluded unless asked for
    (6, 10, "600-50k", "Hidden Trail - 50K", 50.0, "distance", "M", 30, "finished", 5 * 3600, "participant_history"),
]


def build_mirror(path: Path, schema_version: str = "1.0") -> Path:
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    con.execute("INSERT INTO about VALUES (?, now())", [schema_version])
    con.executemany(
        "INSERT INTO results VALUES (?, ?, ?, ?, DATE '2025-06-01', 2025, ?, ?, 'Ann', 'Runner', ?, ?, ?, ?, ?)",
        RESULTS,
    )
    con.execute("INSERT INTO races SELECT DISTINCT did, course_id, race_name, race_date, distance_km, race_format, did = 9, false, did IN (1, 3) FROM results")
    con.close()
    return path


@pytest.fixture
def mirror_file(tmp_path: Path) -> Path:
    return build_mirror(tmp_path / "mirror.duckdb")


@pytest.fixture
def mirror_con(mirror_file: Path):
    con = duckdb.connect(str(mirror_file), read_only=True)
    yield con
    con.close()
