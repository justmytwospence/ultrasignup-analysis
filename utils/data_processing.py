"""
Data Processing Utilities for UltraSignup Bayesian Analysis

This module contains functions for loading and processing race results data
from the UltraSignup database. Used by all Bayesian model notebooks.
"""

import duckdb
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional, Union


def load_results(db_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load race results from DuckDB database.
    
    Parameters
    ----------
    db_path : str or Path, optional
        Path to the DuckDB database file.
        If None, looks for data/ultrasignup.duckdb relative to the repo root.
        Works both locally and in Docker when data is mounted at the same relative path.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing race results with columns:
        - event_distance_id, age, gender, formattime, participant_id, 
          firstname, lastname, time_ms, status
        - name (race name)
    """
    resolved_path: Path
    
    if db_path is None:
        # Find repo root (where utils/ lives) and look for data/ultrasignup.duckdb
        repo_root = Path(__file__).parent.parent
        resolved_path = repo_root / "data" / "ultrasignup.duckdb"
    else:
        resolved_path = Path(db_path)
    
    if not resolved_path.exists():
        raise FileNotFoundError(f"Database file not found at: {resolved_path}")
    
    conn = duckdb.connect(str(resolved_path))
    
    results = conn.execute("""
        SELECT 
            results.event_distance_id,
            results.age,
            results.gender,
            results.formattime,
            results.participant_id,
            results.firstname,
            results.lastname,
            results.time_ms,
            results.status,
            races.name
        FROM results 
        LEFT JOIN races ON results.event_distance_id = races.event_distance_id
    """).fetchdf()
    
    conn.close()
    return results


def is_running_race(name: str) -> bool:
    """
    Determine if a race name indicates a running race.
    
    Filters out non-running events like cycling, swimming, triathlons, etc.
    
    Parameters
    ----------
    name : str
        Race name to check
        
    Returns
    -------
    bool
        True if the race is a running event, False otherwise
    """
    if pd.isna(name):
        return False
    
    name_lower = str(name).lower()
    
    # Keywords indicating non-running events
    non_running_keywords = [
        'bike', 'bicycle', 'cycling', 'duathlon', 'triathlon', 'swim',
        'swimming', 'paddle', 'kayak', 'canoe', 'ski', 'snowshoe', 'hike',
        'walk', 'horse', 'equestrian', 'roller', 'skate', 'scooter', 'mtb',
        'gravel', 'ride', 'cycle'
    ]
    
    for keyword in non_running_keywords:
        if keyword in name_lower:
            return False
    
    return True


def extract_distance_miles(name: str) -> Optional[float]:
    """
    Extract distance in miles from race name string.
    
    Looks for distance patterns in the last part of the race name
    (after the last hyphen) and converts to miles if needed.
    
    Parameters
    ----------
    name : str
        Race name containing distance information
        
    Returns
    -------
    float or None
        Distance in miles, or None if not found
    """
    if pd.isna(name):
        return None
    
    name_lower = str(name).lower()

    # Set up distance patterns to search for along with their conversion to miles
    distance_patterns = {
        r'(\d+(\.\d+)?)\s*miles?': 1,  # e.g., "50 miles"
        r'(\d+(\.\d+)?)\s*mi\b': 1,     # e.g., "50 mi"
        r'(\d+(\.\d+)?)\s*km\b': 0.621371,  # e.g., "80 km"
        r'(\d+(\.\d+)?)\s*k\b': 0.621371,     # e.g., "80 k"
        r'(\d+(\.\d+)?)\s*m\b': 0.000621371,  # e.g., "1609 m"
    }

    # Extract the distance after the last hyphen surrounded by spaces in the name
    match = re.search(r'-\s*([^-\n\r]+)$', name_lower)
    if match:
        distance_part = match.group(1)
        for pattern, conversion in distance_patterns.items():
            distance_match = re.search(pattern, distance_part)
            if distance_match:
                distance_value = float(distance_match.group(1))
                return distance_value * conversion
    
    return None


def process_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    Process race results: filter to running races, extract distances, create finish indicator.
    
    Processing steps:
    1. Filter to running races only (exclude cycling, swimming, etc.)
    2. Extract distance from race names
    3. Create finish indicator (finished = has valid time_ms > 0 AND status != 2)
    4. Filter out results with missing data
    5. Filter to races between 10K and 100 miles
    6. Calculate pace and filter unrealistic times (< 5 min/mile)
    
    Parameters
    ----------
    results : pd.DataFrame
        Raw results DataFrame from load_results()
        
    Returns
    -------
    pd.DataFrame
        Processed results with additional columns:
        - distance_miles: Race distance in miles
        - finished: Boolean indicator (True = finished, False = DNF)
        - pace_min_mi: Pace in minutes per mile (for finishers only)
    """

    # Filter to M and F genders only (exclude X or missing values)
    results = results[results['gender'].isin(['M', 'F'])].copy()

    # Filter to running only
    results = results[results['name'].apply(is_running_race)]
    
    # Extract distance from names
    results['distance_miles'] = results.apply(lambda row: extract_distance_miles(row['name']), axis=1)
    
    # Filter out results with missing distance or age
    results = results[results['distance_miles'].notna() & results['age'].notna()]
    
    # Create finish indicator: finished = has valid time_ms > 0 AND status != 2
    # DNFs are identified by status == 2
    results['finished'] = (
        results['time_ms'].notna() & 
        (results['time_ms'] > 0) &
        (results['status'] != 2)
    )
    
    # Keep only valid records:
    # - Finishers must have time_ms > 0
    # - DNFs (status == 2) can have any time_ms value
    results = results[results['finished'] | (results['status'] == 2)]
    
    # Filter out races longer than 100 miles or less than 10k
    results = results[(results['distance_miles'] <= 100) & (results['distance_miles'] >= 6.2)]
    
    # For finishers, calculate pace and filter unrealistic times
    results.loc[results['finished'], 'pace_min_mi'] = (
        results.loc[results['finished'], 'time_ms'] / 60000
    ) / results.loc[results['finished'], 'distance_miles']
    
    # Filter out finishers with pace under 5 minute mile (likely errors)
    results = results[~results['finished'] | (results['pace_min_mi'] >= 5)]

    return results


def filter_races_with_dnfs(results: pd.DataFrame) -> pd.DataFrame:
    """
    Filter results to only include courses with at least one DNF.
    
    This ensures we only model courses that have meaningful DNF variation.
    Courses with 0% DNF may be non-reporting courses (biased data).
    
    A course is defined by the race name.
    
    Parameters
    ----------
    results : pd.DataFrame
        Processed results DataFrame (output from process_results())
        Must contain 'name' and 'finished' columns
        
    Returns
    -------
    pd.DataFrame
        Filtered results containing only courses with at least one DNF
    """
    
    # Calculate DNF counts per course
    course_dnf_counts = results.groupby('name').agg({
        'finished': lambda x: (~x).sum()  # DNF count
    }).rename(columns={'finished': 'dnf_count'})
    
    # Filter to courses with at least one DNF
    courses_with_dnf = course_dnf_counts[course_dnf_counts['dnf_count'] > 0].index
    results_with_dnf = results[results['name'].isin(courses_with_dnf)].copy()
    return results_with_dnf
