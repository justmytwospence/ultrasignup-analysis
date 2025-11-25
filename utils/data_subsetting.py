"""
Data Subsetting Utilities for UltraSignup Bayesian Analysis

This module contains functions for subsetting race data to create
manageable datasets for Bayesian modeling. Used by all model notebooks.
"""

import pandas as pd
import numpy as np


def subset_data(
    results: pd.DataFrame,
    n_courses: int = 500,
    min_different_distances: int = 3
) -> pd.DataFrame:
    """
    Subset race results to top courses by result count.
    
    Only counts results from runners with diverse distance experience
    (minimum number of different distances raced). This ensures the
    dataset captures runners who have experience across the distance
    spectrum, improving model identifiability.
    
    Parameters
    ----------
    results : pd.DataFrame
        Processed results DataFrame (output from data_processing.process_results)
    n_courses : int, default=500
        Number of courses to include in final dataset (ranked by result count)
    min_different_distances : int, default=3
        Minimum number of different distances a runner must have raced
        to be included in the course selection
        
    Returns
    -------
    pd.DataFrame
        Subset of results containing:
        - Top n_courses by result count (from diverse runners)
        - All results (including DNFs) from diverse runners on these courses
        
    Notes
    -----
    The subsetting process:
    1. Identify runners with sufficient distance diversity
    2. Count results per course, but only for diverse runners
    3. Select top n_courses by this count
    4. Return ALL results (DNFs + finishers) from diverse runners on selected courses
    """
    print(f"Selecting top {n_courses} courses by result count...")
    print(f"  (runners with {min_different_distances}+ different distances)")
    
    # Step 1: Identify runners with sufficient distance diversity
    finisher_results = results[results['finished']].copy()
    participant_distance_counts = finisher_results.groupby('participant_id')['distance_miles'].nunique()
    diverse_runners = set(participant_distance_counts[participant_distance_counts >= min_different_distances].index)
    
    print(f"  Runners with {min_different_distances}+ different distances: {len(diverse_runners):,}")
    
    # Step 2: Count results per course, but only for diverse runners
    diverse_results = finisher_results[finisher_results['participant_id'].isin(diverse_runners)]
    course_result_counts = diverse_results.groupby('name').size().sort_values(ascending=False)
    top_courses = course_result_counts.head(n_courses).index.tolist()
    
    # Step 3: Create final dataset with ALL results (including DNFs) from diverse runners on these courses
    model_data = results[
        (results['participant_id'].isin(diverse_runners)) & 
        (results['name'].isin(top_courses))
    ].copy()
    
    print(f"\n  Final dataset: {len(model_data):,} observations")
    print(f"    Courses: {model_data['name'].nunique():,}")
    print(f"    Races: {model_data['event_distance_id'].nunique():,}")
    print(f"    Participants: {model_data['participant_id'].nunique():,}")
    print(f"    Finishers: {model_data['finished'].sum():,} ({100*model_data['finished'].mean():.1f}%)")
    print(f"    DNFs: {(~model_data['finished']).sum():,} ({100*(~model_data['finished']).mean():.1f}%)")
    
    return model_data
