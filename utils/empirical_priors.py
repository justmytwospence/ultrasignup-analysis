"""
Empirical Prior Calculation for Ultramarathon Bayesian Models

Simple, focused functions for calculating empirical priors from race data.
Each function calculates ONE specific prior - no hierarchical dependencies.

Usage:
    from empirical_priors import (
        calculate_gender_marathon_pace,
        calculate_gender_distance_exponent,
        calculate_course_variation,
        ...
    )
    
    # Model 1 example - only use what you need
    mu_pace_m = calculate_gender_marathon_pace(results, 'M')
    mu_pace_f = calculate_gender_marathon_pace(results, 'F')
    beta_m = calculate_gender_distance_exponent(results, 'M')
    beta_f = calculate_gender_distance_exponent(results, 'F')
    sigma_course = calculate_course_variation(results)
"""

import pandas as pd
import numpy as np
from typing import Dict
from scipy.stats import linregress


# =============================================================================
# SIMPLE, FOCUSED FUNCTIONS - Use these in your notebooks!
# =============================================================================

def calculate_gender_marathon_pace(
    results: pd.DataFrame,
    gender: str,
    reference_distance: float = 26.2
) -> float:
    """
    Calculate log(pace) at marathon distance for a specific gender.
    
    Uses median pace of races within ±2 miles of marathon distance.
    Fully vectorized for performance.
    
    Parameters
    ----------
    results : DataFrame with 'gender', 'distance_miles', 'time_ms' columns
    gender : 'M' or 'F'
    reference_distance : float, default 26.2 (marathon)
    
    Returns
    -------
    float : log(pace in min/mile) at marathon distance
    """
    # Filter to gender (vectorized)
    gender_data = results[results['gender'] == gender]
    
    # Calculate pace for all races (vectorized - no .copy() needed)
    pace = gender_data['time_ms'].values / 60000 / gender_data['distance_miles'].values
    
    # Find races near marathon distance (vectorized boolean indexing)
    marathon_mask = np.abs(gender_data['distance_miles'].values - reference_distance) < 2
    
    if marathon_mask.sum() > 0:
        return float(np.log(np.median(pace[marathon_mask])))
    else:
        # Fallback: use overall median
        return float(np.log(np.median(pace)))


def calculate_gender_distance_exponent(
    results: pd.DataFrame,
    gender: str,
    reference_distance: float = 26.2
) -> float:
    """
    Calculate distance exponent (beta) for power law: pace ~ distance^beta
    
    Fits log(pace) = intercept + beta * log(distance/reference)
    Fully vectorized for performance.
    
    Parameters
    ----------
    results : DataFrame with 'gender', 'distance_miles', 'time_ms' columns
    gender : 'M' or 'F'
    reference_distance : float, default 26.2
    
    Returns
    -------
    float : distance exponent beta
    """
    # Filter to gender (vectorized)
    gender_data = results[results['gender'] == gender]
    
    # Calculate pace and log transformations (all vectorized, no .copy())
    pace = gender_data['time_ms'].values / 60000 / gender_data['distance_miles'].values
    log_distance_ratio = np.log(gender_data['distance_miles'].values / reference_distance)
    log_pace = np.log(pace)
    
    # Linear regression: log(pace) ~ log(distance/ref)
    slope, intercept, r_value, p_value, std_err = linregress(log_distance_ratio, log_pace)
    
    return float(slope)


def calculate_course_variation(
    results: pd.DataFrame,
    reference_distance: float = 26.2
) -> float:
    """
    Calculate standard deviation of course difficulty effects.
    
    For each course, calculates median pace residual after removing
    distance and gender effects using a simple power law model.
    
    Fully vectorized for performance on large datasets.
    
    Parameters
    ----------
    results : DataFrame with 'name', 'gender', 'distance_miles', 'time_ms' columns
    reference_distance : float, default 26.2
    
    Returns
    -------
    float : standard deviation of course effects (sigma_course)
    """
    # First, get gender-specific baselines (vectorized)
    gender_params = {}
    for gender in ['M', 'F']:
        gender_data = results[results['gender'] == gender]
        pace = gender_data['time_ms'].values / 60000 / gender_data['distance_miles'].values
        log_dist_ratio = np.log(gender_data['distance_miles'].values / reference_distance)
        log_pace = np.log(pace)
        
        slope, intercept, _, _, _ = linregress(log_dist_ratio, log_pace)
        gender_params[gender] = {'intercept': intercept, 'slope': slope}
    
    # Fully vectorized calculation of residuals for ALL results at once
    # Create a copy with all needed columns
    data = results[['name', 'gender', 'distance_miles', 'time_ms']].copy()
    data['pace'] = data['time_ms'] / 60000 / data['distance_miles']
    data['log_pace'] = np.log(data['pace'])
    data['log_dist_ratio'] = np.log(data['distance_miles'] / reference_distance)
    
    # Vectorized expected log pace calculation using gender mapping
    expected_m = gender_params['M']['intercept'] + gender_params['M']['slope'] * data['log_dist_ratio']
    expected_f = gender_params['F']['intercept'] + gender_params['F']['slope'] * data['log_dist_ratio']
    data['expected_log_pace'] = np.where(data['gender'] == 'M', expected_m, expected_f)
    
    # Calculate residuals
    data['residual'] = data['log_pace'] - data['expected_log_pace']
    
    # Group by course and calculate median residual (vectorized aggregation)
    course_stats = data.groupby('name').agg({
        'residual': ['median', 'count']
    })
    course_stats.columns = ['median_residual', 'count']
    
    # Filter to courses with at least 5 observations
    course_effects = course_stats[course_stats['count'] >= 5]['median_residual'].values
    
    return float(np.std(course_effects))


# =============================================================================
# LEGACY FUNCTIONS - Keep for backward compatibility with other models
# =============================================================================

def calculate_pace_priors(
    results_finishers: pd.DataFrame,
    reference_distance: float = 26.2,
    gender_col: str = 'gender',
    distance_col: str = 'distance_miles',
    time_col: str = 'time_ms',
    participant_col: str = 'participant_id'
) -> Dict[str, Dict[str, float]]:
    """
    Calculate gender-specific pace priors using power law model.
    
    Power law model: log(pace) = log(pace_ref) + beta * log(d / d_ref)
    
    Args:
        results_finishers: DataFrame with finisher results
        reference_distance: Reference distance in miles (default: marathon = 26.2)
        gender_col: Column name for gender ('M', 'F')
        distance_col: Column name for distance in miles
        time_col: Column name for finish time in milliseconds
        participant_col: Column name for participant identifier
        
    Returns:
        dict: {
            'M': {
                'log_marathon_pace': float,  # log(pace) at reference distance
                'marathon_pace': float,      # pace at reference distance (min/mi)
                'beta': float,                # Power law exponent
                'log_pace_sd': float,        # Between-participant log-pace SD
                'n_observations': int        # Number of observations for this gender
            },
            'F': {...}
        }
    """
    priors = {}
    
    # Calculate pace for all finishers
    observed_paces = results_finishers[time_col].values / 60000 / results_finishers[distance_col].values
    finisher_distances = results_finishers[distance_col].values
    finisher_genders = results_finishers[gender_col].values
    
    for gender in ['M', 'F']:
        gender_mask = finisher_genders == gender
        gender_paces = observed_paces[gender_mask]
        gender_distances = finisher_distances[gender_mask]
        
        # Calculate empirical pace at marathon distance
        marathon_mask = np.abs(gender_distances - reference_distance) < 2
        if marathon_mask.sum() > 0:
            empirical_marathon_pace = np.median(gender_paces[marathon_mask])
            empirical_log_marathon_pace = np.log(empirical_marathon_pace)
        else:
            empirical_log_marathon_pace = np.log(np.median(gender_paces))
        
        # Calculate empirical distance exponent (beta) using linear regression
        log_distance_ratio = np.log(gender_distances / reference_distance)
        log_pace = np.log(gender_paces)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_distance_ratio, log_pace)
        empirical_beta = slope
        
        # Calculate between-participant variance for this gender (vectorized)
        gender_data = results_finishers[results_finishers[gender_col] == gender].copy()
        
        # For each participant, find their pace closest to marathon distance
        gender_data['dist_from_marathon'] = np.abs(gender_data[distance_col] - reference_distance)
        
        # Get the row with minimum distance to marathon for each participant
        closest_to_marathon = gender_data.loc[gender_data.groupby(participant_col)['dist_from_marathon'].idxmin()]
        participant_paces = closest_to_marathon[time_col].values / 60000 / closest_to_marathon[distance_col].values
        
        empirical_log_pace_sd = np.std(np.log(participant_paces))
        
        priors[gender] = {
            'log_marathon_pace': float(empirical_log_marathon_pace),
            'marathon_pace': float(np.exp(empirical_log_marathon_pace)),
            'beta': float(empirical_beta),
            'log_pace_sd': float(empirical_log_pace_sd),
            'n_observations': int(gender_mask.sum())
        }

    return priors


def calculate_pace_priors_pooled(
    results_finishers: pd.DataFrame,
    reference_distance: float = 26.2,
    distance_col: str = 'distance_miles',
    time_col: str = 'time_ms',
    participant_col: str = 'participant_id'
) -> Dict[str, float]:
    """
    Calculate pooled (gender-agnostic) pace priors using power law model.
    
    Power law model: log(pace) = log(pace_ref) + beta * log(d / d_ref)
    
    This is a simplified version that pools all finishers together without
    distinguishing by gender. Useful for models that don't include gender effects.
    
    Args:
        results_finishers: DataFrame with finisher results
        reference_distance: Reference distance in miles (default: marathon = 26.2)
        distance_col: Column name for distance in miles
        time_col: Column name for finish time in milliseconds
        participant_col: Column name for participant identifier
        
    Returns:
        dict: {
            'log_marathon_pace': float,  # log(pace) at reference distance
            'marathon_pace': float,      # pace at reference distance (min/mi)
            'beta': float,                # Power law exponent
            'log_pace_sd': float,        # Between-participant log-pace SD
            'n_observations': int        # Total number of observations
        }
    """
    # Calculate pace for all finishers (pooled across genders)
    observed_paces = results_finishers[time_col].values / 60000 / results_finishers[distance_col].values
    finisher_distances = results_finishers[distance_col].values
    
    # Calculate empirical pace at marathon distance
    marathon_mask = np.abs(finisher_distances - reference_distance) < 2
    if marathon_mask.sum() > 0:
        empirical_marathon_pace = np.median(observed_paces[marathon_mask])
        empirical_log_marathon_pace = np.log(empirical_marathon_pace)
    else:
        empirical_log_marathon_pace = np.log(np.median(observed_paces))
    
    # Calculate empirical distance exponent (beta) using linear regression
    log_distance_ratio = np.log(finisher_distances / reference_distance)
    log_pace = np.log(observed_paces)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_distance_ratio, log_pace)
    empirical_beta = slope
    
    # Calculate between-participant variance (pooled)
    results_finishers_copy = results_finishers.copy()
    results_finishers_copy['dist_from_marathon'] = np.abs(results_finishers_copy[distance_col] - reference_distance)
    
    # Get the row with minimum distance to marathon for each participant
    closest_to_marathon = results_finishers_copy.loc[
        results_finishers_copy.groupby(participant_col)['dist_from_marathon'].idxmin()
    ]
    participant_paces = closest_to_marathon[time_col].values / 60000 / closest_to_marathon[distance_col].values
    
    empirical_log_pace_sd = np.std(np.log(participant_paces))
    
    priors = {
        'log_marathon_pace': float(empirical_log_marathon_pace),
        'marathon_pace': float(np.exp(empirical_log_marathon_pace)),
        'beta': float(empirical_beta),
        'log_pace_sd': float(empirical_log_pace_sd),
        'n_observations': int(len(results_finishers))
    }

    return priors


def calculate_hierarchical_variance_priors(
    results_finishers: pd.DataFrame,
    empirical_priors_pace: Dict[str, Dict[str, float]],
    reference_distance: float = 26.2,
    course_col: str = 'name',
    race_id_col: str = 'event_distance_id',
    gender_col: str = 'gender',
    distance_col: str = 'distance_miles',
    time_col: str = 'time_ms'
) -> Dict[str, float]:
    """
    Calculate empirical variance components for hierarchical model.
    
    Estimates sigma_course and sigma_race from nested variance structure,
    after removing gender and distance effects.
    
    Args:
        results_finishers: DataFrame with finisher results
        empirical_priors_pace: Output from calculate_pace_priors (needed for residuals)
        reference_distance: Reference distance in miles (default: marathon = 26.2)
        course_col: Column name for course identifier (default: 'name')
        race_id_col: Column name for race identifier (nested in course)
        gender_col: Column name for gender
        distance_col: Column name for distance in miles
        time_col: Column name for finish time in milliseconds
        
    Returns:
        dict: {
            'sigma_course': float,  # Between-course standard deviation
            'sigma_race': float     # Between-race (within-course) standard deviation
        }
    """
    # Calculate race-level statistics (removing distance and gender effects)
    race_stats = []
    
    for race_id in results_finishers[race_id_col].unique():
        race_data = results_finishers[results_finishers[race_id_col] == race_id]
        
        if len(race_data) > 5:  # Minimum observations per race
            race_distances = race_data[distance_col].values
            race_paces = race_data[time_col].values / 60000 / race_distances
            race_genders = race_data[gender_col].values
            
            # Remove distance and gender effects
            log_distance_ratio = np.log(race_distances / reference_distance)
            log_pace = np.log(race_paces)
            
            # Calculate expected log pace using gender-specific parameters (vectorized)
            expected_log_pace = np.array([
                empirical_priors_pace[g]['log_marathon_pace'] + empirical_priors_pace[g]['beta'] * log_dist
                for g, log_dist in zip(race_genders, log_distance_ratio)
            ])
            
            # Race effect is the median residual
            race_effect = np.median(log_pace - expected_log_pace)
            race_stats.append(race_effect)
    
    empirical_sigma_race = np.std(race_stats)
    
    # For sigma_course, we'd need course-level aggregation
    # For now, use a fraction of race variance as a reasonable prior
    empirical_sigma_course = empirical_sigma_race * 0.7  # Course effects typically smaller than race-year effects

    return {
        'sigma_course': float(empirical_sigma_course),
        'sigma_race': float(empirical_sigma_race)
    }


def calculate_hierarchical_variance_priors_pooled(
    results_finishers: pd.DataFrame,
    empirical_priors_pace: Dict[str, float],
    reference_distance: float = 26.2,
    course_col: str = 'name',
    race_id_col: str = 'event_distance_id',
    distance_col: str = 'distance_miles',
    time_col: str = 'time_ms'
) -> Dict[str, float]:
    """
    Calculate empirical variance components for hierarchical model (pooled version).
    
    Estimates sigma_course and sigma_race from nested variance structure,
    after removing distance effects. Does not account for gender.
    
    Args:
        results_finishers: DataFrame with finisher results
        empirical_priors_pace: Output from calculate_pace_priors_pooled (pooled priors)
        reference_distance: Reference distance in miles (default: marathon = 26.2)
        course_col: Column name for course identifier (default: 'name')
        race_id_col: Column name for race identifier (nested in course)
        distance_col: Column name for distance in miles
        time_col: Column name for finish time in milliseconds
        
    Returns:
        dict: {
            'sigma_course': float,  # Between-course standard deviation
            'sigma_race': float     # Between-race (within-course) standard deviation
        }
    """
    # Calculate race-level statistics (removing distance effects only, no gender)
    race_stats = []
    
    for race_id in results_finishers[race_id_col].unique():
        race_data = results_finishers[results_finishers[race_id_col] == race_id]
        
        if len(race_data) > 5:  # Minimum observations per race
            race_distances = race_data[distance_col].values
            race_paces = race_data[time_col].values / 60000 / race_distances
            
            # Remove distance effects (no gender adjustment)
            log_distance_ratio = np.log(race_distances / reference_distance)
            log_pace = np.log(race_paces)
            
            # Calculate expected log pace using pooled parameters
            expected_log_pace = (
                empirical_priors_pace['log_marathon_pace'] + 
                empirical_priors_pace['beta'] * log_distance_ratio
            )
            
            # Race effect is the median residual
            race_effect = np.median(log_pace - expected_log_pace)
            race_stats.append(race_effect)
    
    empirical_sigma_race = np.std(race_stats)
    
    # For sigma_course, use a fraction of race variance as a reasonable prior
    empirical_sigma_course = empirical_sigma_race * 0.7  # Course effects typically smaller than race-year effects

    return {
        'sigma_course': float(empirical_sigma_course),
        'sigma_race': float(empirical_sigma_race)
    }


def calculate_dnf_priors(
    results: pd.DataFrame,
    reference_distance: float = 26.2,
    finished_col: str = 'finished',
    distance_col: str = 'distance_miles',
    course_col: str = 'name'
) -> Dict[str, float]:
    """
    Calculate DNF (Did Not Finish) model priors.
    
    Logistic model: logit(P(DNF)) = mu_logit_dnf + beta_dist_dnf * log(d / d_ref)
    
    Args:
        results: DataFrame with all results (finishers + DNF)
        reference_distance: Reference distance for centering (default: marathon)
        finished_col: Column name for finish indicator (True/False)
        distance_col: Column name for distance in miles
        course_col: Column name for course identifier (default: 'name')
        
    Returns:
        dict: {
            'mu_logit_dnf': float,          # Baseline DNF log-odds at reference distance
            'beta_dist_dnf': float,         # Distance effect on DNF probability
            'sigma_course_dnf': float       # Course-level DNF variation
        }
    """
    # Calculate DNF rate at reference distance
    reference_races = results[
        (results[distance_col] >= reference_distance - 2) &
        (results[distance_col] <= reference_distance + 2)
    ].copy()
    
    dnf_rate_ref = (~reference_races[finished_col]).mean()
    
    # Avoid log(0) with small epsilon
    dnf_rate_ref = np.clip(dnf_rate_ref, 0.01, 0.99)
    mu_logit_dnf = float(np.log(dnf_rate_ref / (1 - dnf_rate_ref)))
    
    # Estimate distance effect using binned approach
    distance_bins = np.arange(10, 110, 10)
    binned_dnf_rates = []
    binned_log_distance_ratios = []
    
    for i in range(len(distance_bins) - 1):
        bin_data = results[
            (results[distance_col] >= distance_bins[i]) &
            (results[distance_col] < distance_bins[i + 1])
        ]
        if len(bin_data) >= 50:  # Minimum observations per bin
            dnf_rate = (~bin_data[finished_col]).mean()
            dnf_rate = np.clip(dnf_rate, 0.01, 0.99)  # Avoid log(0)
            
            binned_dnf_rates.append(np.log(dnf_rate / (1 - dnf_rate)))
            bin_center = (distance_bins[i] + distance_bins[i + 1]) / 2
            binned_log_distance_ratios.append(np.log(bin_center / reference_distance))
    
    # Simple linear regression on log-odds
    if len(binned_log_distance_ratios) > 1:
        beta_dist_dnf = float(np.polyfit(binned_log_distance_ratios, binned_dnf_rates, 1)[0])
    else:
        beta_dist_dnf = 0.5  # Default positive effect (DNF increases with distance)
    
    # Calculate course-level DNF variation
    course_dnf_rates = results.groupby(course_col)[finished_col].apply(lambda x: (~x).mean())
    course_dnf_rates = course_dnf_rates.clip(0.01, 0.99)
    course_logit_dnf = np.log(course_dnf_rates / (1 - course_dnf_rates))
    sigma_course_dnf = float(np.std(course_logit_dnf))

    return {
        'mu_logit_dnf': mu_logit_dnf,
        'beta_dist_dnf': beta_dist_dnf,
        'sigma_course_dnf': sigma_course_dnf
    }


def calculate_difficulty_dnf_coupling(
    results: pd.DataFrame,
    empirical_priors_pace: Dict[str, Dict[str, float]],
    reference_distance: float = 26.2,
    course_col: str = 'name',
    finished_col: str = 'finished',
    gender_col: str = 'gender',
    distance_col: str = 'distance_miles',
    time_col: str = 'time_ms'
) -> float:
    """
    Calculate empirical coupling between course difficulty and DNF rate (gamma parameter).
    
    Used in Model 5. Estimates correlation between course pace difficulty
    and DNF probability among courses that report DNFs.
    
    Args:
        results: DataFrame with all results (finishers + DNF)
        empirical_priors_pace: Output from calculate_pace_priors
        reference_distance: Reference distance in miles
        course_col: Column name for course identifier (default: 'name')
        finished_col: Column name for finish indicator
        gender_col: Column name for gender
        distance_col: Column name for distance
        time_col: Column name for finish time in milliseconds
        
    Returns:
        float: gamma (difficulty-DNF coupling coefficient)
    """
    # Get finishers only for difficulty calculation
    results_finishers = results[results[finished_col]].copy()
    
    # Calculate pace residuals (proxy for course difficulty)
    results_finishers['pace_min_mi'] = results_finishers[time_col] / 60000 / results_finishers[distance_col]
    results_finishers['log_pace'] = np.log(results_finishers['pace_min_mi'])
    results_finishers['log_dist_ratio'] = np.log(results_finishers[distance_col] / reference_distance)
    
    # Simple gender-specific baseline (for demonstration)
    gender_medians = results_finishers.groupby(gender_col)['log_pace'].median()
    results_finishers['log_pace_residual'] = results_finishers.apply(
        lambda x: x['log_pace'] - gender_medians[x[gender_col]], axis=1
    )
    
    # Aggregate to course level
    course_stats = results.groupby(course_col).agg({
        finished_col: lambda x: (~x).sum()  # DNF count
    }).rename(columns={finished_col: 'dnf_count'})
    
    # Add total results count
    course_stats['n_results'] = results.groupby(course_col).size()
    course_stats['dnf_rate'] = course_stats['dnf_count'] / course_stats['n_results']
    
    # Calculate median pace residual per course (from finishers)
    course_difficulty = results_finishers.groupby(course_col)['log_pace_residual'].median()
    
    # Filter to courses with at least one DNF (reporting courses)
    reporting_courses = course_stats[course_stats['dnf_count'] > 0].index
    valid_courses = reporting_courses.intersection(course_difficulty.index)
    
    if len(valid_courses) < 10:
        print(f"⚠️  Only {len(valid_courses)} courses with both DNF data and finisher data")
        return 0.3  # Default weak positive coupling
    
    # Get difficulty and DNF rate for valid courses
    difficulty_values = course_difficulty.loc[valid_courses].values
    dnf_rates = course_stats.loc[valid_courses, 'dnf_rate'].values
    
    # Clip DNF rates to avoid log(0)
    dnf_rates = np.clip(dnf_rates, 0.01, 0.99)
    logit_dnf_rates = np.log(dnf_rates / (1 - dnf_rates))
    
    # Calculate correlation (this is gamma)
    gamma = float(np.corrcoef(difficulty_values, logit_dnf_rates)[0, 1])

    return gamma


def calculate_reporting_priors(
    results: pd.DataFrame,
    reference_distance: float = 26.2,
    course_col: str = 'name',
    finished_col: str = 'finished',
    distance_col: str = 'distance_miles'
) -> Dict[str, float]:
    """
    Calculate reporting probability prior (SIMPLE BASELINE).
    
    Model: psi = sigmoid(logit_psi_baseline) = P(course reports DNFs to UltraSignup)
    
    Key insight: Reporting is an organizational policy, not race-size dependent.
    The Bayesian likelihood automatically handles size effects:
    - Small race (n=20) with 0 DNFs → Could be natural zero (uncertain)
    - Large race (n=500) with 0 DNFs → Almost impossible if reporting (confident non-reporting)
    
    The model infers reporting status via P(0 DNFs | reports, n) = (1 - p_DNF)^n,
    which decreases exponentially with n, naturally making large races with 0 DNFs suspicious.
    
    Args:
        results: DataFrame with all results (finishers + DNF)
        reference_distance: Reference distance in miles (not used, kept for compatibility)
        course_col: Column name for course identifier (default: 'name')
        finished_col: Column name for finish indicator
        distance_col: Column name for distance in miles (not used)
        
    Returns:
        dict: {
            'logit_psi': float,  # Baseline reporting log-odds (constant)
            'psi': float         # Baseline reporting probability
        }
    """
    # For each course, determine if it reports ANY DNFs
    reporting_by_course = results.groupby(course_col).agg({
        finished_col: lambda x: (~x).sum()  # DNF count
    }).reset_index()
    reporting_by_course.columns = [course_col, 'dnf_count']
    
    reporting_by_course['reports_dnf'] = (reporting_by_course['dnf_count'] > 0).astype(int)
    
    # Get course distances and sizes
    course_info = results.groupby(course_col).agg({
        distance_col: 'first',
        finished_col: 'size'  # Total results
    }).reset_index()
    course_info.columns = [course_col, 'distance_miles', 'n_results']
    
    reporting_by_course = reporting_by_course.merge(course_info, on=course_col)
    
    # CRITICAL INSIGHT: Calculate psi from LONG DISTANCE races ONLY
    # Why? Short races have low P(any DNF) even if reporting:
    #   - 10K (p_dnf=10%, n=30): P(0 DNFs) = 0.90^30 = 4% (ambiguous!)
    #   - 100K (p_dnf=25%, n=100): P(0 DNFs) = 0.75^100 ≈ 0% (very suspicious!)
    #
    # For ultra races (≥100K = 62 miles), almost ALL reporting courses will have DNFs
    # So: observed_rate ≈ true_psi (detection power ≈ 100%)
    
    ultra_courses = reporting_by_course[reporting_by_course['distance_miles'] >= 62.0]
    
    if len(ultra_courses) >= 30:  # Need enough data
        # For ultra races, observed_rate ≈ true_psi (P(any DNF) ≈ 1)
        empirical_psi = ultra_courses['reports_dnf'].mean()
        n_ultra = len(ultra_courses)
        n_ultra_reporting = ultra_courses['reports_dnf'].sum()
        
        print(f"\n📊 Reporting probability prior (from ≥100K races):")
        print(f"   Using {n_ultra_reporting}/{n_ultra} ultra courses (≥62 miles)")
        print(f"   At these distances, P(any DNF | reports) ≈ 1.0")
        print(f"   → observed rate ≈ true organizational psi")
    else:
        # Fallback: use all courses, but this underestimates true psi
        empirical_psi = reporting_by_course['reports_dnf'].mean()
        print(f"\n📊 Reporting probability prior (all courses - underestimate):")
        print(f"   WARNING: Not enough ultra races, using all courses")
        print(f"   This will UNDERESTIMATE true psi due to low detection power")
    
    # Overall observed rate for comparison
    observed_overall = reporting_by_course['reports_dnf'].mean()
    
    # Convert to logit scale
    empirical_psi = np.clip(empirical_psi, 0.01, 0.99)  # Avoid log(0)
    logit_psi = float(np.log(empirical_psi / (1 - empirical_psi)))
    
    print(f"\n   logit_psi: {logit_psi:.4f}")
    print(f"   psi: {empirical_psi:.1%} (TRUE organizational reporting probability)")
    print(f"   Observed overall: {observed_overall:.1%} (includes detection power effects)")
    print(f"   Difference: {empirical_psi - observed_overall:+.1%} (detection power bias)")
    print(f"\n   Size effects handled naturally by Bayesian likelihood:")
    print(f"   - Small races (n=20) with 0 DNFs: ambiguous (natural zeros plausible)")
    print(f"   - Large races (n=500) with 0 DNFs: strong evidence of non-reporting")
    
    return {
        'logit_psi': logit_psi,
        'psi': empirical_psi
    }

