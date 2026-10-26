"""
Utils package for UltraSignup Bayesian Analysis

This package provides utilities for loading, processing, and analyzing
ultrarunning race data from the UltraSignup database.
"""

from utils.data_processing import (
    load_results,
    process_results,
    filter_races_with_dnfs,
    is_running_race,
    extract_distance_miles,
)

from utils.data_subsetting import *

from utils.empirical_priors import (
    calculate_gender_marathon_pace,
    calculate_gender_distance_exponent,
    calculate_course_variation,
    calculate_dnf_priors,
    calculate_pace_priors,
    calculate_hierarchical_variance_priors,
    calculate_difficulty_dnf_coupling,
    calculate_reporting_priors,
)

from utils.kcore_subsetting import (
    evaluate_all_kcore_candidates,
    subset_kcore_data,
)

from utils.mcmc_notifications import (
    notify_mcmc_start,
    notify_mcmc_complete,
    notify_mcmc_error,
    build_mcmc_summary_string,
)

from utils.utils import plot_posterior_diagnostics

__all__ = [
    # data_processing
    "load_results",
    "process_results",
    "filter_races_with_dnfs",
    "is_running_race",
    "extract_distance_miles",
    # empirical_priors
    "calculate_gender_marathon_pace",
    "calculate_gender_distance_exponent",
    "calculate_course_variation",
    "calculate_dnf_priors",
    "calculate_pace_priors",
    "calculate_hierarchical_variance_priors",
    "calculate_difficulty_dnf_coupling",
    "calculate_reporting_priors",
    # kcore_subsetting
    "evaluate_all_kcore_candidates",
    "subset_kcore_data",
    # mcmc_notifications
    "notify_mcmc_start",
    "notify_mcmc_complete",
    "notify_mcmc_error",
    "build_mcmc_summary_string",
    # utils
    "plot_posterior_diagnostics",
]
