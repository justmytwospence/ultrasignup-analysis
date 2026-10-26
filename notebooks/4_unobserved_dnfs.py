import marimo

__generated_with = "0.23.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model 5b: Reporting-Aware Shared Difficulty Model

    ## Overview

    This model predicts ultramarathon finish times and DNF (Did Not Finish) rates while accounting for systematic reporting bias in the data. The key innovation is a **shared hierarchical difficulty structure** that couples finish times with DNF probabilities, plus a **reporting probability model** that corrects for courses that don't report DNFs to UltraSignup.

    ### Core Components

    **Hierarchical Difficulty:**
    - Course baselines capture the average difficulty of each race course
    - Race-specific deviations model year-to-year variation (weather, course changes, etc.)
    - The same difficulty parameters influence both finish times and DNF rates

    **Reporting Bias Correction:**
    - Many courses don't report DNFs, creating two problems:
      1. DNF rates appear artificially low (often 0%)
      2. Finish times are biased fast (missing slower runners who DNF'd)
    - Model learns distance-dependent reporting probabilities
    - Corrects both DNF estimates and finish time distributions

    ### Data Sampling

    The model uses a hierarchical sampling strategy to balance coverage and statistical power:
    - **Course selection:** Popular courses with multiple race instances
    - **Race filtering:** Races with sufficient finisher counts for reliable estimation
    - **Distance range:** Focus on standard ultra distances (50K through 100+ miles)
    - **Total observations:** Thousands of race instances across hundreds of unique courses

    ### Key Hyperparameters

    **Finish Time Model:**
    - Gender-specific baseline paces
    - Distance scaling coefficients
    - Hierarchical variance components (course-level and race-level)

    **DNF Model:**
    - Baseline DNF rate (distance-adjusted)
    - Distance scaling for DNF probability
    - Difficulty-DNF coupling parameter (γ)

    **Reporting Model:**
    - Baseline reporting probability
    - Distance-dependent reporting adjustment
    - Tighter priors to prevent unrealistic reporting rates

    ### Why This Model?

    Without reporting correction, non-reporting courses appear artificially "easy" (low DNF, fast times). This model disentangles true race difficulty from reporting behavior, enabling fair comparison across all courses and accurate predictions for new races.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preamble
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup

    Import packages, utilities, and configure analysis environment.
    """)
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings('ignore')

    import os
    import numpy as np
    import pandas as pd
    import pymc as pm
    import arviz as az
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from scipy.optimize import minimize
    from scipy.stats import linregress
    import graphviz
    from pathlib import Path
    import time
    from datetime import datetime

    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Configure ArviZ
    az.style.use("arviz-darkgrid")

    # Load our custom utilities
    from utils.kcore_subsetting import subset_kcore_data
    from utils.data_processing import load_results, process_results
    from utils.mcmc_notifications import (
        notify_mcmc_start, 
        notify_mcmc_complete, 
        notify_mcmc_error
    )
    from utils.empirical_priors import (
        calculate_pace_priors,
        calculate_hierarchical_variance_priors,
        calculate_dnf_priors,
        calculate_difficulty_dnf_coupling,
        calculate_reporting_priors
    )

    # Suppress PyMC sampling messages
    import logging
    logging.getLogger("pymc").setLevel(logging.ERROR)

    print("✓ All imports successful")
    print(f"PyMC version: {pm.__version__}")
    print(f"ArviZ version: {az.__version__}")
    return (
        Patch,
        Path,
        az,
        calculate_difficulty_dnf_coupling,
        calculate_dnf_priors,
        calculate_hierarchical_variance_priors,
        calculate_pace_priors,
        calculate_reporting_priors,
        linregress,
        load_results,
        np,
        os,
        pd,
        plt,
        pm,
        process_results,
        subset_kcore_data,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Processing

    Load race results database, filter to M/F genders, and create course identifiers (name + distance pooled across years).
    """)
    return


@app.cell
def _(load_results, process_results):
    # Load and process ALL race results
    results = load_results()
    results = process_results(results)

    # Filter to M and F genders only (exclude X or missing values)
    # This is the only filtering we do - NO subsetting yet
    results = results[results['gender'].isin(['M', 'F'])].copy()

    # Create course identifier using race name + distance (pooled across years)
    results['course_id'] = (
        results['name'].astype(str) + '||' + 
        results['distance_miles'].round(1).astype(str)
    )

    print(f"Total observations: {len(results):,}")
    print(f"Gender distribution: {results['gender'].value_counts().to_dict()}")
    print(f"Participants: {results['participant_id'].nunique():,}")
    print(f"Courses: {results.groupby(['name', 'distance_miles']).ngroups:,}")
    print(f"Races (course-year combinations): {results['event_distance_id'].nunique():,}")
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Justification
    """)
    return


@app.cell
def _(linregress, np, plt, results):
    results_finishers = results[results['finished']].copy()
    reference_distance = 26.2
    results_finishers['pace_min_mi'] = results_finishers['time_ms'] / 60000 / results_finishers['distance_miles']
    results_finishers['log_pace'] = np.log(results_finishers['pace_min_mi'])
    results_finishers['log_dist_ratio'] = np.log(results_finishers['distance_miles'] / reference_distance)
    gender_medians = results_finishers.groupby('gender')['log_pace'].median()
    results_finishers['log_pace_residual'] = results_finishers.apply(lambda x: x['log_pace'] - gender_medians[x['gender']], axis=1)
    course_stats = results.groupby('course_id').agg({'finished': lambda x: (~x).sum(), 'event_distance_id': 'count'}).rename(columns={'finished': 'dnf_count', 'event_distance_id': 'n_results'})
    course_median_pace = results_finishers.groupby('course_id')['log_pace_residual'].median()
    course_stats['median_log_pace'] = course_median_pace
    course_stats['dnf_rate'] = course_stats['dnf_count'] / course_stats['n_results']
    course_stats = course_stats[course_stats['n_results'] >= 20].copy()
    course_stats_reporting = course_stats[course_stats['dnf_count'] > 0].copy()
    print(f'=== DNF REPORTING STATUS ===')
    print(f'Total courses (n >= 20): {len(course_stats):,}')
    print(f'  Reporting courses (DNF > 0): {len(course_stats_reporting):,} ({100 * len(course_stats_reporting) / len(course_stats):.1f}%)')
    print(f'  Non-reporting courses (DNF = 0): {len(course_stats) - len(course_stats_reporting):,} ({100 * (len(course_stats) - len(course_stats_reporting)) / len(course_stats):.1f}%)')
    print(f'\nUsing reporting courses only for γ estimation (unbiased)')
    course_stats_reporting['dnf_pct'] = course_stats_reporting['dnf_rate'] * 100
    course_stats_reporting['pace_pct_diff'] = (np.exp(course_stats_reporting['median_log_pace']) - 1) * 100
    course_stats_reporting['dnf_rate_clipped'] = course_stats_reporting['dnf_rate'].clip(0.001, 0.999)
    course_stats_reporting['logit_dnf'] = np.log(course_stats_reporting['dnf_rate_clipped'] / (1 - course_stats_reporting['dnf_rate_clipped']))
    from scipy.stats import pearsonr
    corr_pct, p_value_pct = pearsonr(course_stats_reporting['pace_pct_diff'], course_stats_reporting['dnf_pct'])
    empirical_gamma_m5, _intercept, _r_value, _p_value_reg, _std_err = linregress(course_stats_reporting['median_log_pace'], course_stats_reporting['logit_dnf'])
    print(f'\n=== JUSTIFICATION: DIFFICULTY-DNF RELATIONSHIP ===')
    print(f'Reporting courses analyzed: {len(course_stats_reporting):,}')
    print(f'Correlation (% scales): {corr_pct:.3f} (p={p_value_pct:.3e})')
    print(f'Empirical γ (logit scale): {empirical_gamma_m5:.3f}')
    _fig, _ax = plt.subplots(1, 1, figsize=(10, 7))
    _ax.scatter(course_stats_reporting['pace_pct_diff'], course_stats_reporting['dnf_pct'], alpha=0.5, s=course_stats_reporting['n_results'] / 5, color='steelblue', edgecolor='black', linewidth=0.5)
    _slope_pct, _intercept_pct = np.polyfit(course_stats_reporting['pace_pct_diff'], course_stats_reporting['dnf_pct'], deg=1)
    _x_line = np.array([course_stats_reporting['pace_pct_diff'].min(), course_stats_reporting['pace_pct_diff'].max()])
    _y_line = _slope_pct * _x_line + _intercept_pct
    _ax.plot(_x_line, _y_line, 'r--', linewidth=2, label=f'Trend: {_slope_pct:.2f}% DNF per 1% slower\n(r = {corr_pct:.3f}, p = {p_value_pct:.3e})')
    _ax.set_xlabel('Course Difficulty (% slower than median)', fontsize=12)
    _ax.set_ylabel('DNF Rate (%)', fontsize=12)
    _ax.set_title('Empirical Justification: Course Difficulty → DNF Rate\n' + 'Model 5b - Shared Difficulty Structure (Reporting Courses Only)', fontsize=14, fontweight='bold')
    _ax.legend(fontsize=11, loc='upper left')
    _ax.grid(alpha=0.3)
    _ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    _ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    median_dnf = course_stats_reporting['dnf_pct'].median()
    _textstr = f'Reporting courses: {len(course_stats_reporting):,}\n'
    _textstr = _textstr + f'Median DNF rate: {median_dnf:.1f}%\n\n'
    if corr_pct > 0.5:
        _textstr = _textstr + '✓ Strong positive correlation\n  Harder courses → more DNFs'
    elif corr_pct > 0.3:
        _textstr = _textstr + '✓ Moderate positive correlation\n  Supports shared structure'
    else:
        _textstr = _textstr + '⚠ Weak correlation\n  Shared structure may not fit well'
    _props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    _ax.text(0.97, 0.03, _textstr, transform=_ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=_props)
    plt.tight_layout()
    plt.show()
    print(f'\n=== REGRESSION STATISTICS (Model parameterization) ===')
    print(f'Linear model on logit scale: logit(DNF) ~ γ * log_pace_residual + intercept')
    print(f'  Slope (γ_empirical): {empirical_gamma_m5:.3f}')
    print(f'  Intercept: {_intercept:.3f}')
    print(f'  R²: {_r_value ** 2:.3f}')
    print(f'  P-value: {_p_value_reg:.3e}')
    print(f'\n=== INTERPRETATION (Percentage scales) ===')
    print(f'✓ For every 1% slower pace, DNF rate increases by ~{_slope_pct:.2f}%')
    print(f'✓ Median DNF rate: {median_dnf:.1f}%')
    print(f"✓ DNF rate range: {course_stats_reporting['dnf_pct'].min():.1f}% - {course_stats_reporting['dnf_pct'].max():.1f}%")
    print(f"✓ Shared difficulty structure is {('well' if corr_pct > 0.3 else 'weakly')} supported by data")
    print(f'\nNote: Using reporting courses only ensures unbiased γ estimation.')
    print(f'      Model accounts for non-reporting via ψ (reporting probability).')
    return (pearsonr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specification
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Empirical Priors

    Calculate data-driven priors for finish time (gender-specific power law), hierarchical variance, DNF rates, and γ (difficulty-DNF coupling).
    """)
    return


@app.cell
def _(
    calculate_difficulty_dnf_coupling,
    calculate_dnf_priors,
    calculate_hierarchical_variance_priors,
    calculate_pace_priors,
    calculate_reporting_priors,
    results,
):
    # ============================================================================
    # CALCULATE ALL EMPIRICAL PRIORS FOR MODEL 5 USING MODULAR FUNCTIONS
    print('=' * 80)
    print('CALCULATING EMPIRICAL PRIORS FOR MODEL 5')
    print('=' * 80)
    results_finishers_1 = results[results['finished']].copy()
    reference_distance_1 = 26.2
    pace_priors = calculate_pace_priors(results_finishers_1, reference_distance_1)
    # Get finishers for pace calculation
    hierarchical_priors = calculate_hierarchical_variance_priors(results_finishers_1, pace_priors, reference_distance_1)
    dnf_priors = calculate_dnf_priors(results, reference_distance_1)  # Marathon
    gamma = calculate_difficulty_dnf_coupling(results, pace_priors, reference_distance_1)
    # Calculate each type of prior using modular functions
    reporting_priors = calculate_reporting_priors(results, reference_distance_1)
    print('\n' + '=' * 80)
    print('✓ All empirical priors calculated successfully')
    print('=' * 80)
    return (
        dnf_priors,
        gamma,
        hierarchical_priors,
        pace_priors,
        reference_distance_1,
        reporting_priors,
        results_finishers_1,
    )


@app.cell
def _(dnf_priors, gamma, hierarchical_priors, pace_priors, reporting_priors):
    # Extract individual prior values for use in model definition
    # (maintaining backward compatibility with existing code)
    prior_sigma_course = hierarchical_priors['sigma_course']
    # Hierarchical variance
    prior_sigma_race = hierarchical_priors['sigma_race']
    prior_mu_logit_dnf = dnf_priors['mu_logit_dnf']
    prior_beta_dist_dnf = dnf_priors['beta_dist_dnf']
    # DNF model baseline
    prior_gamma = gamma
    prior_logit_psi = reporting_priors['logit_psi']
    prior_psi = reporting_priors['psi']
    # Difficulty-DNF coupling
    empirical_priors_m5 = pace_priors
    print('\n=== EXTRACTED PRIORS (READY FOR MODEL) ===')
    # Reporting probability (SIMPLE BASELINE - size effects handled by likelihood)
    print(f'\n1. HIERARCHICAL VARIANCE:')
    print(f'   sigma_course: {prior_sigma_course:.3f}')
    print(f'   sigma_race: {prior_sigma_race:.3f}')
    # Create empirical_priors_m5 dict for gender-specific pace priors (for model definition)
    print(f'\n2. DNF MODEL BASELINE:')
    print(f'   mu_logit_dnf: {prior_mu_logit_dnf:.3f}')
    print(f'   beta_dist_dnf: {prior_beta_dist_dnf:.3f}')
    print(f'\n3. DIFFICULTY-DNF COUPLING:')
    print(f'   gamma: {prior_gamma:.3f}')
    print(f'\n4. REPORTING PROBABILITY (BASELINE ONLY):')
    print(f'   logit_psi: {prior_logit_psi:.3f}')
    print(f'   psi: {prior_psi:.1%} (global baseline)')
    print(f'   Note: Size effects emerge naturally from Bayesian likelihood')
    print(f'\n5. FINISH TIME MODEL (GENDER-SPECIFIC):')
    for _gender in ['M', 'F']:
        priors = empirical_priors_m5[_gender]
        print(f"   {_gender}: log_marathon_pace={priors['log_marathon_pace']:.3f}, beta={priors['beta']:.3f}, log_pace_sd={priors['log_pace_sd']:.3f}")
    return (
        empirical_priors_m5,
        prior_beta_dist_dnf,
        prior_gamma,
        prior_logit_psi,
        prior_mu_logit_dnf,
        prior_sigma_course,
        prior_sigma_race,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### K-Core Data Subsetting

    Use k-core decomposition to find a densely-connected subset of runners and courses for efficient Bayesian inference. The k-core approach:

    1. **Finds symmetric (k,k)-core**: All runners have ≥k results, all courses have ≥k runners
    2. **Adds tiered closure**: Selectively includes sparse entities highly connected to core
    3. **Flags data types**: `in_kcore` (dense, complete) vs `in_closure` (sparse, linked to core)

    This enables **two-stage modeling**:
    - **Stage 1 MCMC**: High-quality posteriors on k-core entities (complete data)
    - **Stage 2 MAP**: Fix core params, estimate closure params using Stage 1 as priors
    """)
    return


@app.cell
def _(results, subset_kcore_data):
    # K-core parameters. Default = (3, 629) for dev iteration.
    # (3, 840): 11 courses → unidentifiable, R-hat ~4.5
    # (3, 423): 266 courses + ~222K runner effects → intractable on macOS (>3h)
    # (3, 629): ~93 courses + ~50K runners → middle ground for dev
    alpha = 3
    beta = 629
    results_1 = subset_kcore_data(results, alpha=alpha, beta=beta)
    # Apply k-core flagging with course-completion closure
    model_data = results_1[results_1['in_kcore'] | results_1['in_closure']]
    print(f"\n{'=' * 80}")
    print(f'SUBSET SUMMARY')
    print(f"{'=' * 80}")
    print(f'K-core subset (alpha={alpha}, beta={beta}):')
    print(f"  K-core results: {results_1['in_kcore'].sum():,}")
    print(f"  Closure results: {results_1['in_closure'].sum():,}")
    print(f'  Total for modeling: {len(model_data):,} (down from {len(results_1):,})')
    print(f"  K-core courses: {results_1[results_1['in_kcore']]['name'].nunique():,}")
    print(f"  K-core runners: {results_1[results_1['in_kcore']]['participant_id'].nunique():,}")
    print(f"\nParticipants: {model_data['participant_id'].nunique():,}")
    print(f"Courses: {model_data.groupby(['name', 'distance_miles']).ngroups:,}")
    print(f"Races: {model_data['event_distance_id'].nunique():,}")
    return alpha, beta, model_data, results_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generative Model

    **Complete mathematical specification with hierarchical structure, shared difficulty parameters, and reporting bias correction.**

    ---

    #### Hierarchical Difficulty Structure

    The same latent difficulty variables drive both finish times and DNF probabilities:

    ```
    course_baseline[c] ~ Normal(0, σ_course)
    race_difficulty[r] ~ Normal(course_baseline[parent_course[r]], σ_race)
    ```

    Where:
    - `σ_course` controls variation across different race courses
    - `σ_race` controls year-to-year variation within the same course

    ---

    #### Finish Time Model

    For finisher `i` in race `r` at distance `d`:

    ```
    log_pace[i] = mu_pace[gender[i]] + beta[gender[i]] * log(d / d_marathon) + race_difficulty[r] + ε
    ε ~ Normal(0, σ_obs)
    ```

    **Priors:**
    - `mu_pace[gender]` ~ Normal(empirical, 0.2) - Gender-specific baseline log-pace at marathon distance
    - `beta[gender]` ~ Normal(empirical, 0.1) - Gender-specific distance scaling
    - `σ_obs` ~ HalfNormal(empirical) - Within-race variation
    - `σ_course` ~ HalfNormal(empirical) - Between-course variation
    - `σ_race` ~ HalfNormal(empirical) - Between-race (within-course) variation

    All empirical priors derived from data to provide informed regularization.

    ---

    #### DNF Model with Difficulty Coupling

    For result `i` in race `r` at distance `d`:

    ```
    logit(P(DNF)[i]) = mu_dnf + beta_dist * log(d / d_marathon) + γ * race_difficulty[r]
                                                                    ^^^^^^^^^^^^^^^^^^^
                                                                    Shared with finish times!
    ```

    **Priors:**
    - `mu_dnf` ~ Normal(empirical, 1.0) - Baseline logit-DNF at marathon distance
    - `beta_dist` ~ Normal(empirical, 0.5) - Distance effect on DNF
    - `γ` ~ Normal(empirical, 0.5) - **Difficulty-DNF coupling** (harder races → more DNFs)

    ---

    #### Reporting Probability Model

    Longer races are more likely to report DNFs (ultras track carefully, road races often don't):

    ```
    logit(psi[d]) = logit_psi_baseline + beta_psi_dist * log(d / d_marathon)
    ```

    **Priors:**
    - `logit_psi_baseline` ~ Normal(-0.4, 0.3) → psi_marathon ≈ 40% (95% CI: 30-55%)
    - `beta_psi_dist` ~ Normal(1.0, 0.5) → Positive slope (longer = higher reporting)

    **Course-specific reporting:**
    ```
    reporting_prob[c] = 1.0                    if observed_dnf_count[c] > 0
                      = psi[distance[c]]       if observed_dnf_count[c] = 0
    ```

    Courses with ANY observed DNFs definitely report. Courses with ZERO observed DNFs use the distance-dependent prior.

    ---

    #### Zero-Inflated Likelihood

    **For DNF observations:**
    ```
    P(observe finish) = reporting_prob[c] * P(true finish) + (1 - reporting_prob[c]) * 1.0
                      = reporting_prob[c] * (1 - P(DNF)) + (1 - reporting_prob[c])
    ```

    This creates a mixture:
    - With probability `reporting_prob[c]`: observe true outcome
    - With probability `1 - reporting_prob[c]`: always observe "finish" (DNF not recorded)

    **For finish times (censoring correction):**

    Courses with zero observed DNFs have biased finish time distributions (missing slow DNFs). Add log-likelihood penalty:

    ```
    For courses with observed_dnf_count[c] = 0:
      censoring_penalty = -n_finishers[c] * log(E[P(finish)] * psi[d])
    ```

    Where `E[P(finish)]` is the expected finish probability for that course's finishers under the model. This corrects the finish time likelihood for the censored slow finishers who DNF'd but weren't recorded.

    ---

    #### Model Summary

    **Total parameters:**
    - 10 hyperparameters (pace, DNF, reporting models)
    - ~1,000 course baseline effects
    - ~2,500 race difficulty effects

    The model jointly estimates finish time distributions and DNF probabilities while correcting for reporting bias, enabling fair comparison across all courses regardless of reporting practices.
    """)
    return


@app.cell
def _(model_data, np, reference_distance_1):
    model_data['course_id'] = model_data['name'].astype(str) + '||' + model_data['distance_miles'].round(1).astype(str)
    unique_courses_m5 = model_data['course_id'].unique()
    course_id_to_idx_m5 = {cid: _idx for _idx, cid in enumerate(unique_courses_m5)}
    model_data['course_idx'] = model_data['course_id'].map(course_id_to_idx_m5)
    unique_races_m5 = model_data['event_distance_id'].unique()
    race_id_to_idx_m5 = {rid: _idx for _idx, rid in enumerate(unique_races_m5)}
    model_data['race_idx'] = model_data['event_distance_id'].map(race_id_to_idx_m5)
    race_to_course_m5 = np.array([course_id_to_idx_m5[model_data[model_data['event_distance_id'] == rid]['course_id'].iloc[0]] for rid in unique_races_m5])
    unique_genders = np.array(['M', 'F'])
    gender_to_idx = {'M': 0, 'F': 1}
    model_data['gender_idx'] = model_data['gender'].map(gender_to_idx)
    n_courses_m5 = len(unique_courses_m5)
    n_races_m5 = len(unique_races_m5)
    n_genders = len(unique_genders)
    n_results_m5 = len(model_data)
    model_data_finishers_m5 = model_data[model_data['finished']].copy()
    model_data_finishers_m5['pace_min_mi'] = model_data_finishers_m5['time_ms'] / 60000 / model_data_finishers_m5['distance_miles']
    model_data_finishers_m5['log_pace'] = np.log(model_data_finishers_m5['pace_min_mi'])
    model_data_finishers_m5['log_dist_ratio'] = np.log(model_data_finishers_m5['distance_miles'] / reference_distance_1)
    model_data_finishers_m5['course_idx'] = model_data_finishers_m5['course_id'].map(course_id_to_idx_m5)
    model_data_finishers_m5['race_idx'] = model_data_finishers_m5['event_distance_id'].map(race_id_to_idx_m5)
    model_data_finishers_m5['gender_idx'] = model_data_finishers_m5['gender'].map(gender_to_idx)
    n_finishers_m5 = len(model_data_finishers_m5)
    course_indices_results_m5 = model_data['course_idx'].values
    race_indices_results_m5 = model_data['race_idx'].values
    gender_indices_results_m5 = model_data['gender_idx'].values
    distances_results_m5 = model_data['distance_miles'].values
    log_dist_ratios_results_m5 = np.log(distances_results_m5 / reference_distance_1)
    finished_results_m5 = model_data['finished'].values.astype(int)
    course_indices_finishers_m5 = model_data_finishers_m5['course_idx'].values
    race_indices_finishers_m5 = model_data_finishers_m5['race_idx'].values
    gender_indices_finishers_m5 = model_data_finishers_m5['gender_idx'].values
    distances_finishers_m5 = model_data_finishers_m5['distance_miles'].values
    log_dist_ratios_finishers_m5 = model_data_finishers_m5['log_dist_ratio'].values
    log_pace_finishers_m5 = model_data_finishers_m5['log_pace'].values
    print(f'=== DATA DIMENSIONS ===')
    print(f'Total results (DNF + finishers): {n_results_m5:,}')
    print(f'  Finishers: {n_finishers_m5:,} ({100 * n_finishers_m5 / n_results_m5:.1f}%)')
    print(f'  DNFs: {n_results_m5 - n_finishers_m5:,} ({100 * (n_results_m5 - n_finishers_m5) / n_results_m5:.1f}%)')
    print(f'\nCourses: {n_courses_m5:,}')
    print(f'Races (course-year combinations): {n_races_m5:,}')
    print(f'Genders: {n_genders}')
    print(f"Participants: {model_data['participant_id'].nunique():,}")
    print(f'\n=== HIERARCHICAL STRUCTURE ===')
    print(f'Race-to-course mapping: {len(race_to_course_m5):,} races mapped to {n_courses_m5:,} courses')
    print(f'Average races per course: {n_races_m5 / n_courses_m5:.1f}')
    assert model_data['course_idx'].isna().sum() == 0, 'Missing course indices'
    assert model_data['race_idx'].isna().sum() == 0, 'Missing race indices'
    assert model_data['gender_idx'].isna().sum() == 0, 'Missing gender indices'
    print(f'\n✓ Index mapping validated: No missing values')
    for _race_idx, _course_idx in enumerate(race_to_course_m5):
        _race_id = unique_races_m5[_race_idx]
        _actual_course_id = model_data[model_data['event_distance_id'] == _race_id]['course_id'].iloc[0]
        actual_course_idx = course_id_to_idx_m5[_actual_course_id]
        assert _course_idx == actual_course_idx, f'Race-to-course mapping error for race {_race_id}'
    print(f'✓ Race-to-course mapping validated')
    course_summary_m5 = model_data.groupby('course_id').agg({'event_distance_id': 'nunique', 'participant_id': 'nunique', 'finished': ['sum', 'count']}).reset_index()
    course_summary_m5.columns = ['course_id', 'n_races', 'n_participants', 'n_finishers', 'n_results']
    course_summary_m5['finish_rate'] = course_summary_m5['n_finishers'] / course_summary_m5['n_results']
    print(f'\n=== COURSE SUMMARY STATISTICS ===')
    print(f"Median races per course: {course_summary_m5['n_races'].median():.0f}")
    print(f"Median participants per course: {course_summary_m5['n_participants'].median():.0f}")
    print(f"Median results per course: {course_summary_m5['n_results'].median():.0f}")
    print(f"Median finish rate: {course_summary_m5['finish_rate'].median():.1%}")
    print(f'\n=== TOP 10 COURSES BY RESULT COUNT ===')
    top_courses = course_summary_m5.nlargest(10, 'n_results')[['course_id', 'n_races', 'n_participants', 'n_results', 'finish_rate']]
    for _idx, _row in top_courses.iterrows():
        _course_name = _row['course_id'].split('||')[0][:40]
        print(f"{_course_name:40s} | Races: {_row['n_races']:3.0f} | Participants: {_row['n_participants']:5.0f} | Results: {_row['n_results']:6.0f} | Finish: {_row['finish_rate']:5.1%}")
    return (
        course_indices_finishers_m5,
        course_indices_results_m5,
        distances_finishers_m5,
        distances_results_m5,
        finished_results_m5,
        gender_indices_finishers_m5,
        log_dist_ratios_finishers_m5,
        log_dist_ratios_results_m5,
        model_data_finishers_m5,
        n_courses_m5,
        n_finishers_m5,
        n_races_m5,
        n_results_m5,
        race_indices_finishers_m5,
        race_indices_results_m5,
        race_to_course_m5,
        unique_courses_m5,
        unique_genders,
        unique_races_m5,
    )


@app.cell
def _(finished_results_m5, model_data_finishers_m5, n_courses_m5, n_races_m5):
    # Create additional arrays and coordinates needed for Model 5

    # Create coordinate labels for interpretability
    course_coords_m5 = [f"Course_{i}" for i in range(n_courses_m5)]
    race_coords_m5 = [f"Race_{i}" for i in range(n_races_m5)]

    # Create did_finish_m5 from finished_results_m5 (fix naming inconsistency)
    did_finish_m5 = finished_results_m5

    # Create observed_times_finishers_m5 from time data
    observed_times_finishers_m5 = model_data_finishers_m5['time_ms'].values / 60000  # Convert to minutes

    print(f"=== ADDITIONAL MODEL VARIABLES CREATED ===")
    print(f"Course coordinates: {len(course_coords_m5):,}")
    print(f"Race coordinates: {len(race_coords_m5):,}")
    print(f"Finish/DNF indicator: {len(did_finish_m5):,} results")
    print(f"Observed times (finishers only): {len(observed_times_finishers_m5):,} times")
    print(f"  Range: {observed_times_finishers_m5.min():.1f} - {observed_times_finishers_m5.max():.1f} minutes")
    return (
        course_coords_m5,
        did_finish_m5,
        observed_times_finishers_m5,
        race_coords_m5,
    )


@app.cell
def _(model_data, n_courses_m5, unique_courses_m5):
    # ============================================================================
    # IDENTIFY NON-REPORTING COURSES EMPIRICALLY (FULLY VECTORIZED)
    # ============================================================================

    # Compute DNF count per course (vectorized aggregation)
    course_dnf_counts_m5 = model_data.groupby('course_id', observed=False)['finished'].agg(['sum', 'count']).reset_index()
    course_dnf_counts_m5.columns = ['course_id', 'n_finished', 'n_total']
    course_dnf_counts_m5['n_dnf'] = course_dnf_counts_m5['n_total'] - course_dnf_counts_m5['n_finished']

    # FULLY VECTORIZED: Use pandas Series with reindex (preserves order of unique_courses_m5)
    course_dnf_series = course_dnf_counts_m5.set_index('course_id')['n_dnf']
    observed_dnf_count_per_course = course_dnf_series.reindex(unique_courses_m5).values

    # Statistics
    n_courses_with_dnf = (observed_dnf_count_per_course > 0).sum()
    n_courses_zero_dnf = (observed_dnf_count_per_course == 0).sum()

    print("=" * 80)
    print("NON-REPORTING COURSE IDENTIFICATION")
    print("=" * 80)
    print(f"Total courses in model: {n_courses_m5:,}")
    print(f"  Courses with ANY DNFs: {n_courses_with_dnf:,} ({100*n_courses_with_dnf/n_courses_m5:.1f}%)")
    print(f"  Courses with ZERO DNFs: {n_courses_zero_dnf:,} ({100*n_courses_zero_dnf/n_courses_m5:.1f}%)")
    print(f"\n✓ Distance-dependent reporting probability will handle non-reporters")
    print(f"✓ All {n_courses_m5:,} courses kept in dataset")
    return (
        n_courses_with_dnf,
        n_courses_zero_dnf,
        observed_dnf_count_per_course,
    )


@app.cell
def _(
    course_coords_m5,
    course_indices_finishers_m5,
    course_indices_results_m5,
    did_finish_m5,
    distances_finishers_m5,
    empirical_priors_m5,
    gender_indices_finishers_m5,
    log_dist_ratios_finishers_m5,
    log_dist_ratios_results_m5,
    model_data,
    n_courses_m5,
    n_courses_with_dnf,
    n_courses_zero_dnf,
    n_finishers_m5,
    n_races_m5,
    n_results_m5,
    np,
    observed_dnf_count_per_course,
    observed_times_finishers_m5,
    pm,
    prior_beta_dist_dnf,
    prior_gamma,
    prior_logit_psi,
    prior_mu_logit_dnf,
    prior_sigma_course,
    prior_sigma_race,
    race_coords_m5,
    race_indices_finishers_m5,
    race_indices_results_m5,
    race_to_course_m5,
    reference_distance_1,
    unique_courses_m5,
    unique_genders,
):
    course_distances_m5 = np.array([model_data[model_data['course_id'] == cid]['distance_miles'].iloc[0] for cid in unique_courses_m5])
    n_results_per_course = np.array([len(model_data[model_data['course_id'] == cid]) for cid in unique_courses_m5])
    reporting_results_mask_m5 = observed_dnf_count_per_course[course_indices_results_m5] > 0
    _reporting_results_idx = np.flatnonzero(reporting_results_mask_m5)
    coords_m5 = {'course': course_coords_m5, 'race': race_coords_m5, 'gender': unique_genders, 'finishers': range(n_finishers_m5), 'results': range(n_results_m5), 'reporting_results': range(len(_reporting_results_idx))}
    with pm.Model(coords=coords_m5) as model_m5:
        mu_pace = pm.Normal('mu_pace', mu=np.array([empirical_priors_m5[g]['log_marathon_pace'] for g in unique_genders]), sigma=np.array([empirical_priors_m5[g]['log_pace_sd'] for g in unique_genders]), dims='gender')
        beta_1 = pm.Normal('beta', mu=np.array([empirical_priors_m5[g]['beta'] for g in unique_genders]), sigma=0.05, dims='gender')
        sigma_course = pm.HalfNormal('sigma_course', sigma=prior_sigma_course)
        sigma_race = pm.HalfNormal('sigma_race', sigma=prior_sigma_race)
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=np.array([empirical_priors_m5[g]['log_pace_sd'] for g in unique_genders]), dims='gender')
        course_baseline_raw = pm.Normal('course_baseline_raw', mu=0, sigma=1, dims='course')
        course_baseline = pm.Deterministic('course_baseline', sigma_course * course_baseline_raw, dims='course')
        race_difficulty_raw = pm.Normal('race_difficulty_raw', mu=0, sigma=1, dims='race')
        race_difficulty = pm.Deterministic('race_difficulty', course_baseline[race_to_course_m5] + sigma_race * race_difficulty_raw, dims='race')
        log_distance_ratio_finishers = log_dist_ratios_finishers_m5
        log_expected_pace = mu_pace[gender_indices_finishers_m5] + beta_1[gender_indices_finishers_m5] * log_distance_ratio_finishers + race_difficulty[race_indices_finishers_m5]
        observed_pace_finishers = observed_times_finishers_m5 / distances_finishers_m5
        pace_likelihood = pm.LogNormal('pace_likelihood', mu=log_expected_pace, sigma=sigma_obs[gender_indices_finishers_m5], observed=observed_pace_finishers, dims='finishers')
        mu_logit_dnf = pm.Normal('mu_logit_dnf', mu=prior_mu_logit_dnf, sigma=0.5)
        beta_dist_dnf = pm.Normal('beta_dist_dnf', mu=prior_beta_dist_dnf, sigma=0.4)
        beta_dist_dnf_quad = pm.Normal('beta_dist_dnf_quad', mu=0.5, sigma=0.3)
        gamma_1 = pm.TruncatedNormal('gamma', mu=prior_gamma, sigma=0.5, lower=0)
        logit_psi = pm.Normal('logit_psi', mu=prior_logit_psi, sigma=0.5)
        _psi = pm.Deterministic('psi', pm.math.sigmoid(logit_psi))
        log_dist_ratio_per_course = np.log(course_distances_m5 / reference_distance_1)
        logit_expected_dnf_per_course = mu_logit_dnf + beta_dist_dnf * log_dist_ratio_per_course + beta_dist_dnf_quad * log_dist_ratio_per_course ** 2
        expected_dnf_prob_per_course = pm.math.sigmoid(logit_expected_dnf_per_course)
        reporting_prob_course = pm.Deterministic('reporting_prob_course', pm.math.switch(observed_dnf_count_per_course > 0, 1.0, _psi), dims='course')
        zero_dnf_mask = observed_dnf_count_per_course == 0
        prob_zero_dnf_if_reporting = pm.math.switch(zero_dnf_mask, (1 - expected_dnf_prob_per_course) ** n_results_per_course, 1.0)
        pm.Potential('zero_dnf_likelihood', pm.math.log(reporting_prob_course * prob_zero_dnf_if_reporting + (1 - reporting_prob_course) * 1.0).sum())
        reporting_prob_results = reporting_prob_course[course_indices_results_m5]
        logit_true_dnf = mu_logit_dnf + beta_dist_dnf * log_dist_ratios_results_m5 + beta_dist_dnf_quad * log_dist_ratios_results_m5 ** 2 + gamma_1 * race_difficulty[race_indices_results_m5]
        true_dnf_prob = pm.math.sigmoid(logit_true_dnf)
        observed_dnf_prob = pm.Deterministic('observed_dnf_prob', reporting_prob_results * true_dnf_prob, dims='results')
        # Zero-DNF courses are covered ONLY by the marginal mixture Potential above;
        # including their results in the per-result Bernoulli too would count the same
        # evidence twice (and treat the shared reporting indicator as independent
        # across results). Courses with >=1 observed DNF must be reporters, so their
        # results get the Bernoulli with true_dnf_prob directly.
        finish_prob_reporting = 1 - true_dnf_prob[_reporting_results_idx]
        finish_likelihood = pm.Bernoulli('finish_likelihood', p=finish_prob_reporting, observed=did_finish_m5[_reporting_results_idx], dims='reporting_results')
        logit_dnf_expected = mu_logit_dnf + beta_dist_dnf * log_dist_ratios_finishers_m5 + beta_dist_dnf_quad * log_dist_ratios_finishers_m5 ** 2
        expected_dnf_prob_finishers = pm.math.sigmoid(logit_dnf_expected)
        expected_finish_prob_finishers = 1 - expected_dnf_prob_finishers
        reporting_prob_finishers = reporting_prob_course[course_indices_finishers_m5]
        selection_prob = reporting_prob_finishers * expected_finish_prob_finishers + (1 - reporting_prob_finishers)
        selection_weight = 1 / selection_prob
        pm.Potential('selection_correction', pm.math.log(selection_weight))
    print(f'✅ Model 5c defined with deterministic reporting probability + Bernoulli likelihood')
    print(f'   Courses with DNFs: {n_courses_with_dnf:,} → reporting_prob = 1.0, results in Bernoulli ({len(_reporting_results_idx):,} results)')
    print(f'   Courses with zero DNFs: {n_courses_zero_dnf:,} → marginal mixture Potential only (no per-result Bernoulli)')
    print(f'   Distance-specific DNF rates used for zero-DNF course likelihood')
    print(f'   Total parameters: {10} hyperparameters + {n_courses_m5:,} courses + {n_races_m5:,} races')
    return model_m5, reporting_results_mask_m5, sigma_obs


@app.cell
def _(distances_finishers_m5, distances_results_m5, np, reference_distance_1):
    # Pre-compute logarithmic distance ratios (constant transformations)
    # These are used multiple times in the model but depend only on data, not parameters
    # Pre-computing reduces redundant PyMC operations during MCMC sampling
    log_dist_ratios_finishers_m5_1 = np.log(distances_finishers_m5 / reference_distance_1)
    # For finishers (used in pace model and selection correction)
    log_dist_ratios_results_m5_1 = np.log(distances_results_m5 / reference_distance_1)
    print(f'✅ Pre-computed distance transformations:')
    # For all results (used in DNF model)
    print(f'   Finishers: {len(log_dist_ratios_finishers_m5_1):,} log-distance ratios')
    print(f'   Results:   {len(log_dist_ratios_results_m5_1):,} log-distance ratios')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Diagram

    Visualize the PyMC model structure showing dependencies between parameters and observed data.
    """)
    return


@app.cell
def _(model_m5, pm):
    # Generate model diagram
    graph_m5 = pm.model_to_graphviz(model_m5)
    graph_m5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior Predictive Check
    """)
    return


@app.cell
def _(model_m5, pm):
    with model_m5:
        prior_pred_m5 = pm.sample_prior_predictive(samples=1000, random_seed=42)

    print(f"Available variables: {list(prior_pred_m5.prior_predictive.data_vars)}")
    return (prior_pred_m5,)


@app.cell
def _(
    distances_finishers_m5,
    gender_indices_finishers_m5,
    np,
    plt,
    prior_pred_m5,
    results_finishers_1,
    unique_genders,
):
    prior_pace_samples_m5 = prior_pred_m5.prior_predictive['pace_likelihood'].values
    prior_pace_flat_m5 = prior_pace_samples_m5.reshape(-1, prior_pace_samples_m5.shape[-1])
    standard_distances = [('10K', 6.21371, 0.3), ('10mi', 10.0, 0.3), ('Half', 13.1, 0.3), ('Marathon', 26.2, 0.5), ('50K', 31.0686, 0.5), ('50mi', 50.0, 1.0), ('100K', 62.1371, 1.0), ('100mi', 100.0, 1.0)]
    for _gender_idx, _gender in enumerate(unique_genders):
        _fig, _axes = plt.subplots(2, 4, figsize=(16, 8))
        _axes = _axes.flatten()
        gender_mask_model = gender_indices_finishers_m5 == _gender_idx
        gender_mask_full = results_finishers_1['gender'] == _gender
        for _col_idx, (_name, _dist_value, _tolerance) in enumerate(standard_distances):
            _ax = _axes[_col_idx]
            _obs_mask_full = np.abs(results_finishers_1['distance_miles'] - _dist_value) < _tolerance
            combined_mask_full = _obs_mask_full & gender_mask_full
            _obs_mask_model = np.abs(distances_finishers_m5 - _dist_value) < _tolerance
            combined_mask_model = _obs_mask_model & gender_mask_model
            if not combined_mask_full.any():
                _ax.axis('off')
                _ax.set_title(f'{_name}\nNo data', fontsize=10)
                continue
            _obs_times_binned = results_finishers_1.loc[combined_mask_full, 'time_ms'].values / 60000
            prior_pace_binned = prior_pace_flat_m5[:, combined_mask_model]
            _distances_binned = distances_finishers_m5[combined_mask_model]
            prior_times_binned = prior_pace_binned * _distances_binned[np.newaxis, :]
            prior_times_flat = prior_times_binned.flatten()
            _obs_min, _obs_max = (_obs_times_binned.min(), _obs_times_binned.max())
            _x_range = _obs_max - _obs_min
            _bin_edges = np.linspace(max(0, _obs_min - 0.1 * _x_range), _obs_max + 0.3 * _x_range, 31)
            _ax.hist(_obs_times_binned, bins=_bin_edges, alpha=0.6, label='Observed', density=True, color='steelblue')
            _ax.hist(prior_times_flat, bins=_bin_edges, alpha=0.6, label='Prior', density=True, color='orange')
            _ax.set_xlim(_bin_edges[0], _bin_edges[-1])
            _ax.set_xlabel('Time (min)', fontsize=10)
            _ax.set_ylabel('Density', fontsize=10)
            _ax.set_title(f'{_name} ({_dist_value:.1f}mi)\n{len(_obs_times_binned):,} finishers', fontsize=10)
            _ax.legend(fontsize=9)
            _ax.tick_params(labelsize=9)
        _gender_label = 'Male' if _gender == 'M' else 'Female'
        _fig.suptitle(f'Prior Predictive Check: Finish Times ({_gender_label})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    return (standard_distances,)


@app.cell
def _(
    course_indices_results_m5,
    distances_results_m5,
    np,
    plt,
    prior_pred_m5,
    race_indices_results_m5,
    reporting_results_mask_m5,
    results_1,
    standard_distances,
    unique_courses_m5,
    unique_races_m5,
):
    prior_finish_samples_m5 = prior_pred_m5.prior_predictive['finish_likelihood'].values
    prior_dnf_samples_m5 = 1 - prior_finish_samples_m5
    prior_dnf_flat_m5 = prior_dnf_samples_m5.reshape(-1, prior_dnf_samples_m5.shape[-1])
    # finish_likelihood is defined over reporting-course results only; restrict the
    # model-side index arrays to the same subset so columns line up.
    _distances_results_rep = distances_results_m5[reporting_results_mask_m5]
    _race_indices_results_rep = race_indices_results_m5[reporting_results_mask_m5]
    _course_indices_results_rep = course_indices_results_m5[reporting_results_mask_m5]
    _fig, _axes = plt.subplots(2, 4, figsize=(16, 8))
    _axes = _axes.flatten()
    for _col_idx, (_name, _dist_value, _tolerance) in enumerate(standard_distances):
        _ax = _axes[_col_idx]
        _obs_mask_full = np.abs(results_1['distance_miles'] - _dist_value) < _tolerance
        _obs_mask_model = np.abs(_distances_results_rep - _dist_value) < _tolerance
        if not _obs_mask_full.any():
            _ax.axis('off')
            _ax.set_title(f'{_name}\nNo data', fontsize=10)
            continue
        _race_results_full = results_1[_obs_mask_full]
        race_dnf_counts = _race_results_full.groupby('event_distance_id')['finished'].apply(lambda x: (~x).sum())
        reporting_races = race_dnf_counts[race_dnf_counts > 0].index
        _race_results_full = _race_results_full[_race_results_full['event_distance_id'].isin(reporting_races)]
        _unique_races_full = _race_results_full['event_distance_id'].unique()
        _obs_dnf_rates = []
        for _race_id in _unique_races_full:
            _race_data = _race_results_full[_race_results_full['event_distance_id'] == _race_id]
            _dnf_rate = (~_race_data['finished']).mean()
            _obs_dnf_rates.append(_dnf_rate)
        _obs_dnf_rates = np.array(_obs_dnf_rates)
        _race_indices_dist = _race_indices_results_rep[_obs_mask_model]
        _unique_races_dist = np.unique(_race_indices_dist)
        prior_dnf_rates = []
        for _race_idx in _unique_races_dist:
            if unique_races_m5[_race_idx] not in reporting_races:
                continue
            _race_mask_dist = _race_indices_dist == _race_idx
            _race_results_indices = np.where(_obs_mask_model)[0][_race_mask_dist]
            prior_dnf_race = prior_dnf_flat_m5[:, _race_results_indices]
            for _sample_idx in range(prior_dnf_race.shape[0]):
                _n_dnfs = prior_dnf_race[_sample_idx, :].sum()
                if _n_dnfs >= 1:
                    prior_dnf_rate_sample = prior_dnf_race[_sample_idx, :].mean()
                    prior_dnf_rates.append(prior_dnf_rate_sample)
        prior_dnf_rates = np.array(prior_dnf_rates)
        _bin_edges = np.linspace(0, 0.6, 31)
        _ax.hist(_obs_dnf_rates, bins=_bin_edges, alpha=0.6, label='Observed', density=True, color='steelblue')
        _ax.hist(prior_dnf_rates, bins=_bin_edges, alpha=0.6, label='Prior', density=True, color='orange')
        _ax.set_xlim(0, 0.6)
        _ax.set_xlabel('DNF Rate (by race)', fontsize=10)
        _ax.set_ylabel('Density', fontsize=10)
        _ax.set_title(f'{_name} ({_dist_value:.1f}mi)\n{len(_unique_races_full)} reporting races', fontsize=10)
        _ax.legend(fontsize=9)
        _ax.tick_params(labelsize=9)
    _fig.suptitle(f'Prior Predictive Check: DNF Rates by Race (Reporting Only)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    _fig, _axes = plt.subplots(2, 4, figsize=(16, 8))
    _axes = _axes.flatten()
    for _col_idx, (_name, _dist_value, _tolerance) in enumerate(standard_distances):
        _ax = _axes[_col_idx]
        _obs_mask_full = np.abs(results_1['distance_miles'] - _dist_value) < _tolerance
        _obs_mask_model = np.abs(_distances_results_rep - _dist_value) < _tolerance
        if not _obs_mask_full.any():
            _ax.axis('off')
            _ax.set_title(f'{_name}\nNo data', fontsize=10)
            continue
        course_results_full = results_1[_obs_mask_full]
        course_dnf_counts = course_results_full.groupby('course_id')['finished'].apply(lambda x: (~x).sum())
        reporting_courses = course_dnf_counts[course_dnf_counts > 0].index
        course_results_full = course_results_full[course_results_full['course_id'].isin(reporting_courses)]
        unique_courses_full = course_results_full['course_id'].unique()
        obs_dnf_rates_course = []
        for _course_id in unique_courses_full:
            _course_data = course_results_full[course_results_full['course_id'] == _course_id]
            _dnf_rate = (~_course_data['finished']).mean()
            obs_dnf_rates_course.append(_dnf_rate)
        obs_dnf_rates_course = np.array(obs_dnf_rates_course)
        course_indices_dist = _course_indices_results_rep[_obs_mask_model]
        unique_courses_dist = np.unique(course_indices_dist)
        prior_dnf_rates_course = []
        for _course_idx in unique_courses_dist:
            _actual_course_id = unique_courses_m5[_course_idx]
            if _actual_course_id not in reporting_courses:
                continue
            course_mask_dist = course_indices_dist == _course_idx
            course_results_indices = np.where(_obs_mask_model)[0][course_mask_dist]
            prior_dnf_course = prior_dnf_flat_m5[:, course_results_indices]
            for _sample_idx in range(prior_dnf_course.shape[0]):
                _n_dnfs = prior_dnf_course[_sample_idx, :].sum()
                if _n_dnfs >= 1:
                    prior_dnf_rate_sample = prior_dnf_course[_sample_idx, :].mean()
                    prior_dnf_rates_course.append(prior_dnf_rate_sample)
        prior_dnf_rates_course = np.array(prior_dnf_rates_course)
        _bin_edges = np.linspace(0, 0.6, 31)
        _ax.hist(obs_dnf_rates_course, bins=_bin_edges, alpha=0.6, label='Observed', density=True, color='steelblue')
        _ax.hist(prior_dnf_rates_course, bins=_bin_edges, alpha=0.6, label='Prior', density=True, color='orange')
        _ax.set_xlim(0, 0.6)
        _ax.set_xlabel('DNF Rate (by course)', fontsize=10)
        _ax.set_ylabel('Density', fontsize=10)
        _ax.set_title(f'{_name} ({_dist_value:.1f}mi)\n{len(unique_courses_full)} reporting courses', fontsize=10)
        _ax.legend(fontsize=9)
        _ax.tick_params(labelsize=9)
    _fig.suptitle(f'Prior Predictive Check: DNF Rates by Course (Reporting Only)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np, plt, prior_pred_m5, results_1, standard_distances):
    _fig, _axes = plt.subplots(2, 4, figsize=(16, 10))
    _axes = _axes.flatten()
    psi_samples = prior_pred_m5.prior['psi'].values.flatten()
    course_stats_1 = results_1.groupby('course_id').agg({'finished': [lambda x: (~x).sum() > 0, 'count'], 'distance_miles': 'first'}).reset_index()
    course_stats_1.columns = ['course_id', 'has_dnf', 'n_results', 'distance_miles']
    print('=' * 80)
    print('PRIOR PREDICTIVE CHECK: REPORTING PROBABILITY')
    print('=' * 80)
    print(f'Baseline psi prior: {psi_samples.mean():.3f} ± {psi_samples.std():.3f}')
    print(f"Observed overall reporting: {course_stats_1['has_dnf'].mean():.3f}")
    print(f'\nGenerating simulated reporting patterns from prior...')
    print('=' * 80 + '\n')
    for _col_idx, (_name, _dist_value, _tolerance) in enumerate(standard_distances):
        if _col_idx >= len(_axes):
            break
        _ax = _axes[_col_idx]
        _mask = (course_stats_1['distance_miles'] >= _dist_value - _tolerance) & (course_stats_1['distance_miles'] <= _dist_value + _tolerance)
        courses_at_dist = course_stats_1[_mask]
        if len(courses_at_dist) == 0:
            _ax.axis('off')
            _ax.set_title(f'{_name}\nNo data', fontsize=10)
            continue
        obs_reporting_rate = courses_at_dist['has_dnf'].mean()
        n_courses = len(courses_at_dist)
        typical_dnf_rate = 0.1 if _dist_value < 50 else 0.15 if _dist_value < 100 else 0.25
        n_sim = min(1000, len(psi_samples))
        psi_sim_samples = np.random.choice(psi_samples, size=n_sim, replace=False)
        simulated_reporting_rates = []
        for _psi in psi_sim_samples:
            course_sizes = courses_at_dist['n_results'].values
            n_courses_sim = len(course_sizes)
            truly_reports = np.random.binomial(1, _psi, size=n_courses_sim)
            _n_dnfs = np.random.binomial(course_sizes, typical_dnf_rate)
            observed_dnf_count = truly_reports * _n_dnfs
            has_observed_dnf = (observed_dnf_count > 0).astype(int)
            simulated_reporting_rates.append(has_observed_dnf.mean())
        simulated_reporting_rates = np.array(simulated_reporting_rates)
        _bin_edges = np.linspace(0, 1, 21)
        _ax.hist(simulated_reporting_rates, bins=_bin_edges, alpha=0.6, density=True, color='orange', edgecolor='black', linewidth=0.5, label='Prior predictive')
        _ax.axvline(obs_reporting_rate, color='steelblue', linewidth=3, label=f'Observed: {obs_reporting_rate:.1%}', linestyle='--')
        prior_pred_mean = simulated_reporting_rates.mean()
        _ax.axvline(prior_pred_mean, color='orange', linewidth=2, alpha=0.7, label=f'Prior mean: {prior_pred_mean:.1%}', linestyle=':')
        _ax.set_xlim(0, 1)
        _ax.set_xlabel('Fraction Reporting', fontsize=10)
        _ax.set_ylabel('Density', fontsize=10)
        _ax.set_title(f'{_name}\n{n_courses} courses', fontsize=10)
        _ax.legend(fontsize=7, loc='upper left')
        _ax.tick_params(labelsize=9)
        p_lower = (simulated_reporting_rates < obs_reporting_rate).mean()
        p_upper = (simulated_reporting_rates > obs_reporting_rate).mean()
        _p_value = 2 * min(p_lower, p_upper)
        _median_size = int(courses_at_dist['n_results'].median())
        if _p_value < 0.05:
            _status = '⚠️ Mismatch'
            bgcolor = 'lightcoral'
        else:
            _status = '✓ Reasonable'
            bgcolor = 'lightgreen'
        _textstr = f'p={_p_value:.3f}\n{_status}\n\nn_med={_median_size}\np_dnf={typical_dnf_rate:.0%}'
        _ax.text(0.97, 0.97, _textstr, transform=_ax.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor=bgcolor, alpha=0.5))
    for _col_idx in range(len(standard_distances), len(_axes)):
        _axes[_col_idx].axis('off')
    _fig.suptitle('Prior Predictive Check: Reporting Probability by Distance\nSimulates: truly_reports ~ Bernoulli(psi), observed = reports × (n_dnf > 0)\nOrange = simulated, Blue = observed, Green box = pass, Red box = fail', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print('\n' + '=' * 80)
    print('INTERPRETATION')
    print('=' * 80)
    print('✓ p ≥ 0.05: Prior generative model matches observed data')
    print('⚠️ p < 0.05: Prior inconsistent with observations')
    print('')
    print('Generative process (matching the actual model):')
    print('  1. truly_reports ~ Bernoulli(psi) - organizational characteristic')
    print('  2. n_dnf ~ Binomial(n, p_dnf) - actual DNFs that occur')
    print('  3. observed_dnf = truly_reports × n_dnf - zero-inflated')
    print('  4. has_observed_dnf = (observed_dnf > 0) - what we measure')
    print('')
    print("This EXACTLY matches your model's pm.math.switch logic!")
    print('=' * 80)
    return course_stats_1, psi_samples


@app.cell
def _(course_stats_1, psi_samples, standard_distances):
    print('=' * 80)
    print('INVESTIGATING PRIOR/OBSERVED MISMATCH')
    print('=' * 80)
    empirical_reporting_full = course_stats_1['has_dnf'].mean()
    print(f'\n1. EMPIRICAL REPORTING RATE (all courses):')
    print(f"   {empirical_reporting_full:.1%} ({course_stats_1['has_dnf'].sum():,} / {len(course_stats_1):,} courses)")
    print(f"\n2. PRIOR PSI (what we're simulating with):")
    print(f'   Mean: {psi_samples.mean():.1%}')
    print(f'   Std:  {psi_samples.std():.3f}')
    print(f'\n3. DISCREPANCY:')
    print(f'   Prior psi - Empirical = {psi_samples.mean() - empirical_reporting_full:.1%}')
    print(f'   → Prior is TOO HIGH!')
    print(f'\n4. WHERE DID PSI=48% COME FROM?')
    print(f'   From empirical_priors.py: calculate_reporting_priors()')
    print(f"   This should match empirical rate IF we're using full data")
    print(f"   But it's {psi_samples.mean():.1%} instead of {empirical_reporting_full:.1%}")
    print(f'\n5. HYPOTHESIS: Prior calculated on REPORTING COURSES ONLY?')
    print(f'   If we only use courses WITH DNFs to estimate psi...')
    print(f'   That would give psi=1.0 (circular!)')
    print(f'   ')
    print(f'   Let me check typical DNF rates to see if detection power explains it:')
    for dist_idx, (_name, _dist_value, _tolerance) in enumerate(standard_distances[:4]):
        _mask = (course_stats_1['distance_miles'] >= _dist_value - _tolerance) & (course_stats_1['distance_miles'] <= _dist_value + _tolerance)
        courses = course_stats_1[_mask]
        if len(courses) == 0:
            continue
        _median_size = int(courses['n_results'].median())
        typical_dnf = 0.1 if _dist_value < 50 else 0.15
        p_any_dnf = 1 - (1 - typical_dnf) ** _median_size
        expected_obs = empirical_reporting_full * p_any_dnf
        actual_obs = courses['has_dnf'].mean()
        print(f'   {_name:12s}: Expected {expected_obs:.1%}, Observed {actual_obs:.1%}, Diff {actual_obs - expected_obs:+.1%}')
    print(f'\n6. CONCLUSION:')
    print(f'   If prior psi is correctly set to empirical rate (~{empirical_reporting_full:.1%}),')
    print(f'   then prior predictive should match observed much better!')
    print(f'   ')
    print(f'   ISSUE: Prior psi = {psi_samples.mean():.1%} is too high!')
    print(f'   FIX: Check empirical_priors.py calculate_reporting_priors()')
    print('=' * 80)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Geometry Check
    """)
    return


@app.cell
def _(
    az,
    linregress,
    model_m5,
    n_courses_m5,
    n_races_m5,
    np,
    pearsonr,
    plt,
    pm,
    sigma_obs,
    warnings,
):
    warnings.filterwarnings('ignore', message='.*Potentials.*prior predictive.*')
    with model_m5:
        prior_samples_m5 = pm.sample_prior_predictive(samples=500, random_seed=42)
    _fig = az.plot_pair(prior_samples_m5, group='prior', var_names=['mu_pace', 'beta', 'sigma_obs', 'sigma_course', 'sigma_race', 'mu_logit_dnf', 'beta_dist_dnf', 'beta_dist_dnf_quad', 'gamma', 'logit_psi'], kind='scatter', marginals=True, figsize=(20, 20), scatter_kwargs={'alpha': 0.4, 's': 10}, point_estimate='mean')
    plt.suptitle('Prior Pair Plot: All Hyperparameters - Model 5b\n(Checking identifiability and funnel geometry)', fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.show()
    sigma_course_prior_m5_check = prior_samples_m5.prior['sigma_course'].values.flatten()
    course_baseline_raw_prior_m5_check = prior_samples_m5.prior['course_baseline_raw'].values
    course_baseline_var_m5 = course_baseline_raw_prior_m5_check.std(axis=-1).flatten()
    sigma_race_prior_m5_check = prior_samples_m5.prior['sigma_race'].values.flatten()
    race_difficulty_raw_prior_m5_check = prior_samples_m5.prior['race_difficulty_raw'].values
    race_difficulty_var_m5 = race_difficulty_raw_prior_m5_check.std(axis=-1).flatten()
    pearson_corr_course_m5, pearson_p_course_m5 = pearsonr(sigma_course_prior_m5_check, course_baseline_var_m5)
    pearson_corr_race_m5, pearson_p_race_m5 = pearsonr(sigma_race_prior_m5_check, race_difficulty_var_m5)
    print('=' * 70)
    print('HIERARCHICAL GEOMETRY CHECK - MODEL 5')
    print('=' * 70)
    print('\n📊 FUNNEL DIAGNOSTICS (Non-centered parameterization)')
    print('   Goal: Verify σ (hyperparameter) is independent of realized variation')
    print('   Expected: Pearson r ≈ 0 (horizontal scatter around Std = 1.0)\n')
    print('   Course Baseline (σ_course):')
    print(f'      Pearson r = {pearson_corr_course_m5:+.4f} (p = {pearson_p_course_m5:.2e})')
    if abs(pearson_corr_course_m5) < 0.1:
        print(f'      ✅ No funnel detected - excellent geometry')
    elif abs(pearson_corr_course_m5) < 0.3:
        print(f'      ⚠️  Minor funnel detected - acceptable but monitor')
    else:
        print(f'      ❌ Strong funnel detected - may cause sampling issues')
    print('\n   Race-Year Difficulty (σ_race):')
    print(f'      Pearson r = {pearson_corr_race_m5:+.4f} (p = {pearson_p_race_m5:.2e})')
    if abs(pearson_corr_race_m5) < 0.1:
        print(f'      ✅ No funnel detected - excellent geometry')
    elif abs(pearson_corr_race_m5) < 0.3:
        print(f'      ⚠️  Minor funnel detected - acceptable but monitor')
    else:
        print(f'      ❌ Strong funnel detected - may cause sampling issues')
    print('\n' + '-' * 70)
    print('VARIANCE HIERARCHY CHECK')
    print('-' * 70)
    sigma_obs_mean = sigma_obs.eval().mean()
    sigma_course_mean = prior_samples_m5.prior['sigma_course'].values.mean()
    sigma_race_mean = prior_samples_m5.prior['sigma_race'].values.mean()
    ratio_race_course = sigma_race_mean / sigma_course_mean
    print(f'\nObservation-level variation:      σ_obs     = {sigma_obs_mean:.3f}')
    print(f'Course-level variation:           σ_course  = {sigma_course_mean:.3f}')
    print(f'Race-year variation (within course): σ_race = {sigma_race_mean:.3f}')
    print(f'\nRatio (σ_race / σ_course):        {ratio_race_course:.3f}')
    if ratio_race_course < 0.5:
        print('   → Year-to-year variation is SMALL relative to course differences')
        print('   → Most difficulty is course-specific (persistent across years)')
    elif ratio_race_course > 1.5:
        print('   → Year-to-year variation is LARGE relative to course differences')
        print('   → Substantial race-specific effects (weather, conditions, etc.)')
    else:
        print('   → Year-to-year and course-level variation are COMPARABLE')
        print('   → Both persistent and transient difficulty factors matter')
    _fig, _axes = plt.subplots(1, 2, figsize=(16, 6))
    _ax = _axes[0]
    _ax.scatter(sigma_course_prior_m5_check, course_baseline_var_m5, alpha=0.3, s=10)
    _ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Expected Std = 1.0')
    slope, _intercept, _r_value, _p_value, _std_err = linregress(sigma_course_prior_m5_check, course_baseline_var_m5)
    _x_line = np.linspace(sigma_course_prior_m5_check.min(), sigma_course_prior_m5_check.max(), 100)
    _y_line = slope * _x_line + _intercept
    _ax.plot(_x_line, _y_line, 'b-', linewidth=2, label=f'Regression (r={pearson_corr_course_m5:.3f})')
    _ax.set_xlabel('σ_course (hyperparameter)', fontsize=11)
    _ax.set_ylabel('Std(course_baseline_raw) across samples', fontsize=11)
    _ax.set_title('Course Baseline Funnel Check', fontsize=12, fontweight='bold')
    _ax.legend(fontsize=10)
    _ax.grid(alpha=0.3)
    _textstr = f'Pearson r = {pearson_corr_course_m5:+.4f}\np = {pearson_p_course_m5:.2e}'
    _props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    _ax.text(0.05, 0.95, _textstr, transform=_ax.transAxes, fontsize=10, verticalalignment='top', bbox=_props)
    _ax = _axes[1]
    _ax.scatter(sigma_race_prior_m5_check, race_difficulty_var_m5, alpha=0.3, s=10)
    _ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Expected Std = 1.0')
    _slope_pct, _intercept_pct, _r_value, _p_value_reg, _std_err = linregress(sigma_race_prior_m5_check, race_difficulty_var_m5)
    _x_line = np.linspace(sigma_race_prior_m5_check.min(), sigma_race_prior_m5_check.max(), 100)
    _y_line = _slope_pct * _x_line + _intercept_pct
    _ax.plot(_x_line, _y_line, 'b-', linewidth=2, label=f'Regression (r={pearson_corr_race_m5:.3f})')
    _ax.set_xlabel('σ_race (hyperparameter)', fontsize=11)
    _ax.set_ylabel('Std(race_difficulty_raw) across samples', fontsize=11)
    _ax.set_title('Race Difficulty Funnel Check', fontsize=12, fontweight='bold')
    _ax.legend(fontsize=10)
    if abs(pearson_corr_race_m5) < 0.1:
        _status = '✅ EXCELLENT\nNo funnel'
        color = 'lightgreen'
    elif abs(pearson_corr_race_m5) < 0.3:
        _status = '⚠️ MODERATE\nMinor funnel'
        color = 'lightyellow'
    else:
        _status = '❌ WARNING\nStrong funnel'
        color = 'lightcoral'
    _ax.text(0.05, 0.95, _status, transform=_ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    plt.suptitle('Hierarchical Funnel Diagnostics - Model 5\n(Non-centered parameterization should show horizontal scatter around 1.0)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    print('\n' + '=' * 70)
    print('GEOMETRY CHECK SUMMARY')
    print('=' * 70)
    print(f'\n✅ Prior predictive sampling complete:')
    print(f'   • 500 prior samples drawn')
    print(f'   • {n_courses_m5:,} courses analyzed')
    print(f'   • {n_races_m5:,} races analyzed')
    print(f'   • {n_courses_m5 * 500:,} course-level data points visualized')
    print(f'   • {n_races_m5 * 500:,} race-level data points visualized')
    no_strong_funnels = abs(pearson_corr_course_m5) < 0.3 and abs(pearson_corr_race_m5) < 0.3
    if no_strong_funnels:
        print(f'\n✅ Model 5 is ready for MCMC sampling')
        print(f'   No strong funnel pathologies detected in hierarchical structure')
    else:
        print(f'\n⚠️  Review funnel diagnostics before proceeding to MCMC sampling')
        print(f'   Consider reparameterization if strong correlations persist')
    print('=' * 70)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MCMC Inference
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sampling

    Run NUTS MCMC with caching to disk. Draws high-quality posterior samples from k-core entities.
    """)
    return


@app.cell
def _(Path, alpha, az, beta, model_data, model_m5, os, pm):
    # Sampling parameters for Model 5. Dev iteration: small budget for fast feedback.
    tune_m5 = 500
    draws_m5 = 500
    target_accept_m5 = 0.95
    thin_m5 = 2  # Keep every 2nd sample → reduces memory by 50%
    # Anchor cache to the notebook file so it resolves regardless of marimo CWD.
    model_m5_dir = Path(__file__).resolve().parent.parent / 'data' / 'cache' / 'model_m5' / f'alpha{alpha}_beta{beta}'
    os.makedirs(model_m5_dir, exist_ok=True)
    cache_file_m5 = str(model_m5_dir / f'tune{tune_m5}_draws{draws_m5}_thin{thin_m5}_accept{target_accept_m5}.nc')
    n_participants = model_data['participant_id'].nunique()
    n_courses_m5_1 = model_data.groupby(['name', 'distance_miles']).ngroups
    n_races_m5_1 = model_data['event_distance_id'].nunique()
    if os.path.exists(cache_file_m5):
        print(f'✅ Loading cached trace from {cache_file_m5}')
        trace_m5 = az.from_netcdf(cache_file_m5)
        print(f"   Loaded: {trace_m5.posterior.dims['draw']} draws × {trace_m5.posterior.dims['chain']} chains")
    else:
        print(f'🚀 Running MCMC sampling for Model 5b...')
        print(f'  Configuration: tune={tune_m5}, draws={draws_m5}, thin={thin_m5}, target_accept={target_accept_m5}')
        print(f'  Data: participants={n_participants:,}, courses={n_courses_m5_1:,}, races={n_races_m5_1:,}')
        with model_m5:
            trace_m5 = pm.sample(
                draws=draws_m5,
                tune=tune_m5,
                chains=4,
                cores=4,
                target_accept=target_accept_m5,
                random_seed=42,
                return_inferencedata=True,
                idata_kwargs={'log_likelihood': False},
                nuts_sampler='nutpie',
            )
        print(f'💾 Saving trace to {cache_file_m5}')
        trace_m5.to_netcdf(cache_file_m5)
    return (
        draws_m5,
        model_m5_dir,
        n_courses_m5_1,
        n_participants,
        n_races_m5_1,
        target_accept_m5,
        thin_m5,
        trace_m5,
        tune_m5,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Traceplot

    Visual inspection of MCMC chains to detect mixing issues, convergence problems, or parameter correlations.
    """)
    return


@app.cell
def _(
    az,
    draws_m5,
    hyperparam_vars_m5,
    model_m5_dir,
    n_courses_m5_1,
    n_participants,
    n_races_m5_1,
    np,
    plt,
    target_accept_m5,
    thin_m5,
    trace_m5,
    tune_m5,
):
    # Traceplot for hyperparameters - Model 5b
    # Includes finish time parameters (with course + race decomposition) AND DNF parameters
    # KEY DIFFERENCE FROM MODEL 4: gamma (γ) couples difficulty across finish time and DNF models
    # MODEL 5B UPDATE: Simplified reporting model (single psi, no distance dependence) + quadratic DNF distance effect
    traceplot_file_m5 = f'{model_m5_dir}/tune{tune_m5}_draws{draws_m5}_thin{thin_m5}_accept{target_accept_m5}_traceplot.png'
    # Define traceplot cache file (matching naming pattern with .nc file)
    print(f'Generating traceplot...')
    az.plot_trace(trace_m5, var_names=hyperparam_vars_m5, compact=True, figsize=(12, 18))
    # Always generate traceplot (overwrite if exists)
    plt.suptitle(f'MCMC Traces: All Hyperparameters - Model 5b\n{n_participants:,} participants, {n_courses_m5_1:,} courses, {n_races_m5_1:,} races', fontsize=14, fontweight='bold')
    # Combined traceplot for all hyperparameters (finish time + DNF)
    plt.tight_layout()
    plt.savefig(traceplot_file_m5, dpi=300, bbox_inches='tight')
    print(f'Saved traceplot to {traceplot_file_m5}')
    plt.show()
    course_coords_m5_trace = list(trace_m5.posterior.coords['course'].values)
    sample_course_ids_m5_trace = np.random.choice(course_coords_m5_trace, size=min(20, len(course_coords_m5_trace)), replace=False)
    az.plot_trace(trace_m5, var_names=['course_baseline'], coords={'course': sample_course_ids_m5_trace}, compact=True, figsize=(12, 20))
    plt.suptitle(f'MCMC Traces: Sample Course Baseline Parameters - Model 5\n{len(course_coords_m5_trace):,} total courses (showing 20 random samples)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # Traceplot for sample of course baseline parameters
    plt.show()
    race_coords_m5_trace = list(trace_m5.posterior.coords['race'].values)
    sample_race_ids_m5_trace = np.random.choice(race_coords_m5_trace, size=min(20, len(race_coords_m5_trace)), replace=False)
    az.plot_trace(trace_m5, var_names=['race_difficulty'], coords={'race': sample_race_ids_m5_trace}, compact=True, figsize=(12, 20))
    plt.suptitle(f'MCMC Traces: Sample Race Difficulty Parameters - Model 5\nShared structure: affects both finish times and DNF via γ\n{len(race_coords_m5_trace):,} total races (showing 20 random samples)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # Traceplot for sample of race difficulty parameters (race-year effects)
    # NOTE: In Model 5, race_difficulty is SHARED between finish time and DNF models via gamma
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Posterior Diagnostics

    Quantitative convergence metrics (R-hat, ESS) and energy diagnostics to validate MCMC sampling quality.
    """)
    return


@app.cell
def _(az, n_courses_m5_1, n_participants, n_races_m5_1, plt, trace_m5):
    hyperparam_vars_m5 = ['mu_pace', 'beta', 'sigma_course', 'sigma_race', 'sigma_obs', 'mu_logit_dnf', 'beta_dist_dnf', 'beta_dist_dnf_quad', 'gamma', 'logit_psi']
    az.rhat(trace_m5, var_names=hyperparam_vars_m5)  # Finish time (5 params)
    az.plot_rank(trace_m5, var_names=hyperparam_vars_m5, kind='bars', figsize=(16, 8))  # DNF with quadratic + difficulty coupling (4 params)
    plt.suptitle(f'Rank Plot: Visual Convergence Check - Model 5b\n{n_participants:,} participants, {n_courses_m5_1:,} courses, {n_races_m5_1:,} races', fontsize=14, fontweight='bold')  # Reporting probability (1 param) - SIMPLIFIED from distance-dependent
    plt.tight_layout()
    plt.show()
    # Display R-hat values
    # Rank plot for visual convergence check
    az.summary(trace_m5, var_names=hyperparam_vars_m5, kind='diagnostics')
    return (hyperparam_vars_m5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Posterior Predictive Check

    Compare posterior predictions to observed data distributions. Validates model fit and identifies systematic biases.
    """)
    return


@app.cell
def _(model_m5, np, pm, trace_m5):
    # Posterior Predictive Check - Model 5
    # Generate predictions from the fitted model and compare to observed data
    # OPTIMIZED VERSION: Uses fewer samples and predictions=True for 10x+ speedup
    n_samples_ppc = 100
    # MAJOR OPTIMIZATION: Use predictions=True to avoid recomputing the full model graph
    # This computes predictions directly from parameters instead of re-running model
    # Use only 100 random posterior samples (sufficient for posterior predictive checks)
    total_samples = len(trace_m5.posterior.draw)
    random_draws = np.random.choice(total_samples, size=n_samples_ppc, replace=False)
    # Randomly select 100 samples from the posterior (from all chains)
    print(f'Using {n_samples_ppc} random posterior samples for PPC (out of {total_samples} total)')
    with model_m5:
        post_pred_m5 = pm.sample_posterior_predictive(trace_m5.posterior.isel(draw=random_draws), predictions=True, random_seed=42, progressbar=True)
    trace_m5.extend(post_pred_m5)
    # Add posterior predictive to trace
    print(f'Available variables: {list(post_pred_m5.predictions.data_vars)}')  # KEY: Use predictions mode for 10x+ speedup
    return post_pred_m5, random_draws


@app.cell
def _(
    distances_finishers_m5,
    gender_indices_finishers_m5,
    np,
    observed_times_finishers_m5,
    plt,
    post_pred_m5,
    unique_genders,
):
    post_pred_pace_m5 = post_pred_m5.predictions['pace_likelihood'].values
    post_pred_pace_flat_m5 = post_pred_pace_m5.reshape(-1, post_pred_pace_m5.shape[-1])
    standard_distances_1 = [('10K', 6.21371, 0.3), ('10mi', 10.0, 0.3), ('Half', 13.1, 0.3), ('Marathon', 26.2, 0.5), ('50K', 31.0686, 0.5), ('50mi', 50.0, 1.0), ('100K', 62.1371, 1.0), ('100mi', 100.0, 1.0)]
    for _gender_idx, _gender in enumerate(unique_genders):
        _fig, _axes = plt.subplots(2, 4, figsize=(16, 8))
        _axes = _axes.flatten()
        gender_mask = gender_indices_finishers_m5 == _gender_idx
        for _col_idx, (_name, _dist_value, _tolerance) in enumerate(standard_distances_1):
            _ax = _axes[_col_idx]
            obs_mask = np.abs(distances_finishers_m5 - _dist_value) < _tolerance
            combined_mask = obs_mask & gender_mask
            if not combined_mask.any():
                _ax.axis('off')
                _ax.set_title(f'{_name}\nNo data', fontsize=10)
                continue
            _obs_times_binned = observed_times_finishers_m5[combined_mask]
            pred_pace_binned = post_pred_pace_flat_m5[:, combined_mask]
            _distances_binned = distances_finishers_m5[combined_mask]
            pred_times_binned = pred_pace_binned * _distances_binned[np.newaxis, :]
            pred_times_flat = pred_times_binned.flatten()
            _obs_min, _obs_max = (_obs_times_binned.min(), _obs_times_binned.max())
            _x_range = _obs_max - _obs_min
            _bin_edges = np.linspace(max(0, _obs_min - 0.1 * _x_range), _obs_max + 0.3 * _x_range, 31)
            _ax.hist(_obs_times_binned, bins=_bin_edges, alpha=0.6, label='Observed', density=True, color='steelblue')
            _ax.hist(pred_times_flat, bins=_bin_edges, alpha=0.6, label='Predicted', density=True, color='orange')
            _ax.set_xlim(_bin_edges[0], _bin_edges[-1])
            _ax.set_xlabel('Time (min)', fontsize=10)
            _ax.set_ylabel('Density', fontsize=10)
            _ax.set_title(f'{_name} ({_dist_value:.1f}mi)\n{len(_obs_times_binned):,} finishers', fontsize=10)
            _ax.legend(fontsize=9)
            _ax.tick_params(labelsize=9)
        _gender_label = 'Male' if _gender == 'M' else 'Female'
        _fig.suptitle(f'Posterior Predictive Check: Finish Times ({_gender_label})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    return (standard_distances_1,)


@app.cell
def _(
    distances_results_m5,
    np,
    plt,
    post_pred_m5,
    race_indices_results_m5,
    race_to_course_m5,
    random_draws,
    reporting_results_mask_m5,
    results_1,
    standard_distances_1,
    trace_m5,
):
    post_pred_dnf_m5 = post_pred_m5.predictions['finish_likelihood'].values
    post_pred_dnf_flat_m5 = 1 - post_pred_dnf_m5.reshape(-1, post_pred_dnf_m5.shape[-1])
    # finish_likelihood is defined over reporting-course results only; restrict the
    # model-side index arrays to the same subset so columns line up.
    _distances_results_rep = distances_results_m5[reporting_results_mask_m5]
    _race_indices_results_rep = race_indices_results_m5[reporting_results_mask_m5]
    psi_course_samples = trace_m5.posterior['psi_course'].isel(draw=random_draws).values
    psi_course_flat = psi_course_samples.reshape(-1, psi_course_samples.shape[-1])
    _fig, _axes = plt.subplots(2, 4, figsize=(16, 8))
    _axes = _axes.flatten()
    for _col_idx, (_name, _dist_value, _tolerance) in enumerate(standard_distances_1):
        _ax = _axes[_col_idx]
        _obs_mask_full = np.abs(results_1['distance_miles'] - _dist_value) < _tolerance
        _obs_mask_model = np.abs(_distances_results_rep - _dist_value) < _tolerance
        if not _obs_mask_full.any():
            _ax.axis('off')
            _ax.set_title(f'{_name}\nNo data', fontsize=10)
            continue
        _race_results_full = results_1[_obs_mask_full]
        _unique_races_full = _race_results_full['event_distance_id'].unique()
        _obs_dnf_rates = []
        for _race_id in _unique_races_full:
            _race_data = _race_results_full[_race_results_full['event_distance_id'] == _race_id]
            _dnf_rate = (~_race_data['finished']).mean()
            _obs_dnf_rates.append(_dnf_rate)
        _obs_dnf_rates = np.array(_obs_dnf_rates)
        _race_indices_dist = _race_indices_results_rep[_obs_mask_model]
        _unique_races_dist = np.unique(_race_indices_dist)
        post_dnf_rates = []
        for _race_idx in _unique_races_dist:
            _race_mask_dist = _race_indices_dist == _race_idx
            _race_results_indices = np.where(_obs_mask_model)[0][_race_mask_dist]
            post_dnf_race = post_pred_dnf_flat_m5[:, _race_results_indices]
            _course_idx = race_to_course_m5[_race_idx]
            psi_for_race = psi_course_flat[:, _course_idx]
            for _sample_idx in range(post_dnf_race.shape[0]):
                psi_sample = psi_for_race[_sample_idx]
                reports = np.random.binomial(1, psi_sample)
                if reports:
                    post_dnf_rate_sample = post_dnf_race[_sample_idx, :].mean()
                else:
                    post_dnf_rate_sample = 0.0
                post_dnf_rates.append(post_dnf_rate_sample)
        post_dnf_rates = np.array(post_dnf_rates)
        _bin_edges = np.linspace(0, 0.6, 31)
        _ax.hist(_obs_dnf_rates, bins=_bin_edges, alpha=0.6, label='Observed', density=True, color='steelblue')
        _ax.hist(post_dnf_rates, bins=_bin_edges, alpha=0.6, label='Predicted', density=True, color='orange')
        _ax.set_xlim(0, 0.6)
        _ax.set_xlabel('DNF Rate (by race)', fontsize=10)
        _ax.set_ylabel('Density', fontsize=10)
        _ax.set_title(f'{_name} ({_dist_value:.1f}mi)\n{len(_unique_races_full)} races', fontsize=10)
        _ax.legend(fontsize=9)
        _ax.tick_params(labelsize=9)
    _fig.suptitle(f'Posterior Predictive Check: DNF Rates by Race', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Runner Parameters

    Gender-specific pace-distance curves showing how predicted pace scales with distance for M/F runners.
    """)
    return


@app.cell
def _(az, trace_m5):
    hyperparam_vars_m5_1 = ['mu_pace', 'beta', 'sigma_course', 'sigma_race', 'sigma_obs', 'mu_logit_dnf', 'beta_dist_dnf', 'beta_dist_dnf_quad', 'gamma', 'logit_psi']
    print(az.summary(trace_m5, var_names=hyperparam_vars_m5_1))  # Finish time (5 params)  # DNF with quadratic + difficulty coupling (4 params)  # Reporting probability (1 param) - SIMPLIFIED from distance-dependent
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Course Parameters
    """)
    return


@app.cell
def _(
    Patch,
    model_data,
    np,
    pd,
    plt,
    trace_m5,
    unique_courses_m5,
    unique_races_m5,
):
    # ============================================================================
    # PART 3: HARDEST AND EASIEST COURSES
    course_baseline_samples_m5 = trace_m5.posterior['course_baseline'].values
    course_baseline_mean_m5 = course_baseline_samples_m5.mean(axis=(0, 1))
    # Extract course baseline from posterior (persistent difficulty across all years)
    course_baseline_std_m5 = course_baseline_samples_m5.std(axis=(0, 1))
    race_difficulty_samples_m5 = trace_m5.posterior['race_difficulty'].values
    race_difficulty_mean_m5 = race_difficulty_samples_m5.mean(axis=(0, 1))
    race_difficulty_std_m5 = race_difficulty_samples_m5.std(axis=(0, 1))
    # Also extract race difficulty for comparison
    course_info_list = []  # Shape: (chains, draws, races)
    for _idx, _course_id in enumerate(unique_courses_m5):
        _course_data = model_data[model_data['course_idx'] == _idx]
        _course_name = _course_data.iloc[0]['name']
    # Create DataFrame with course info
        course_distance = _course_data.iloc[0]['distance_miles']
        n_races = _course_data['event_distance_id'].nunique()
        n_total_results = len(_course_data)
        course_info_list.append({'course_id': _course_id, 'course_idx': _idx, 'name': _course_name, 'distance_miles': course_distance, 'course_baseline_mean': course_baseline_mean_m5[_idx], 'course_baseline_std': course_baseline_std_m5[_idx], 'n_races': n_races, 'n_total_results': n_total_results})
    course_df_m5 = pd.DataFrame(course_info_list)
    course_df_sorted_m5 = course_df_m5.sort_values('course_baseline_mean')
    _fig, _ax = plt.subplots(1, 1, figsize=(14, 14))
    _n_show = 15
    easiest = course_df_sorted_m5.head(_n_show)
    hardest = course_df_sorted_m5.tail(_n_show)
    combined = pd.concat([easiest, hardest])
    combined = combined.sort_values('course_baseline_mean')
    combined['label'] = combined.apply(lambda x: f"{x['name'][:40]}... ({x['distance_miles']:.0f}mi, {x['n_races']} races)" if len(x['name']) > 40 else f"{x['name']} ({x['distance_miles']:.0f}mi, {x['n_races']} races)", axis=1)
    _colors = ['green'] * _n_show + ['red'] * _n_show
    _y_pos = np.arange(len(combined))
    _ax.barh(_y_pos, combined['course_baseline_mean'].values, xerr=combined['course_baseline_std'].values, color=_colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels(combined['label'].values, fontsize=8)
    _ax.set_xlabel('Course Baseline Difficulty (log scale)', fontsize=11, fontweight='bold')
    _ax.set_title(f'Easiest and Hardest Courses (Top {_n_show} Each) - Model 5\n' + f'Course Baseline = Persistent difficulty across all race years\n' + f'(Shared structure: affects both finish times and DNF via γ)\n' + f'Total: {len(course_df_m5)} courses', fontsize=13, fontweight='bold')
    _ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    # Sort by course baseline difficulty
    _ax.grid(axis='x', alpha=0.3)
    _legend_elements = [Patch(facecolor='green', alpha=0.6, label=f'{_n_show} Easiest'), Patch(facecolor='red', alpha=0.6, label=f'{_n_show} Hardest')]
    # Plot top 15 easiest and hardest courses
    _ax.legend(handles=_legend_elements, loc='lower right', fontsize=10)
    plt.tight_layout()
    # Get top 15 easiest and hardest
    plt.show()
    race_info_m5 = []
    for _idx, _race_id in enumerate(unique_races_m5):
        _race_data = model_data[model_data['race_idx'] == _idx]
    # Combine
        race_name = _race_data.iloc[0]['name']
        race_distance = _race_data.iloc[0]['distance_miles']
        race_year = _race_data.iloc[0]['race_date'].year if 'race_date' in _race_data.columns else 'Unknown'
    # Create labels with race count and distance info
        _course_idx = _race_data.iloc[0]['course_idx']
        race_info_m5.append({'race_id': _race_id, 'race_idx': _idx, 'name': race_name, 'year': race_year, 'distance_miles': race_distance, 'course_idx': _course_idx, 'course_baseline': course_baseline_mean_m5[_course_idx], 'race_difficulty_mean': race_difficulty_mean_m5[_idx], 'race_difficulty_std': race_difficulty_std_m5[_idx], 'n_observations': len(_race_data)})
    race_df_m5 = pd.DataFrame(race_info_m5)
    course_variation_stats = []
    for _course_idx in range(len(unique_courses_m5)):
        _course_id = unique_courses_m5[_course_idx]
        course_races = race_df_m5[race_df_m5['course_idx'] == _course_idx]
    # Color code: green for easy, red for hard
        if len(course_races) > 1:
            race_diff_var = course_races['race_difficulty_mean'].std()
    # Plot horizontal bar chart
            course_variation_stats.append({'course_id': _course_id, 'course_idx': _course_idx, 'name': course_races.iloc[0]['name'], 'distance_miles': course_races.iloc[0]['distance_miles'], 'course_baseline': course_baseline_mean_m5[_course_idx], 'n_years': len(course_races), 'race_year_std': race_diff_var, 'n_total_observations': course_races['n_observations'].sum()})
    course_var_df = pd.DataFrame(course_variation_stats)
    _fig, _axes = plt.subplots(1, 2, figsize=(16, 7))
    _ax = _axes[0]
    course_var_sorted = course_var_df.sort_values('race_year_std')
    top_consistent = course_var_sorted.head(15)
    top_consistent['label'] = top_consistent.apply(lambda x: f"{x['name'][:35]}... ({x['distance_miles']:.0f}mi, {x['n_years']} yrs)" if len(x['name']) > 35 else f"{x['name']} ({x['distance_miles']:.0f}mi, {x['n_years']} yrs)", axis=1)
    _y_pos = np.arange(len(top_consistent))
    _ax.barh(_y_pos, top_consistent['race_year_std'].values, color='green', alpha=0.6, edgecolor='black', linewidth=0.5)
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels(top_consistent['label'].values, fontsize=8)
    _ax.set_xlabel('Year-to-Year Difficulty Variation (SD)', fontsize=11)
    _ax.set_title('Most Consistent Courses - Model 5\n(Lowest year-to-year variation)', fontsize=12, fontweight='bold')
    _ax.grid(axis='x', alpha=0.3)
    _ax = _axes[1]
    top_variable = course_var_sorted.tail(15).sort_values('race_year_std')
    # Add legend
    top_variable['label'] = top_variable.apply(lambda x: f"{x['name'][:35]}... ({x['distance_miles']:.0f}mi, {x['n_years']} yrs)" if len(x['name']) > 35 else f"{x['name']} ({x['distance_miles']:.0f}mi, {x['n_years']} yrs)", axis=1)
    _y_pos = np.arange(len(top_variable))
    _ax.barh(_y_pos, top_variable['race_year_std'].values, color='red', alpha=0.6, edgecolor='black', linewidth=0.5)
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels(top_variable['label'].values, fontsize=8)
    _ax.set_xlabel('Year-to-Year Difficulty Variation (SD)', fontsize=11)
    _ax.set_title('Most Variable Courses - Model 5\n(Highest year-to-year variation)', fontsize=12, fontweight='bold')
    _ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    # PART 4: COURSE BASELINE VS RACE-YEAR VARIATION
    sigma_course_samples = trace_m5.posterior['sigma_course'].values.flatten()
    sigma_race_samples = trace_m5.posterior['sigma_race'].values.flatten()
    # Create DataFrame with race info for variation analysis
    print(f'\n=== HIERARCHICAL VARIANCE COMPONENTS ===')
    print(f'Course baseline variation (σ_course): {sigma_course_samples.mean():.3f} ± {sigma_course_samples.std():.3f}')
    print(f'Race-year variation (σ_race): {sigma_race_samples.mean():.3f} ± {sigma_race_samples.std():.3f}')
    print(f'Ratio (σ_race / σ_course): {(sigma_race_samples / sigma_course_samples).mean():.2f}')
    print(f'\nInterpretation:')
    if (sigma_race_samples / sigma_course_samples).mean() < 0.5:
        print('  • Course-level effects dominate (persistent terrain/elevation)')
    elif (sigma_race_samples / sigma_course_samples).mean() > 1.5:
        print('  • Race-year effects dominate (variable weather/conditions)')
    else:
    # For each course, calculate the variation in race difficulty across years
    # Plot courses with highest year-to-year variation
    # Panel 1: Courses with most consistent difficulty (low year-to-year variation)
    # Create labels
    # Panel 2: Courses with most variable difficulty (high year-to-year variation)
    # Extract sigma_course and sigma_race for comparison
        print('  • Both course and race-year effects are important')  # Need at least 2 years to measure variation
    return (race_df_m5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Race-Year Conditions
    """)
    return


@app.cell
def _(Patch, np, pd, plt, race_df_m5):
    race_df_sorted_m5 = race_df_m5.sort_values('race_difficulty_mean')
    _n_show = 20
    easiest_races = race_df_sorted_m5.head(_n_show)
    hardest_races = race_df_sorted_m5.tail(_n_show)
    combined_races = pd.concat([easiest_races, hardest_races])
    combined_races = combined_races.sort_values('race_difficulty_mean')
    combined_races['label'] = combined_races.apply(lambda x: f"{x['name'][:35]}... {x['year']} ({x['distance_miles']:.0f}mi, n={x['n_observations']})" if len(x['name']) > 35 else f"{x['name']} {x['year']} ({x['distance_miles']:.0f}mi, n={x['n_observations']})", axis=1)
    _colors = ['green'] * _n_show + ['red'] * _n_show
    _fig, _ax = plt.subplots(1, 1, figsize=(14, 16))
    _y_pos = np.arange(len(combined_races))
    _ax.barh(_y_pos, combined_races['race_difficulty_mean'].values, xerr=combined_races['race_difficulty_std'].values, color=_colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels(combined_races['label'].values, fontsize=8)
    _ax.set_xlabel('Race-Year Difficulty (race_difficulty parameter)', fontsize=11, fontweight='bold')
    _ax.set_title(f'Easiest and Hardest Race-Year Conditions (Top {_n_show} Each) - Model 5\n' + f'Race Difficulty = Year-specific conditions (weather, trail conditions, etc.)\n' + f'(Shared structure: affects both finish times and DNF via γ)\n' + f'Total: {len(race_df_m5)} race-years', fontsize=13, fontweight='bold')
    _ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='Mean')
    _ax.grid(axis='x', alpha=0.3)
    _legend_elements = [Patch(facecolor='green', alpha=0.6, label=f'{_n_show} Easiest Conditions'), Patch(facecolor='red', alpha=0.6, label=f'{_n_show} Hardest Conditions')]
    _ax.legend(handles=_legend_elements, loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.show()
    print('\n' + '=' * 80)
    print('RACE-YEAR DIFFICULTY SUMMARY STATISTICS')
    print('=' * 80)
    print(f'\nTotal race-years: {len(race_df_m5)}')
    print(f"Mean race difficulty: {race_df_m5['race_difficulty_mean'].mean():.3f}")
    print(f"SD of race difficulty: {race_df_m5['race_difficulty_mean'].std():.3f}")
    print(f"Min race difficulty: {race_df_m5['race_difficulty_mean'].min():.3f}")
    print(f"Max race difficulty: {race_df_m5['race_difficulty_mean'].max():.3f}")
    print(f"\n{'=' * 80}")
    print('TOP 5 EASIEST RACE-YEAR CONDITIONS:')
    print(f"{'=' * 80}")
    for _idx, _row in easiest_races.head(5).iterrows():
        print(f"{_row['name']} ({_row['year']}) - {_row['distance_miles']:.0f}mi")
        print(f"  Race difficulty: {_row['race_difficulty_mean']:.3f} ± {_row['race_difficulty_std']:.3f}")
        print(f"  Course baseline: {_row['course_baseline']:.3f}")
        print(f"  Sample size: n={_row['n_observations']}")
        print()
    print(f"{'=' * 80}")
    print('TOP 5 HARDEST RACE-YEAR CONDITIONS:')
    print(f"{'=' * 80}")
    for _idx, _row in hardest_races.tail(5).iloc[::-1].iterrows():
        print(f"{_row['name']} ({_row['year']}) - {_row['distance_miles']:.0f}mi")
        print(f"  Race difficulty: {_row['race_difficulty_mean']:.3f} ± {_row['race_difficulty_std']:.3f}")
        print(f"  Course baseline: {_row['course_baseline']:.3f}")
        print(f"  Sample size: n={_row['n_observations']}")
        print()
    return


if __name__ == "__main__":
    app.run()
