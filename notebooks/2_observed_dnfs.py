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
    # Observed DNF Model

    **Hierarchical Bayesian Model for Ultrarunning Pace with Selection Bias Quantification**

    This notebook implements a sophisticated two-stage hierarchical Bayesian model to estimate ultrarunning race performance while **directly quantifying selection bias** from DNFs (Did Not Finish). The model uses correlated course effects to learn the relationship between course difficulty and DNF rates, enabling inference about unobserved finish times for runners who DNF'd.

    ## Model Architecture

    **Gender-Specific Power Law + Correlated Course Effects:**
    - **Finish Time Model**: `pace[gender] ~ distance^β[gender] × course_difficulty`
    - **DNF Model**: `P(DNF | course, distance)` with course-specific intercepts
    - **Key Innovation**: LKJ correlation prior jointly models finish time and DNF effects
      - **Positive correlation** → harder courses cause more DNFs → observed times underestimate population mean
      - **Correlation magnitude** directly quantifies selection bias severity

    ## Two-Stage Estimation Strategy

    **Stage 1: K-Core MCMC (High-Quality Hyperparameters)**
    - **K-core subset** (alpha, beta parameters): Densely-connected runners + courses
    - **Full Bayesian inference**: MCMC sampling for hyperparameters + course effects
    - **Purpose**: Obtain reliable estimates of population-level parameters

    **Stage 2: MAP Extension (Full Dataset)**
    - **Fixed hyperparameters** from K-core posterior medians
    - **Estimate only**: Course-specific effects for all courses (including sparse ones)
    - **Purpose**: Extend predictions to full dataset efficiently

    ## Data Structure

    **Finish Time Observations** (n_finishers):
    - **Coordinates**: `gender` (M/F), `course` (race name + distance), `log_distance_ratio`
    - **Outcome**: `time` (observed finish time in seconds)
    - **Likelihood**: LogNormal with **fixed noise** (σ=0.15) to prevent unrealistic heavy tails

    **DNF Observations** (n_results):
    - **Coordinates**: `course`, `log_distance_ratio`
    - **Outcome**: `did_finish` (binary: 1=finished, 0=DNF)
    - **Likelihood**: Bernoulli(logit_p)
    - **Note**: DNF model **pools across genders** (assumes equal DNF rates M/F)

    ## Model Parameters

    **Population-Level Hyperparameters:**
    - `pace_marathon[gender]`: Baseline pace at marathon distance (gender-specific)
    - `pace_distance_effect[gender]`: Power law exponent (how pace degrades with distance)
    - `dnf_rate_marathon`: Baseline DNF probability at marathon distance (logit scale)
    - `dnf_distance_multiplier`: How DNF probability scales with log(distance)

    **Correlated Course Effects** (n_courses × 2):
    - `course_effects[:, 0]`: Finish time multiplier (shared across genders)
    - `course_effects[:, 1]`: DNF probability multiplier
    - `course_correlation`: **THE KEY PARAMETER** - quantifies selection bias
      - Positive correlation → selection bias present
      - Magnitude → severity of bias

    **Non-Centered Parameterization:**
    ```python
    course_effects_raw ~ Normal(0, 1)
    chol ~ LKJCholeskyCov(n=2, eta=2, sd_dist=course_effect_sds)
    course_effects = course_effects_raw @ chol  # Correlated effects
    ```

    ## Key Design Decisions

    1. **Fixed Observation Noise (σ=0.15)**: Prevents unrealistic heavy tails in LogNormal likelihood
    2. **Gender-Pooled DNF Model**: Assumes M/F have equal DNF rates (could be relaxed)
    3. **Shared Course Difficulty**: Course effects identical for M/F (could allow gender-specific)
    4. **Strong Regularization**: Prior widths tightened 10× to prevent unrealistic predictions
    5. **DNF Filtering**: Only includes courses with ≥1 observed DNF to avoid non-reporting bias

    ## Model Output

    **Primary Inference Target:**
    - **Selection Bias Estimate**: `expected_bias_pct ≈ median_dnf_rate × median_correlation`
      - Quantifies how much observed finish times underestimate population mean
      - Direct measurement of survivorship bias in ultrarunning performance data

    **Secondary Outputs:**
    - Gender-specific pace predictions across distances
    - Course-specific difficulty rankings
    - Posterior predictive distributions for counterfactual scenarios
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    Load and process the whole dataset
    """)
    return


@app.cell
def _():
    import os
    import platform
    import multiprocessing

    # CRITICAL: Set JAX environment variables BEFORE any imports that might load JAX
    os.environ['MallocStackLogging'] = '0'

    IS_MACOS = platform.system() == 'Darwin'
    IS_LINUX = platform.system() == 'Linux'
    N_CPUS = multiprocessing.cpu_count()

    if IS_MACOS:
        # macOS: Force CPU-only to avoid Metal backend issues
        os.environ['JAX_PLATFORMS'] = 'cpu'
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
    elif IS_LINUX:
        # Linux: Enable GPU if available, otherwise optimize CPU
        os.environ['OMP_NUM_THREADS'] = str(max(1, N_CPUS // 4))
        os.environ['MKL_NUM_THREADS'] = str(max(1, N_CPUS // 4))

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    import pickle
    from pathlib import Path
    import time

    import jax
    import numpy as np
    import pandas as pd
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Configure matplotlib
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Our utilities
    from utils.data_processing import load_results, process_results, filter_races_with_dnfs
    from utils.kcore_subsetting import subset_kcore_data
    from utils.mcmc_notifications import notify_mcmc_start, notify_mcmc_complete, notify_mcmc_error, build_mcmc_summary_string
    from utils.empirical_priors import (
        calculate_gender_marathon_pace,
        calculate_gender_distance_exponent,
        calculate_course_variation,
        calculate_dnf_priors
    )
    from utils.utils import plot_posterior_diagnostics

    # Detect GPU and configure sampler (after JAX import)
    GPU_DEVICES = [d for d in jax.devices() if d.platform == 'gpu']
    HAS_GPU = len(GPU_DEVICES) > 0
    JAX_BACKEND = jax.default_backend()
    JAX_DEVICE_COUNT = jax.local_device_count()  # Number of devices available for parallel chains

    if IS_MACOS:
        # macOS: Use blackjax for better CPU compatibility
        NUTS_SAMPLER = 'blackjax'
        N_CHAINS = 4
        N_CORES = 4
        PLATFORM_CONFIG = f"macOS (CPU + blackjax, {JAX_DEVICE_COUNT} devices)"
    elif IS_LINUX and HAS_GPU:
        # Linux with GPU: Use numpyro for CUDA optimization
        NUTS_SAMPLER = 'numpyro'
        N_CHAINS = 4
        N_CORES = 4
        PLATFORM_CONFIG = f"Linux + GPU (numpyro, {len(GPU_DEVICES)} GPU(s), {JAX_DEVICE_COUNT} JAX device(s))"
    else:
        # Linux CPU-only: Use numpyro with CPU chains
        NUTS_SAMPLER = 'numpyro'
        N_CHAINS = min(4, N_CPUS)
        N_CORES = N_CHAINS
        PLATFORM_CONFIG = f"Linux CPU (numpyro, {JAX_DEVICE_COUNT} devices)"

    print(f"✅ Platform configuration: {PLATFORM_CONFIG}")
    print(f"   JAX backend: {JAX_BACKEND}")
    print(f"   JAX device count: {JAX_DEVICE_COUNT}")
    print(f"   Sampler: {NUTS_SAMPLER}, Chains: {N_CHAINS}, Cores: {N_CORES}")
    if JAX_DEVICE_COUNT < N_CHAINS:
        print(f"   ⚠️  Note: {N_CHAINS} chains will run sequentially on {JAX_DEVICE_COUNT} device(s)")
    return (
        NUTS_SAMPLER,
        PLATFORM_CONFIG,
        Path,
        az,
        build_mcmc_summary_string,
        calculate_course_variation,
        calculate_dnf_priors,
        calculate_gender_distance_exponent,
        calculate_gender_marathon_pace,
        filter_races_with_dnfs,
        jax,
        load_results,
        multiprocessing,
        notify_mcmc_complete,
        notify_mcmc_error,
        notify_mcmc_start,
        np,
        os,
        pd,
        platform,
        plot_posterior_diagnostics,
        plt,
        pm,
        process_results,
        pt,
        sns,
        subset_kcore_data,
        time,
    )


@app.cell
def _(az, jax, multiprocessing, np, os, platform, pm):
    print('=' * 80)
    print('HARDWARE INFORMATION')
    print('=' * 80)
    print(f'Platform: {platform.platform()}')
    print(f'Processor: {platform.processor()}')
    print(f'Python: {platform.python_version()}')
    cpu_count = multiprocessing.cpu_count()
    print(f'\nCPU Cores: {cpu_count}')
    jax_cpu_devices = [d for d in jax.devices() if d.platform == 'cpu']
    print(f'JAX CPU Devices: {len(jax_cpu_devices)}')
    xla_flags = os.environ.get('XLA_FLAGS', 'Not set')
    print(f'XLA_FLAGS: {xla_flags}')
    omp_threads = os.environ.get('OMP_NUM_THREADS', 'Not set')
    mkl_threads = os.environ.get('MKL_NUM_THREADS', 'Not set')
    print(f'OMP_NUM_THREADS: {omp_threads}')
    print(f'MKL_NUM_THREADS: {mkl_threads}')
    print(f'\nJAX Backend: {jax.default_backend()}')
    print(f'JAX Device Count: {jax.local_device_count()} (available for parallel chains)')
    gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
    if gpu_devices:
        print(f'GPU Available: YES - {len(gpu_devices)} device(s)')
        for _i, device in enumerate(gpu_devices):
            print(f'  GPU {_i}: {device.device_kind}')
        try:
            if hasattr(jax.lib, 'xla_bridge'):
                cuda_version = jax.lib.xla_bridge.get_backend().platform_version
                print(f'  CUDA Version: {cuda_version}')
        except Exception:
            pass
    else:
        print(f'GPU Available: NO (using CPU)')
    print(f'\nPackage Versions:')
    print(f'  NumPy: {np.__version__}')
    print(f'  JAX: {jax.__version__}')
    print(f'  PyMC: {pm.__version__}')
    print(f'  ArviZ: {az.__version__}')
    return


@app.cell
def _(load_results, process_results):
    # load and process data

    results = load_results()
    results = process_results(results)
    results = results[results['distance_miles'] >= 26.2]
    return (results,)


@app.cell
def _():
    # Standard race distances for analysis and visualization
    # Format: (distance_miles, label, tolerance_for_binning)
    standard_distances = [
        (26.2, 'Marathon', 0.5),
        (31.0686, '50K', 0.5),
        (50.0, '50mi', 1.0),
        (62.1371, '100K', 1.0),
        (100.0, '100mi', 1.0)
    ]

    reference_distance = 26.2  # Marathon
    return reference_distance, standard_distances


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Filtering

    Filter the dataset to the results relevant to this model
    """)
    return


@app.cell
def _(filter_races_with_dnfs, results):
    # filter out non-DNF reporting races
    # this model does not attempt to model DNF reporting, but does model DNFs themselves
    results_1 = filter_races_with_dnfs(results)
    results_1 = results_1[results_1['distance_miles'] >= 26.2]
    return (results_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Subsetting

    Subset the data to a well-connected subgraph for computational efficiency
    """)
    return


@app.cell
def _(Path, os, results_1, subset_kcore_data):
    # K-core subsetting
    alpha = 3
    # (3, 840): 11 courses
    # (3, 629): 93 courses
    # (3, 423): 210 courses
    # (3, 220): 541 courses
    # (3, 233): 510 courses
    beta = 233
    # Path is anchored to the notebook file so caches resolve regardless of
    # marimo's CWD (which is the dir you ran 'uv run marimo' from, not notebooks/)
    model_dir = Path(__file__).resolve().parent / 'models' / 'model_2'
    os.makedirs(model_dir, exist_ok=True)
    subset_dir = f'{model_dir}/alpha{alpha}_beta{beta}'
    os.makedirs(subset_dir, exist_ok=True)
    results_2 = subset_kcore_data(results_1, alpha=alpha, beta=beta)
    model_data = results_2[results_2['in_kcore'] | results_2['in_closure']]
    return alpha, beta, model_data, model_dir, results_2, subset_dir


@app.cell
def _(alpha, beta, model_data, np, plt, results_2):
    kcore_data = model_data[model_data['in_kcore']].copy()
    closure_data = model_data[model_data['in_closure']].copy()
    kcore_runners = kcore_data['participant_id'].unique()
    runner_total_courses = results_2[results_2['participant_id'].isin(kcore_runners)].groupby('participant_id')['name'].nunique()
    runner_kcore_courses = kcore_data.groupby('participant_id')['name'].nunique()
    runner_kcore_completeness = (runner_kcore_courses / runner_total_courses * 100).values
    kcore_courses = kcore_data[['name', 'distance_miles']].drop_duplicates()
    kcore_courses['key'] = kcore_courses['name'] + '_' + kcore_courses['distance_miles'].astype(str)
    results_2['key'] = results_2['name'] + '_' + results_2['distance_miles'].astype(str)
    kcore_data['key'] = kcore_data['name'] + '_' + kcore_data['distance_miles'].astype(str)
    total_runners_per_course = results_2.groupby('key')['participant_id'].nunique()
    kcore_runners_per_course = kcore_data.groupby('key')['participant_id'].nunique()
    course_kcore_completeness = (kcore_runners_per_course / total_runners_per_course * 100).reindex(kcore_courses['key']).values
    closure_runners = closure_data['participant_id'].unique()
    runner_total_courses_closure = results_2[results_2['participant_id'].isin(closure_runners)].groupby('participant_id')['name'].nunique()
    runner_closure_courses = closure_data.groupby('participant_id')['name'].nunique()
    runner_closure_completeness = (runner_closure_courses / runner_total_courses_closure * 100).values
    model_data_grouped = model_data.assign(key=model_data['name'] + '_' + model_data['distance_miles'].astype(str))
    all_model_courses = model_data_grouped[['name', 'distance_miles', 'key']].drop_duplicates()
    model_runners_per_course = model_data_grouped.groupby('key')['participant_id'].nunique()
    course_kcore_completeness = (kcore_runners_per_course / total_runners_per_course * 100).reindex(kcore_courses['key']).values
    course_closure_completeness = (model_runners_per_course / total_runners_per_course * 100).reindex(all_model_courses['key']).values

    def freedman_diaconis_bins(data):
        """Calculate number of bins using Freedman-Diaconis rule"""
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / len(data) ** (1 / 3)
        n_bins = int(np.ceil((data.max() - data.min()) / bin_width))
        return max(10, min(n_bins, 100))
    all_data = np.concatenate([runner_kcore_completeness, course_kcore_completeness, runner_closure_completeness, course_closure_completeness])
    n_bins = freedman_diaconis_bins(all_data)
    runner_kcore_mean = runner_kcore_completeness.mean()
    runner_closure_mean = runner_closure_completeness.mean()
    course_kcore_mean = course_kcore_completeness.mean()
    course_closure_mean = course_closure_completeness.mean()
    n_kcore_courses = len(course_kcore_completeness)
    n_all_model_courses = len(course_closure_completeness)
    _fig, (_ax_runner, _ax_course) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    _ax_runner.hist(runner_kcore_completeness, bins=n_bins, range=(0, 100), density=True, label=f'K-Core (n={len(runner_kcore_completeness):,}, μ={runner_kcore_mean:.1f}%)', color='C0', alpha=0.7, edgecolor='none')
    _ax_runner.hist(runner_closure_completeness, bins=n_bins, range=(0, 100), density=True, label=f'Closure (n={len(runner_closure_completeness):,}, μ={runner_closure_mean:.1f}%)', color='C1', alpha=0.7, edgecolor='none')
    _ax_runner.axvline(runner_kcore_mean, linestyle='--', color='C0', linewidth=2, alpha=0.8)
    _ax_runner.axvline(runner_closure_mean, linestyle='--', color='C1', linewidth=2, alpha=0.8)
    _ax_runner.set_ylabel('Density', fontsize=11)
    _ax_runner.set_title('Runners', fontsize=11, fontweight='bold', loc='left')
    _ax_runner.legend(loc='upper left', fontsize=9, framealpha=0.9)
    _ax_runner.grid(True, alpha=0.3)
    _ax_runner.set_xlim(0, 100)
    _ax_course.hist(course_kcore_completeness, bins=n_bins, range=(0, 100), density=True, label=f'K-Core Only (n={n_kcore_courses:,}, μ={course_kcore_mean:.1f}%)', color='C0', alpha=0.7, edgecolor='none')
    _ax_course.hist(course_closure_completeness, bins=n_bins, range=(0, 100), density=True, label=f'K-Core + Closure (n={n_all_model_courses:,}, μ={course_closure_mean:.1f}%)', color='C1', alpha=0.7, edgecolor='none')
    _ax_course.axvline(course_kcore_mean, linestyle='--', color='C0', linewidth=2, alpha=0.8)
    _ax_course.axvline(course_closure_mean, linestyle='--', color='C1', linewidth=2, alpha=0.8)
    _ax_course.set_xlabel('Completeness (%)', fontsize=11)
    _ax_course.set_ylabel('Density', fontsize=11)
    _ax_course.set_title('Courses', fontsize=11, fontweight='bold', loc='left')
    _ax_course.legend(loc='upper left', fontsize=9, framealpha=0.9)
    _ax_course.grid(True, alpha=0.3)
    _ax_course.set_xlim(0, 100)
    _fig.suptitle(f'K-Core and Closure Completeness Distributions (α={alpha}, β={beta})', fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    return (kcore_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specification

    Define the model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Empirical Priors

    Compute empirircal priors on all the results relvant to this model
    """)
    return


@app.cell
def _(
    calculate_course_variation,
    calculate_dnf_priors,
    calculate_gender_distance_exponent,
    calculate_gender_marathon_pace,
    np,
    results_2,
):
    # Calculate empirical priors from observed data
    print('\n' + '=' * 80)
    print('MODEL 2 EMPIRICAL PRIORS')
    print('=' * 80)
    REFERENCE_DISTANCE = 26.2
    # Reference distance for marathon (miles)
    print(f'Reference distance: {REFERENCE_DISTANCE} miles (marathon)\n')
    results_finishers = results_2[results_2['finished'] & (results_2['time_ms'] > 0)].copy()
    person_counts = results_2.groupby('participant_id').size()
    # Create results_finishers (subset of results with finished=True and valid times)
    multi_race_people = person_counts[person_counts >= 2].index
    print(f'Data overview:')
    # Identify multi-race participants for k-core filtering
    print(f'  Total results: {len(results_2):,}')
    print(f'  Finishers: {len(results_finishers):,}')
    print(f'  Multi-race participants (≥2 races): {len(multi_race_people):,}')
    print()
    print('FINISH TIME MODEL (computed from finishers only):')
    mu_pace_m = calculate_gender_marathon_pace(results_finishers, 'M', REFERENCE_DISTANCE)
    beta_m = calculate_gender_distance_exponent(results_finishers, 'M', REFERENCE_DISTANCE)
    mu_pace_f = calculate_gender_marathon_pace(results_finishers, 'F', REFERENCE_DISTANCE)
    beta_f = calculate_gender_distance_exponent(results_finishers, 'F', REFERENCE_DISTANCE)
    # FINISH TIME MODEL - Use simple utility functions from empirical_priors module
    print(f'M (M):')
    print(f'  Log marathon pace (mu_pace): {mu_pace_m:.3f}')
    print(f'  Marathon pace: {np.exp(mu_pace_m):.2f} min/mile')
    print(f'  Distance exponent (beta): {beta_m:.4f}')
    print()
    print(f'F (F):')
    print(f'  Log marathon pace (mu_pace): {mu_pace_f:.3f}')
    print(f'  Marathon pace: {np.exp(mu_pace_f):.2f} min/mile')
    print(f'  Distance exponent (beta): {beta_f:.4f}')
    print()
    model_data_finishers_for_course_calc = results_finishers[results_finishers['participant_id'].isin(multi_race_people)]
    sigma_course_prior = calculate_course_variation(model_data_finishers_for_course_calc, REFERENCE_DISTANCE)
    print(f'Course variation (sigma_course): {sigma_course_prior:.3f} (from k-core participants)')
    print()
    sigma_pace_marathon = 2 * abs(mu_pace_m - mu_pace_f)
    sigma_pace_distance_effect = 2 * abs(beta_m - beta_f)
    sigma_finish_time_noise = 0.1
    print('PRIOR WIDTHS (for weakly informative priors):')
    # Course-level variation - Calculate on k-core subset
    print(f'  pace_marathon sigma: {sigma_pace_marathon:.3f} (2x gender difference)')
    print(f'  pace_distance_effect sigma: {sigma_pace_distance_effect:.3f} (2x gender difference)')
    print(f'  finish_time_noise sigma: {sigma_finish_time_noise:.2f} (domain knowledge: ~20% CV)')
    print(f'    → Cannot calculate empirically - this is residual AFTER all modeling')
    print(f'    → Based on typical within-race variation in ultramarathons')
    # PRIOR WIDTHS: Calculate from empirical data using gender differences
    print()
    print('DNF MODEL (computed from all results):')
    dnf_priors = calculate_dnf_priors(results_2, REFERENCE_DISTANCE)
    # FINISH TIME NOISE: Domain knowledge prior (not empirically calculated)
    # Rationale: Noise is what's left AFTER modeling all effects (gender, distance, course)
    # We haven't modeled yet, so we can't know this empirically
    # Use domain knowledge: typical within-race CV for ultramarathons is 15-20%
    # For log-normal, sigma ≈ CV, so use 0.2 as weakly informative prior scale
    mu_logit_dnf = dnf_priors['mu_logit_dnf']  # Domain knowledge: ~20% CV
    beta_dist_dnf = dnf_priors['beta_dist_dnf']
    sigma_course_dnf_prior = dnf_priors['sigma_course_dnf']
    course_dnf_rates = results_2.groupby('name')['finished'].apply(lambda x: (~x).mean())
    median_course_dnf_rate = course_dnf_rates.median()
    median_course_dnf_rate = np.clip(median_course_dnf_rate, 0.01, 0.99)
    median_course_dnf_rate = median_course_dnf_rate * 1.15
    median_course_dnf_rate = np.clip(median_course_dnf_rate, 0.01, 0.99)
    mu_logit_dnf = float(np.log(median_course_dnf_rate / (1 - median_course_dnf_rate)))
    sigma_course_dnf_prior = sigma_course_dnf_prior * 0.4
    # DNF model parameters - all from empirical priors function
    beta_dist_dnf = beta_dist_dnf * 0.7
    print(f'  Baseline DNF log-odds (MEDIAN COURSE + 15%): {mu_logit_dnf:.3f}')
    print(f'  → Corresponds to DNF rate: {median_course_dnf_rate:.3f}')
    print(f'  Distance effect on DNF (beta_dist_dnf): {beta_dist_dnf:.3f} (scaled down 30%)')
    print(f'  Course-level DNF variation (sigma_course_dnf): {sigma_course_dnf_prior:.3f} (scaled down 60%)')
    print()
    sigma_dnf_rate_marathon = sigma_course_dnf_prior
    # OVERRIDE: Use median course DNF rate instead of marathon-specific rate
    # The marathon baseline (4% DNF) is too low for the typical ultra course (~30% DNF)
    # This causes the prior to predict too many courses with near-zero DNF rates
    sigma_dnf_distance_multiplier = abs(beta_dist_dnf) * 0.5
    print(f'DNF PRIOR WIDTHS:')
    print(f'  dnf_rate_marathon sigma: {sigma_dnf_rate_marathon:.3f} (course variation)')
    print(f'  dnf_distance_multiplier sigma: {sigma_dnf_distance_multiplier:.3f} (50% of estimate)')
    # BOOST the baseline slightly above median to match observed peak around 0.35
    # Median is ~0.3 but observed distributions peak around 0.35
    # OVERRIDE: Reduce course-level DNF variation to prevent extreme predictions
    # The empirical sigma_course_dnf is calculated from ALL courses (including outliers)
    # This creates too much variation, predicting both peaks at 0 and tails at >0.6
    # Scale down by 60% to further constrain high DNF predictions
    # OVERRIDE: Reduce distance effect to prevent extreme DNF predictions at long distances
    # The empirical beta_dist_dnf creates too much increase for 100K/100mi races
    # Scale down by 30% to reduce the right tail of high DNF rate predictions
    # Calculate prior widths for DNF hyperparameters
    # For mu_logit_dnf: Use course-level variation as uncertainty
    # For beta_dist_dnf: Use a conservative fraction of the estimate itself
    print('=' * 80 + '\n')  # Boost by 15%  # Use empirical course variation  # 50% uncertainty
    return (
        REFERENCE_DISTANCE,
        beta_dist_dnf,
        beta_f,
        beta_m,
        mu_logit_dnf,
        mu_pace_f,
        mu_pace_m,
        sigma_course_dnf_prior,
        sigma_course_prior,
        sigma_dnf_distance_multiplier,
        sigma_dnf_rate_marathon,
        sigma_finish_time_noise,
        sigma_pace_distance_effect,
        sigma_pace_marathon,
    )


@app.cell
def _(REFERENCE_DISTANCE, model_data, np):
    # Extract model arrays from subsetted data
    # Separate finishers (for finish time model) from all results (for DNF model)
    model_data_finishers = model_data[model_data['finished'] & (model_data['time_ms'] > 0)].copy()
    unique_courses = model_data['name'].unique()
    # Create categorical indices for vectorized operations
    unique_genders = np.array(['M', 'F'])
    course_to_idx = {course: _idx for _idx, course in enumerate(unique_courses)}
    gender_to_idx = {'M': 0, 'F': 1}
    # Map courses and genders to integer indices
    course_indices_finishers = model_data_finishers['name'].map(course_to_idx).values
    gender_indices_finishers = model_data_finishers['gender'].map(gender_to_idx).values
    distances_finishers = model_data_finishers['distance_miles'].values
    # Finish time model data (finishers only)
    observed_times_finishers = model_data_finishers['time_ms'].values / 60000
    course_indices_results = model_data['name'].map(course_to_idx).values
    distances_results = model_data['distance_miles'].values
    did_finish = model_data['finished'].values.astype(int)  # Convert to minutes
    n_courses = len(unique_courses)
    # DNF model data (all results)
    _n_runners = model_data['participant_id'].nunique()
    unique_course_distances = np.array([distances_finishers[course_indices_finishers == _i].mean() if (course_indices_finishers == _i).any() else distances_results[course_indices_results == _i].mean() if (course_indices_results == _i).any() else REFERENCE_DISTANCE for _i in range(n_courses)])
    unique_log_distance_ratios = np.log(unique_course_distances / REFERENCE_DISTANCE)
    log_distance_ratio_finishers = np.log(distances_finishers / REFERENCE_DISTANCE)
    log_distances_finishers = np.log(distances_finishers)
    print(f'Model data prepared:')
    print(f'  Runners: {_n_runners:,}')
    # PERFORMANCE OPTIMIZATION: Pre-compute course distances BEFORE model definition
    # This prevents Python loops inside the PyMC model context (enables JAX compilation)
    print(f'  Total results (finishers + DNFs): {len(did_finish):,}')
    print(f'  Courses: {n_courses:,}')
    # PERFORMANCE OPTIMIZATION: Pre-compute log distance ratios (avoid computing logs every MCMC iteration)
    print(f'  Finisher observations: {len(observed_times_finishers):,}')
    return (
        model_data_finishers,
        n_courses,
        unique_courses,
        unique_genders,
        unique_log_distance_ratios,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generative Model
    """)
    return


@app.cell
def _(
    REFERENCE_DISTANCE,
    beta_dist_dnf,
    beta_f,
    beta_m,
    mu_logit_dnf,
    mu_pace_f,
    mu_pace_m,
    np,
    pm,
    pt,
    sigma_course_dnf_prior,
    sigma_course_prior,
    sigma_dnf_distance_multiplier,
    sigma_dnf_rate_marathon,
    sigma_finish_time_noise,
    sigma_pace_distance_effect,
    sigma_pace_marathon,
):
    def define_model(model_data, fixed_params=None):
        """
        Define the hierarchical Bayesian model for ultramarathon finish times and DNF rates.

        This function builds the complete model structure with correlated course effects.
        Hyperparameters can either be estimated via priors (MCMC) or fixed at provided values (MAP).

        Parameters
        ----------
        model_data : pd.DataFrame
            DataFrame containing race results with columns: name (course), gender, distance_miles, 
            time_ms (for finishers), finished (bool)
        fixed_params : dict, optional
            Dictionary of hyperparameter values to fix. If None, all parameters are estimated.
            Expected keys (when provided):
            - 'pace_marathon': array of length 2 [M, F]
            - 'pace_distance_effect': array of length 2 [M, F]
            - 'dnf_rate_marathon': scalar
            - 'dnf_distance_multiplier': scalar
            - 'chol': pre-computed Cholesky matrix (2x2)
            If any key is missing, that parameter will be estimated via priors.

        Returns
        -------
        pm.Model
            PyMC model ready for sampling (MCMC) or optimization (MAP)
        """
        unique_courses = model_data['name'].unique()
        unique_genders = np.array(['M', 'F'])  # === PREPARE DATA ARRAYS ===
        course_to_idx = {course: _idx for _idx, course in enumerate(unique_courses)}
        gender_to_idx = {'M': 0, 'F': 1}  # Extract unique coordinate values
        finishers = model_data[model_data['finished'] & (model_data['time_ms'] > 0)].copy()
        distances_finishers = finishers['distance_miles'].values
        gender_indices_finishers = finishers['gender'].map(gender_to_idx).values
        course_indices_finishers = finishers['name'].map(course_to_idx).values  # Create mappings
        observed_times_finishers = finishers['time_ms'].values / 60000
        log_distances_finishers = np.log(distances_finishers)
        log_distance_ratio_finishers = np.log(distances_finishers / REFERENCE_DISTANCE)
        distances_results = model_data['distance_miles'].values  # Prepare finisher data
        course_indices_results = model_data['name'].map(course_to_idx).values
        did_finish = (~model_data['finished']).values.astype(int)
        unique_course_distances = np.array([distances_finishers[course_indices_finishers == _i].mean() if (course_indices_finishers == _i).any() else distances_results[course_indices_results == _i].mean() if (course_indices_results == _i).any() else REFERENCE_DISTANCE for _i in range(len(unique_courses))])
        unique_log_distance_ratios = np.log(unique_course_distances / REFERENCE_DISTANCE)
        coords = {'course': unique_courses, 'gender': unique_genders, 'course_effect_type': ['finish_time_total', 'dnf_total'], 'finishers': np.arange(len(observed_times_finishers)), 'obs_results': np.arange(len(did_finish))}  # Convert to minutes
        model = pm.Model(coords=coords)
        with model:
            is_fixed_mode = fixed_params is not None
            if is_fixed_mode and 'pace_marathon' in fixed_params:  # Prepare DNF data (all results)
                pace_marathon = pm.Data('pace_marathon', fixed_params['pace_marathon'], dims='gender')  # CHANGED: Flip the coding so did_finish represents DNF (1=DNF, 0=finished)
            else:  # This matches the DNF semantics of logit_p_dnf
                pace_marathon = pm.Normal('pace_marathon', mu=[mu_pace_m, mu_pace_f], sigma=sigma_pace_marathon, dims='gender')
            if is_fixed_mode and 'pace_distance_effect' in fixed_params:
                pace_distance_effect = pm.Data('pace_distance_effect', fixed_params['pace_distance_effect'], dims='gender')  # 1=DNF, 0=finished
            else:
                pace_distance_effect = pm.Normal('pace_distance_effect', mu=[beta_m, beta_f], sigma=sigma_pace_distance_effect, dims='gender')  # Compute log distance ratios for each unique course
            if is_fixed_mode:
                finish_time_noise = np.array([sigma_finish_time_noise, sigma_finish_time_noise])
            else:
                finish_time_noise = pm.HalfNormal('finish_time_noise', sigma=sigma_finish_time_noise, dims='gender')
            if is_fixed_mode and 'dnf_rate_marathon' in fixed_params:
                dnf_rate_marathon = pm.Data('dnf_rate_marathon', fixed_params['dnf_rate_marathon'])
            else:
                dnf_rate_marathon = pm.Normal('dnf_rate_marathon', mu=mu_logit_dnf, sigma=sigma_dnf_rate_marathon)
            if is_fixed_mode and 'dnf_distance_multiplier' in fixed_params:
                dnf_distance_multiplier = pm.Data('dnf_distance_multiplier', fixed_params['dnf_distance_multiplier'])
            else:  # Define coordinates for PyMC
                dnf_distance_multiplier = pm.Normal('dnf_distance_multiplier', mu=beta_dist_dnf, sigma=sigma_dnf_distance_multiplier)
            pace_marathon_avg = (pace_marathon[0] + pace_marathon[1]) / 2
            pace_distance_effect_avg = (pace_distance_effect[0] + pace_distance_effect[1]) / 2
            course_distance_finish_baseline = pace_marathon_avg + pace_distance_effect_avg * unique_log_distance_ratios
            course_distance_dnf_baseline = dnf_rate_marathon + dnf_distance_multiplier * unique_log_distance_ratios
            if is_fixed_mode and 'chol' in fixed_params:
                chol = pt.as_tensor_variable(fixed_params['chol'])
            else:
                chol, corr, stds = pm.LKJCholeskyCov('course_chol', n=2, eta=2.0, sd_dist=pm.HalfNormal.dist(sigma=[sigma_course_prior, sigma_course_dnf_prior]), compute_corr=True)  # === BUILD MODEL ===
            course_effects_raw = pm.Normal('course_effects_raw', mu=0, sigma=1, dims=('course', 'course_effect_type'))
            course_effects_centered = pm.Deterministic('course_effects_centered', pm.math.dot(course_effects_raw, chol), dims=('course', 'course_effect_type'))
            course_total_finish_effect = course_distance_finish_baseline + course_effects_centered[:, 0]
            course_total_dnf_effect = course_distance_dnf_baseline + course_effects_centered[:, 1]
            gender_deviation = pace_marathon[gender_indices_finishers] - pace_marathon_avg + (pace_distance_effect[gender_indices_finishers] - pace_distance_effect_avg) * log_distance_ratio_finishers
            expected_log_pace = course_total_finish_effect[course_indices_finishers] + gender_deviation  # --- HYPERPARAMETERS ---
            expected_log_time = expected_log_pace + log_distances_finishers  # Either estimated via priors or fixed at provided values
            finish_likelihood = pm.LogNormal('finisher_times', mu=expected_log_time, sigma=finish_time_noise[gender_indices_finishers], observed=observed_times_finishers, dims='finishers')
            logit_p_dnf = course_total_dnf_effect[course_indices_results]
            pm.Bernoulli('did_finish_obs', p=pm.math.invlogit(logit_p_dnf), observed=did_finish, dims='obs_results')
        return model  # Finish time noise - always fixed at sigma_finish_time_noise constant  # --- DISTANCE BASELINES ---  # --- CORRELATED COURSE EFFECTS ---  # Use pre-computed Cholesky matrix  # Estimate correlation structure via LKJCholeskyCov  # Non-centered parameterization with standard normal raw effects  # Transform to correlated effects with proper scaling  # Add distance baselines to get total effects (in log-space)  # --- LIKELIHOOD ---  # FINISH TIME MODEL (LogNormal)  # Compute gender deviation from population average (in log-space)  # OPTIMIZATION: Stay in log-space throughout  # LogNormal likelihood  # DNF MODEL (additive effects on logit scale)  # logit_p_dnf represents logit(P(DNF))  # CHANGED: Removed negation - now directly modeling P(DNF)  # Bernoulli likelihood: did_finish is now coded as 1=DNF, 0=finished

    return (define_model,)


@app.cell
def _(define_model, model_data, model_dir, pm):
    # Build MCMC model with all hyperparameters estimated via priors
    model = define_model(model_data, fixed_params=None)
    _graph = pm.model_to_graphviz(model)
    # Display model structure
    _graph.graph_attr['size'] = '10,12'
    _graph.render(model_dir / 'model_graph', format='png', cleanup=True)
    _graph
    return (model,)


@app.cell
def _(np):
    # Helper function to transform log-space course effects to interpretable multipliers
    # This is needed because we stay in log-space during sampling but need linear-space for visualization
    def compute_course_multipliers_from_trace(trace):
        """
        Transform log-space course effects to interpretable pace multipliers.
        Used for visualizations after sampling.

        Args:
            trace: ArviZ InferenceData object with posterior samples

        Returns:
            Array of course multipliers (shape: chains × draws × n_courses)
        """
        # Extract centered effects (in log-space for finish times)
        course_effects_centered = trace.posterior['course_effects_centered']

        # For finish times: multiplier = exp(effect)
        # NOTE: Since we removed max(effect, 0) clipping from the model, 
        # multipliers can now be < 1.0 (courses that make you faster)
        # Use .sel() to select the finish_time dimension by name
        finish_effects = course_effects_centered.sel(course_effect_type='finish_time_total')
        course_multipliers_finish = np.exp(finish_effects.values)

        return course_multipliers_finish

    return


@app.cell
def _(model, pm):
    # Sample from prior
    with model:
        prior_pred = pm.sample_prior_predictive(samples=1000, random_seed=42)
    return (prior_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior Visualization
    """)
    return


@app.cell
def _(
    beta_dist_dnf,
    beta_f,
    beta_m,
    mu_logit_dnf,
    mu_pace_f,
    mu_pace_m,
    np,
    plt,
    prior_pred,
    sigma_course_dnf_prior,
    sigma_course_prior,
    sns,
    unique_genders,
):
    # Prior Distribution Visualizations
    # Extract prior samples from prior_pred and plot as KDEs to show theoretical distributions
    n_genders = len(unique_genders)  # Compute locally
    pace_marathon_prior = prior_pred.prior['pace_marathon'].values
    pace_distance_effect_prior = prior_pred.prior['pace_distance_effect'].values  # Shape: (chains, draws, n_genders)
    finish_time_noise_prior = prior_pred.prior['finish_time_noise'].values
    # NOTE: finish_time_noise is now estimated (HalfNormal prior)
    dnf_rate_marathon_prior = prior_pred.prior['dnf_rate_marathon'].values  # Shape: (chains, draws, n_genders)
    dnf_distance_multiplier_prior = prior_pred.prior['dnf_distance_multiplier'].values  # Scalar
    _chol_prior_xr = prior_pred.prior['course_chol']  # Scalar
    _chol_prior = _chol_prior_xr.values
    # Extract hyperparameters from chol matrix (removed as Deterministic for performance)
    if _chol_prior.ndim == 3:  # xarray with named dimensions
        course_effect_sds_prior = np.stack([_chol_prior[:, 0, 0], np.sqrt(_chol_prior[:, 1, 0] ** 2 + _chol_prior[:, 1, 1] ** 2)], axis=-1)  # Convert to numpy array
    # Cholesky: L = [[a, 0], [b, c]], Stds: [a, sqrt(b^2 + c^2)], Corr: b / (a * sqrt(b^2 + c^2))
        course_correlation_prior = _chol_prior[:, 1, 0] / (_chol_prior[:, 0, 0] * np.sqrt(_chol_prior[:, 1, 0] ** 2 + _chol_prior[:, 1, 1] ** 2))
    # Check dimensions and handle accordingly
    else:
        course_effect_sds_prior = np.stack([_chol_prior[:, :, 0, 0], np.sqrt(_chol_prior[:, :, 1, 0] ** 2 + _chol_prior[:, :, 1, 1] ** 2)], axis=-1)  # Shape is (chains*draws, 2, 2) - already flattened
        course_correlation_prior = _chol_prior[:, :, 1, 0] / (_chol_prior[:, :, 0, 0] * np.sqrt(_chol_prior[:, :, 1, 0] ** 2 + _chol_prior[:, :, 1, 1] ** 2))
    _pace_marathon_flat = pace_marathon_prior.reshape(-1, n_genders)  # Finish time SD  
    _pace_distance_effect_flat = pace_distance_effect_prior.reshape(-1, n_genders)  # DNF SD
    finish_time_noise_flat = finish_time_noise_prior.reshape(-1, n_genders)  # Shape: (samples, 2)
    dnf_rate_marathon_flat = dnf_rate_marathon_prior.flatten()
    dnf_distance_multiplier_flat = dnf_distance_multiplier_prior.flatten()
    _course_effect_sds_flat = course_effect_sds_prior.reshape(-1, 2)  # Shape: (samples,)
    course_correlation_flat = course_correlation_prior.flatten()
    pace_marathon_interpretable = np.exp(_pace_marathon_flat)  # Shape is (chains, draws, 2, 2) - need to extract chains/draws
    pace_distance_effect_interpretable = (np.power(2, _pace_distance_effect_flat) - 1) * 100
    finish_time_noise_interpretable = finish_time_noise_flat * 100  # Finish time SD
    dnf_prob_marathon_pct = 1 / (1 + np.exp(-dnf_rate_marathon_flat)) * 100  # DNF SD
    dnf_distance_multiplier_interpretable = np.exp(dnf_distance_multiplier_flat * np.log(2))  # Shape: (chains, draws, 2)
    course_finish_time_std_interpretable = (np.exp(_course_effect_sds_flat[:, 0]) - 1) * 100
    course_dnf_std_interpretable = np.exp(_course_effect_sds_flat[:, 1])
    _fig, _axes = plt.subplots(3, 3, figsize=(15, 12))  # Scalar correlation
    _axes = _axes.flatten()
    # Flatten to (samples,) for scalar parameters or (samples, dim) for vector parameters
    _ax = _axes[0]
    sns.kdeplot(data=pace_marathon_interpretable[:, 0], ax=_ax, color='steelblue', linewidth=2.5, fill=False)
    empirical_pace_m = np.exp(mu_pace_m)
    _ax.axvline(x=empirical_pace_m, color='steelblue', linestyle='--', linewidth=2, alpha=0.8, label='Empirical prior')
    _ax.set_xlabel('Pace (min/mile)', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')  # (samples, 2): [finish, dnf]
    _ax.set_title('pace_marathon (Male)\nAverage marathon pace', fontsize=12, fontweight='bold')  # Scalar correlation parameter
    _ax.legend()
    # Convert to interpretable units
    # pace_marathon: exp(log pace) = pace in min/mile
    _ax.grid(alpha=0.3)
    _ax = _axes[1]
    # pace_distance_effect: convert to % slowdown per 2x distance increase
    sns.kdeplot(data=pace_marathon_interpretable[:, 1], ax=_ax, color='#FF69B4', linewidth=2.5, fill=False)
    empirical_pace_f = np.exp(mu_pace_f)
    # finish_time_noise: convert from log-scale sigma to coefficient of variation (%)
    _ax.axvline(x=empirical_pace_f, color='#FF69B4', linestyle='--', linewidth=2, alpha=0.8, label='Empirical prior')
    _ax.set_xlabel('Pace (min/mile)', fontsize=11, fontweight='bold')
    # DNF parameters: convert from logit scale to probability/percentage for marathon distance
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    _ax.set_title('pace_marathon (Female)\nAverage marathon pace', fontsize=12, fontweight='bold')
    # dnf_distance_multiplier: convert to odds ratio per 2x distance increase
    _ax.legend()
    _ax.grid(alpha=0.3)
    # course_effect_sds: separate finish time and DNF standard deviations
    # Finish time: convert from log SD to typical % variation
    _ax = _axes[2]
    # DNF: convert from logit SD to odds ratio range
    sns.kdeplot(data=pace_distance_effect_interpretable[:, 0], ax=_ax, color='steelblue', linewidth=2.5, fill=False)
    empirical_dist_effect_m = (2 ** beta_m - 1) * 100
    # Create figure with 3 rows x 3 columns (9 subplots)
    _ax.axvline(x=empirical_dist_effect_m, color='steelblue', linestyle='--', linewidth=2, alpha=0.8, label='Empirical prior')
    _ax.set_xlabel('% slower per 2x distance', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    # 1. pace_marathon - Male (vector)
    _ax.set_title('pace_distance_effect (Male)\nSlowdown rate with distance', fontsize=12, fontweight='bold')
    _ax.legend()
    _ax.grid(alpha=0.3)
    _ax = _axes[3]
    # Add empirical prior reference (marathon pace for males)
    sns.kdeplot(data=pace_distance_effect_interpretable[:, 1], ax=_ax, color='#FF69B4', linewidth=2.5, fill=False)
    empirical_dist_effect_f = (2 ** beta_f - 1) * 100
    _ax.axvline(x=empirical_dist_effect_f, color='#FF69B4', linestyle='--', linewidth=2, alpha=0.8, label='Empirical prior')
    _ax.set_xlabel('% slower per 2x distance', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    _ax.set_title('pace_distance_effect (Female)\nSlowdown rate with distance', fontsize=12, fontweight='bold')
    _ax.legend()
    _ax.grid(alpha=0.3)
    _ax = _axes[4]
    sns.kdeplot(data=dnf_prob_marathon_pct, ax=_ax, color='#8B0000', linewidth=2.5, fill=False)
    # 2. pace_marathon - Female (vector)
    empirical_dnf_pct = 1 / (1 + np.exp(-mu_logit_dnf)) * 100
    _ax.axvline(x=empirical_dnf_pct, color='#8B0000', linestyle='--', linewidth=2, alpha=0.8, label='Empirical prior')
    _ax.set_xlabel('DNF rate (%)', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    # Add empirical prior reference (marathon pace for females)
    _ax.set_title('dnf_rate_marathon\nBaseline DNF rate at marathon', fontsize=12, fontweight='bold')
    _ax.legend()
    _ax.grid(alpha=0.3)
    _ax = _axes[5]
    sns.kdeplot(data=dnf_distance_multiplier_interpretable, ax=_ax, color='#8B0000', linewidth=2.5, fill=False)
    empirical_dnf_multiplier = np.exp(beta_dist_dnf * np.log(2))
    _ax.axvline(x=empirical_dnf_multiplier, color='#8B0000', linestyle='--', linewidth=2, alpha=0.8, label='Empirical prior')
    _ax.axvline(x=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='No effect')
    _ax.set_xlabel('DNF odds ratio per 2x distance', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    # 3. pace_distance_effect - Male (vector) - as % slowdown per 2x distance increase
    _ax.set_title('dnf_distance_multiplier\nDNF rate increase with distance', fontsize=12, fontweight='bold')
    _ax.legend()
    _ax.grid(alpha=0.3)
    _ax = _axes[6]
    sns.kdeplot(data=course_finish_time_std_interpretable, ax=_ax, color='#4a4a4a', linewidth=2.5, fill=False)
    empirical_finish_time_pct = (np.exp(sigma_course_prior) - 1) * 100
    _ax.axvline(x=empirical_finish_time_pct, color='#4a4a4a', linestyle='--', linewidth=2, alpha=0.8, label='Empirical prior')
    _ax.axvline(x=0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='No variation')
    _ax.set_xlabel('Time variation (±%)', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    _ax.set_title('course_effect_sds (finish time)\nTypical course difficulty variation', fontsize=12, fontweight='bold')
    _ax.legend()
    _ax.grid(alpha=0.3)
    _ax = _axes[7]
    # 4. pace_distance_effect - Female (vector) - as % slowdown per 2x distance increase
    sns.kdeplot(data=course_dnf_std_interpretable, ax=_ax, color='#8B0000', linewidth=2.5, fill=False)
    empirical_dnf_odds_ratio = np.exp(sigma_course_dnf_prior)
    _ax.axvline(x=empirical_dnf_odds_ratio, color='#8B0000', linestyle='--', linewidth=2, alpha=0.8, label='Empirical prior')
    _ax.axvline(x=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='No variation')
    _ax.set_xlabel('DNF odds ratio range', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    _ax.set_title('course_effect_sds (DNF)\nTypical course DNF rate variation', fontsize=12, fontweight='bold')
    _ax.legend()
    _ax.grid(alpha=0.3)
    _ax = _axes[8]
    sns.kdeplot(data=course_correlation_flat, ax=_ax, color='#8B0000', linewidth=2.5, fill=False)
    _ax.axvline(x=0.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='No correlation')
    _ax.set_xlabel('Correlation', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    # 5. dnf_rate_marathon (scalar) - as % DNF at marathon distance
    _ax.set_title('course_correlation\nCorrelation: finish difficulty ↔ DNF rate', fontsize=12, fontweight='bold')
    _ax.legend()
    _ax.grid(alpha=0.3)
    # Empirical prior
    _ax.set_xlim(-1, 1)
    plt.tight_layout()
    plt.suptitle('Hyperparameter Priors', fontsize=16, fontweight='bold', y=0.995)
    # 6. dnf_distance_multiplier (scalar) - as odds ratio per 2x distance increase
    # 6. course_effect_sds - Finish time (scalar) - as % variation
    # 7. course_effect_sds - DNF (scalar) - as odds ratio
    # 8. course_correlation (scalar) - THE KEY PARAMETER for selection bias quantification
    # No empirical prior for correlation (learned from LKJ)
    plt.show()
    return


@app.cell
def _(np, plt, prior_pred, sns):
    # Course Effect Priors
    # Extract BOTH raw (truncated) and centered (transformed) course effects
    _course_effects_raw_prior = prior_pred.prior['course_effects_raw']  # xarray with dims: (chain, draw, course, course_effect_type)
    course_effects_centered_prior = prior_pred.prior['course_effects_centered']  # xarray with dims: (chain, draw, course, course_effect_type)
    _n_samples = _course_effects_raw_prior.sizes['chain'] * _course_effects_raw_prior.sizes['draw']
    # Use .sel() to select by dimension name, then flatten
    course_pace_raw = _course_effects_raw_prior.sel(course_effect_type='finish_time_total').values.reshape(_n_samples, -1)
    course_dnf_raw = _course_effects_raw_prior.sel(course_effect_type='dnf_total').values.reshape(_n_samples, -1)
    course_pace_centered = course_effects_centered_prior.sel(course_effect_type='finish_time_total').values.reshape(_n_samples, -1)
    course_dnf_centered = course_effects_centered_prior.sel(course_effect_type='dnf_total').values.reshape(_n_samples, -1)
    course_pace_centered_pct = (np.exp(course_pace_centered) - 1) * 100
    course_dnf_centered_odds = np.exp(course_dnf_centered)
    # Transform centered effects to interpretable units
    _fig, _axes = plt.subplots(2, 3, figsize=(15, 8))  # Convert log scale to ±%
    _ax = _axes[0, 0]  # Convert log odds to odds ratio
    all_pace_raw = course_pace_raw.flatten()
    # Create figure with 2 rows x 3 columns
    sns.kdeplot(data=all_pace_raw, ax=_ax, color='#4a4a4a', linewidth=2.5, fill=False, label='Raw Pace Effects')
    _ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Mean')
    # ===== ROW 1: RAW EFFECTS (where truncation is applied) =====
    # Plot 1: Distribution of raw course pace effects (TRUNCATED)
    _ax.axvline(-2, color='darkgray', linestyle=':', linewidth=2, alpha=0.8, label='Truncation bounds')
    _ax.axvline(2.5, color='darkgray', linestyle=':', linewidth=2, alpha=0.8)
    _ax.set_xlabel('Standardized effect', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    _ax.set_title('course_effects_raw (pace)\nStandardized before transformation', fontsize=12, fontweight='bold')
    _ax.legend()
    _ax.grid(alpha=0.3)
    _ax = _axes[0, 1]
    all_dnf_raw = course_dnf_raw.flatten()
    sns.kdeplot(data=all_dnf_raw, ax=_ax, color='#4a4a4a', linewidth=2.5, fill=False, label='Raw DNF Effects')
    _ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Mean')
    _ax.axvline(-2, color='darkgray', linestyle=':', linewidth=2, alpha=0.8, label='Truncation bounds')
    _ax.axvline(2.5, color='darkgray', linestyle=':', linewidth=2, alpha=0.8)
    # Plot 2: Distribution of raw course DNF effects (TRUNCATED)
    _ax.set_xlabel('Standardized effect', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    _ax.set_title('course_effects_raw (DNF)\nStandardized before transformation', fontsize=12, fontweight='bold')
    _ax.legend()
    _ax.grid(alpha=0.3)
    _ax = _axes[0, 2]
    n_viz_samples = min(1000, _n_samples)
    sample_indices = np.random.choice(_n_samples, n_viz_samples, replace=False)
    for _i in sample_indices[:200]:
        _ax.scatter(course_pace_raw[_i, :], course_dnf_raw[_i, :], alpha=0.02, s=5, color='#4a4a4a')
    _ax.axhline(0, color='darkgray', linestyle='-', linewidth=0.5, alpha=0.5)
    _ax.axvline(0, color='darkgray', linestyle='-', linewidth=0.5, alpha=0.5)
    _ax.axhline(-2, color='darkgray', linestyle=':', linewidth=1, alpha=0.5)
    # Plot 3: Correlation in raw effects (before Cholesky transform)
    _ax.axhline(2.5, color='darkgray', linestyle=':', linewidth=1, alpha=0.5)
    _ax.axvline(-2, color='darkgray', linestyle=':', linewidth=1, alpha=0.5)
    _ax.axvline(2.5, color='darkgray', linestyle=':', linewidth=1, alpha=0.5)
    _ax.set_xlabel('Standardized pace effect', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Standardized DNF effect', fontsize=11, fontweight='bold')
    _ax.set_title('course_effects_raw correlation\nIndependent by construction', fontsize=12, fontweight='bold')
    _ax.grid(alpha=0.3)
    _ax = _axes[1, 0]
    all_pace_centered_pct = course_pace_centered_pct.flatten()
    pace_98p = np.percentile(all_pace_centered_pct, 98)
    all_pace_centered_pct_clipped = all_pace_centered_pct[all_pace_centered_pct <= pace_98p]
    sns.kdeplot(data=all_pace_centered_pct_clipped, ax=_ax, color='#4a4a4a', linewidth=2.5, fill=False, label='Centered Pace Effects')
    _ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='No effect')
    _ax.set_xlabel('Time difference from average course (±%)', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    _ax.set_title('course_effects_centered (pace)\nCourse-specific time adjustments', fontsize=12, fontweight='bold')
    _ax.legend()
    _ax.grid(alpha=0.3)
    _ax = _axes[1, 1]
    # ===== ROW 2: CENTERED EFFECTS (after scaling & correlation) =====
    # Plot 4: Distribution of centered course pace effects
    all_dnf_centered_odds = course_dnf_centered_odds.flatten()
    dnf_98p = np.percentile(all_dnf_centered_odds, 98)
    # Clip extreme values for visualization (keep 98% of data)
    all_dnf_centered_odds_clipped = all_dnf_centered_odds[all_dnf_centered_odds <= dnf_98p]
    sns.kdeplot(data=all_dnf_centered_odds_clipped, ax=_ax, color='#4a4a4a', linewidth=2.5, fill=False, label='Centered DNF Effects')
    _ax.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='No effect')
    _ax.set_xlabel('DNF odds ratio (vs. average course)', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    _ax.set_title('course_effects_centered (DNF)\nCourse-specific DNF rate adjustments', fontsize=12, fontweight='bold')
    _ax.legend()
    _ax.grid(alpha=0.3)
    _ax = _axes[1, 2]
    for _i in sample_indices[:200]:
        _ax.scatter(course_pace_centered_pct[_i, :], course_dnf_centered_odds[_i, :], alpha=0.02, s=5, color='#4a4a4a')
    # Plot 5: Distribution of centered course DNF effects
    _ax.axhline(1.0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    _ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    _ax.set_xlabel('Time difference (±%)', fontsize=11, fontweight='bold')
    _ax.set_ylabel('DNF odds ratio', fontsize=11, fontweight='bold')
    _ax.set_title('course_effects_centered correlation\nCorrelated course difficulty effects', fontsize=12, fontweight='bold')
    _ax.set_xlim(np.percentile(course_pace_centered_pct, 1), pace_98p)
    _ax.set_ylim(0, dnf_98p)
    _ax.grid(alpha=0.3)
    sample_corrs = []
    for _i in range(min(100, _n_samples)):
        corr = np.corrcoef(course_pace_centered[_i, :], course_dnf_centered[_i, :])[0, 1]
        sample_corrs.append(corr)
    _mean_corr = np.mean(sample_corrs)
    _ax.text(0.05, 0.95, f'Mean correlation: {_mean_corr:.3f}', transform=_ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # Plot 6: Correlation in centered effects (after Cholesky transform)
    plt.suptitle('Course Effect Priors: Raw (Truncated) vs Centered (Transformed)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    # Use percentile-based limits to match the distribution plots
    # Add correlation coefficient
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior Predictive Check
    """)
    return


@app.cell
def _(
    model_data_finishers,
    np,
    plt,
    prior_pred,
    results_2,
    standard_distances,
):
    def plot_ppc(fig, distances, full_data, model_data, predictions, xlabel, ylabel, title_template, colors=None, xlim=None):
        """
        Generic PPC plotting function for 2x4 grid of distance facets.

        Creates a 2x4 grid showing observed vs predicted distributions across distances.
        Works for both continuous (finish times) and binary (DNF rates) outcomes.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object to plot on
        distances : list of tuple
            Distance specifications as (value, label, tolerance) tuples
        full_data : dict
            Full observed data with keys determined by data type
        model_data : dict
            Model subset data with keys determined by data type
        predictions : dict
            Predicted values indexed by distance index
        xlabel : str
            Label for x-axis
        ylabel : str
            Label for y-axis
        title_template : callable
            Function taking (name, dist_value, n_full, n_model) -> title string
        colors : dict, optional
            Color mapping for distributions. Default: steelblue, green, orange
        xlim : tuple, optional
            Fixed x-axis limits as (min, max). If None, auto-determined per subplot
        """
        if colors is None:
            colors = {'Observed (Full)': 'steelblue', 'Observed (Model)': 'green', 'Predictions': 'orange'}
        _axes = fig.subplots(2, 4)
        _axes = _axes.flatten()
        for dist_idx, distance_tuple in enumerate(distances):
            dist_value, name, tolerance = distance_tuple
            _ax = _axes[dist_idx]
            _ax.set_facecolor('white')
            full_filtered = full_data['filter_fn'](full_data, dist_value, tolerance)
            model_filtered = model_data['filter_fn'](model_data, dist_value, tolerance)
            pred_filtered = predictions.get(dist_idx, np.array([]))
            if xlim is None:
                all_data = []
                if len(full_filtered) > 0:
                    all_data.append(full_filtered)
                if len(model_filtered) > 0:
                    all_data.append(model_filtered)
                if len(pred_filtered) > 0:
                    all_data.append(pred_filtered)
                if all_data:
                    all_combined = np.concatenate(all_data)
                    x_min = 0 if all_combined.min() >= 0 else all_combined.min()
                    x_max = np.percentile(all_combined, 99) * 1.05
                    x_range = (x_min, x_max)
                else:
                    x_range = None
            else:
                x_range = xlim
            if x_range is not None:
                bins = np.linspace(x_range[0], x_range[1], 51)
            else:
                bins = 50
            distributions = {'Observed (Full)': full_filtered, 'Observed (Model)': model_filtered, 'Predictions': pred_filtered}
            for dist_name, dist_data in distributions.items():
                if len(dist_data) == 0:
                    continue
                _color = colors[dist_name]
                _ax.hist(dist_data, bins=bins, histtype='step', linewidth=2, density=True, color=_color, linestyle='-', alpha=0.8, range=x_range)
            if x_range is not None:
                _ax.set_xlim(x_range)
            _ax.set_xlabel(xlabel, fontsize=10)
            _ax.set_ylabel(ylabel, fontsize=10)
            _ax.set_title(title_template(name, dist_value, len(full_filtered), len(model_filtered)), fontsize=10)
            _ax.tick_params(labelsize=9)
            _ax.grid(True, color='lightgray', alpha=0.7, linewidth=0.5)
        for _idx in range(len(distances), len(_axes)):
            _axes[_idx].axis('off')

    def plot_finish_time_ppc(inference_data, results, model_data, distances, title):
        """  # Create bins
        Plot finish time predictive check by gender.

        Creates separate figures for Male and Female showing 8 distance facets
        with observed and predicted finish time distributions.

        Parameters  # Plot distributions
        ----------
        inference_data : arviz.InferenceData
            InferenceData with prior_predictive or posterior_predictive
        results : pd.DataFrame
            Full results dataset
        model_data : pd.DataFrame
            Model subset data
        distances : list of tuple
            Distance specifications
        title : str
            Base title for figures

        Returns
        -------  # Set limits
        dict
            Dictionary mapping gender labels to Figure objects
        """
        model_data_finishers = model_data[model_data['finished'] & (model_data['time_ms'] > 0)].copy()
        unique_genders = np.array(['M', 'F'])
        gender_to_idx = {'M': 0, 'F': 1}
        if hasattr(inference_data, 'posterior_predictive') and 'finisher_times' in inference_data.posterior_predictive:
            finish_times = inference_data.posterior_predictive['finisher_times'].values
        elif hasattr(inference_data, 'prior_predictive') and 'finisher_times' in inference_data.prior_predictive:
            finish_times = inference_data.prior_predictive['finisher_times'].values
        else:
            raise ValueError("No 'finisher_times' found in prior_predictive or posterior_predictive")
        finish_times_flat = finish_times.reshape(-1, finish_times.shape[-1])
        predictions_by_gender = {}
        for _gender_idx, _gender in enumerate(unique_genders):
            predictions_by_gender[_gender_idx] = {}
            for dist_idx, distance_tuple in enumerate(distances):
                dist_value, name, tolerance = distance_tuple
                model_mask = (model_data_finishers['gender'].map(gender_to_idx) == _gender_idx) & (np.abs(model_data_finishers['distance_miles'] - dist_value) <= tolerance)
                finisher_indices = np.where(model_mask)[0]
                if len(finisher_indices) > 0:
                    predictions_by_gender[_gender_idx][dist_idx] = finish_times_flat[:, finisher_indices].flatten()
                else:
                    predictions_by_gender[_gender_idx][dist_idx] = np.array([])
        finishers_full = results[results['finished']].copy()
        full_times = finishers_full['time_ms'].values / 60000
        full_gender_indices = finishers_full['gender'].map(gender_to_idx).values
        full_distances = finishers_full['distance_miles'].values
        finishers_model = model_data_finishers.copy()
        model_times = finishers_model['time_ms'].values / 60000
        model_gender_indices = finishers_model['gender'].map(gender_to_idx).values
        model_distances = finishers_model['distance_miles'].values
        gender_display_names = {'M': 'Male', 'F': 'Female'}
        figures = {}
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='steelblue', edgecolor='steelblue', label='Observed (Full)'), Patch(facecolor='green', edgecolor='green', label='Observed (Model)'), Patch(facecolor='orange', edgecolor='orange', label='Predictions')]
        for _gender_idx, _gender in enumerate(unique_genders):
            _fig = plt.figure(figsize=(16, 8))
            _fig.patch.set_facecolor('white')
            full_data = {'times': full_times, 'gender_indices': full_gender_indices, 'distances': full_distances, 'gender_idx': _gender_idx, 'filter_fn': lambda data, dist_val, tol: data['times'][(data['gender_indices'] == data['gender_idx']) & (np.abs(data['distances'] - dist_val) <= tol)]}
            model_data_dict = {'times': model_times, 'gender_indices': model_gender_indices, 'distances': model_distances, 'gender_idx': _gender_idx, 'filter_fn': lambda data, dist_val, tol: data['times'][(data['gender_indices'] == data['gender_idx']) & (np.abs(data['distances'] - dist_val) <= tol)]}

            def title_fn(name, dist_val, n_full, n_model):
                model_pct = f' ({100 * n_model / n_full:.0f}% in model)' if n_full > 0 and n_model > 0 else ''
                return f'{name} ({dist_val:.1f}mi)\n{n_full:,} finishers{model_pct}'
            plot_ppc(_fig, distances, full_data, model_data_dict, predictions_by_gender[_gender_idx], xlabel='Time (min)', ylabel='Density', title_template=title_fn)
            gender_label = gender_display_names[_gender]
            _fig.suptitle(f'{title} - {gender_label}', fontsize=15, fontweight='bold', y=0.98)
            _fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=10, frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, 0.94))
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            figures[gender_label] = _fig
        return figures
    figures = plot_finish_time_ppc(inference_data=prior_pred, results=results_2, model_data=model_data_finishers, distances=standard_distances, title='Prior Predictive Check: Finish Times by Gender and Distance')
    for gender_label, _fig in figures.items():
        plt.show()
    return plot_finish_time_ppc, plot_ppc


@app.cell
def _(
    kcore_data,
    np,
    plot_ppc,
    plt,
    prior_pred,
    reference_distance,
    results_2,
    standard_distances,
):
    def plot_dnf_rate_ppc(inference_data, results, model_data, distances, title, reference_distance=26.2):
        """
        Plot predictive check for DNF rates using the generic plot_ppc function.

        Creates a single 2x4 grid showing overlaid histograms
        of observed and predicted DNF rate distributions.

        Parameters
        ----------
        inference_data : arviz.InferenceData
            InferenceData with prior or posterior samples
        results : pd.DataFrame
            Full results dataset with columns: finished, name, distance_miles
        model_data : pd.DataFrame
            Model subset with columns: finished, name, distance_miles
        distances : list of tuple
            Standard distances for binning (distance_miles, label, tolerance)
        title : str
            Title for the overall figure
        reference_distance : float
            Reference distance for log-space calculations (default: 26.2)

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        unique_courses = model_data['name'].unique()
        unique_course_distances = np.array([model_data[model_data['name'] == course]['distance_miles'].mean() for course in unique_courses])
        unique_log_distance_ratios = np.log(unique_course_distances / reference_distance)
        if hasattr(inference_data, 'posterior'):
            dnf_rate_marathon = inference_data.posterior['dnf_rate_marathon'].values.flatten()
            dnf_distance_multiplier = inference_data.posterior['dnf_distance_multiplier'].values.flatten()
            course_effects_centered = inference_data.posterior['course_effects_centered'].values
            course_effects_centered_flat = course_effects_centered.reshape(-1, course_effects_centered.shape[-2], course_effects_centered.shape[-1])
        else:
            dnf_rate_marathon = inference_data.prior['dnf_rate_marathon'].values.flatten()
            dnf_distance_multiplier = inference_data.prior['dnf_distance_multiplier'].values.flatten()
            dnf_rate_marathon_full = inference_data.prior['dnf_rate_marathon'].values
            dnf_distance_multiplier_full = inference_data.prior['dnf_distance_multiplier'].values
            course_effects_centered = inference_data.prior['course_effects_centered'].values
            course_distance_dnf_baseline = dnf_rate_marathon_full[:, :, np.newaxis] + dnf_distance_multiplier_full[:, :, np.newaxis] * unique_log_distance_ratios
            course_total_dnf_effect = course_distance_dnf_baseline + course_effects_centered[..., 1]
            course_effects_centered_flat = course_total_dnf_effect.reshape(-1, course_total_dnf_effect.shape[-1])
        course_name_to_idx = {name: _idx for _idx, name in enumerate(unique_courses)}
        model_course_names_for_pred = model_data['name'].values
        model_race_distances_for_pred = model_data['distance_miles'].values
        predictions = {}
        for dist_idx, distance_tuple in enumerate(distances):
            dist_value, name, tolerance = distance_tuple
            model_distance_mask = np.abs(model_race_distances_for_pred - dist_value) < tolerance
            if not model_distance_mask.any():
                predictions[dist_idx] = np.array([])
                continue
            courses_in_bin = model_course_names_for_pred[model_distance_mask]
            unique_courses_in_bin = np.unique(courses_in_bin)
            valid_course_indices = []
            for _course_name in unique_courses_in_bin:
                if _course_name in course_name_to_idx:
                    valid_course_indices.append(course_name_to_idx[_course_name])
            if len(valid_course_indices) > 0:
                _n_samples = min(1000, len(dnf_rate_marathon))
                random_draws = np.random.choice(len(dnf_rate_marathon), _n_samples, replace=False)
                stratified_courses = np.random.choice(valid_course_indices, size=_n_samples, replace=True)
                if hasattr(inference_data, 'posterior'):
                    log_distance_ratio = np.log(dist_value / reference_distance)
                    course_deviation_dnf = course_effects_centered_flat[random_draws, stratified_courses, 1]
                    logit_p_dnf = dnf_rate_marathon[random_draws] + dnf_distance_multiplier[random_draws] * log_distance_ratio + course_deviation_dnf
                else:
                    logit_p_dnf = course_effects_centered_flat[random_draws, stratified_courses]
                p_dnf = 1 / (1 + np.exp(-logit_p_dnf))
                predictions[dist_idx] = p_dnf
            else:
                _n_samples = min(1000, len(dnf_rate_marathon))
                log_distance_ratio = np.log(dist_value / reference_distance)
                logit_p_dnf = dnf_rate_marathon[:_n_samples] + dnf_distance_multiplier[:_n_samples] * log_distance_ratio
                p_dnf = 1 / (1 + np.exp(-logit_p_dnf))
                predictions[dist_idx] = p_dnf
        full_did_finish = results['finished'].values
        full_course_names = results['name'].values
        full_race_distances = results['distance_miles'].values
        model_did_finish = model_data['finished'].values
        model_course_names = model_data['name'].values
        model_race_distances = model_data['distance_miles'].values

        def compute_course_dnf_rates(data, dist_val, tol):
            """Compute per-course DNF rates for a distance bin."""
            distance_mask = np.abs(data['distances'] - dist_val) < tol
            if not distance_mask.any():
                return np.array([])
            courses_in_bin = data['course_names'][distance_mask]
            did_finish_in_bin = data['did_finish'][distance_mask]
            course_dnf_rates = []
            for _course_name in np.unique(courses_in_bin):
                course_mask = courses_in_bin == _course_name
                course_finishes = did_finish_in_bin[course_mask]
                if len(course_finishes) >= 5:
                    dnf_rate = 1 - course_finishes.mean()
                    course_dnf_rates.append(dnf_rate)
            return np.array(course_dnf_rates)
        full_data = {'did_finish': full_did_finish, 'course_names': full_course_names, 'distances': full_race_distances, 'filter_fn': compute_course_dnf_rates}
        model_data_dict = {'did_finish': model_did_finish, 'course_names': model_course_names, 'distances': model_race_distances, 'filter_fn': compute_course_dnf_rates}

        def dnf_title_template(name, dist_value, n_courses_full, n_courses_model):
            distance_mask = np.abs(full_race_distances - dist_value) < 0.5
            n_obs = distance_mask.sum()
            model_pct = f'{100 * n_courses_model / n_courses_full:.0f}%' if n_courses_full > 0 else '0%'
            return f'{name} ({dist_value:.1f}mi)\n{n_courses_model}/{n_courses_full} courses ({model_pct} in model), {n_obs:,} results'
        _fig, _axes = plt.subplots(2, 4, figsize=(16, 8))
        _fig.patch.set_facecolor('white')
        plot_ppc(fig=_fig, distances=distances, full_data=full_data, model_data=model_data_dict, predictions=predictions, xlabel='DNF Rate', ylabel='Density', title_template=dnf_title_template, xlim=(0, 1))
        handles = [plt.matplotlib.patches.Patch(color='steelblue', label='Observed (Full)'), plt.matplotlib.patches.Patch(color='green', label='Observed (Model)'), plt.matplotlib.patches.Patch(color='orange', label='Predictions')]
        _fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3, fontsize=10, frameon=False)
        _fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        return _fig
    _fig = plot_dnf_rate_ppc(inference_data=prior_pred, results=results_2, model_data=kcore_data, distances=standard_distances, title='Prior Predictive Check: DNF Rates by Distance', reference_distance=reference_distance)
    plt.show()
    return (plot_dnf_rate_ppc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Geometry Check
    """)
    return


@app.cell
def _(np, plt, prior_pred):
    # PRE-SAMPLING GEOMETRY DIAGNOSTIC
    # Comprehensive check for sampling issues using ArviZ
    # Detects: funnels, high correlations, poor prior specification
    _chol_prior_xr = prior_pred.prior['course_chol']
    # ============================================================================
    # GLOBAL FUNNEL CHECK: Hyperprior vs Aggregate Course Variance
    _chol_prior = _chol_prior_xr.values
    # Check if the scale parameters (hyperpriors) correlate with the variance of ALL raw course parameters
    # This is the RIGHT way to check for funnel geometry in hierarchical models
    # This model has 2D correlated course effects: finish_time and dnf
    if _chol_prior.ndim == 3:
    # Extract chol matrix and compute course_effect_sds from it (not stored as Deterministic for performance)
        _chol_flat = _chol_prior.reshape(-1, 3)
    else:
        _chol_flat = _chol_prior
    # The chol matrix is in packed form (chain, draw, 3) where the 3 elements are the lower triangle
    # For 2x2 matrix: [[a, 0], [b, c]] stored as [a, b, c]
    # We need to reshape to (chains*draws, 3) and then extract elements
    _a = _chol_flat[:, 0]
    # Flatten chain and draw dimensions
    _b = _chol_flat[:, 1]
    _c = _chol_flat[:, 2]  # Shape: (chains, draws, 3) -> (chains*draws, 3)
    _course_effect_sds_flat = np.stack([_a, np.sqrt(_b ** 2 + _c ** 2)], axis=-1)
    course_effects_raw_prior_xr = prior_pred.prior['course_effects_raw']
    _course_effects_raw_prior = course_effects_raw_prior_xr.values  # Already flat
    if _course_effects_raw_prior.ndim == 4:
        course_effects_raw_var = _course_effects_raw_prior.std(axis=2).reshape(-1, 2)
    # Extract Cholesky elements: [[a, 0], [b, c]]
    elif _course_effects_raw_prior.ndim == 3:  # (0,0) element
        course_effects_raw_var = _course_effects_raw_prior.std(axis=1)  # (1,0) element  
    else:  # (1,1) element
        raise ValueError(f'Unexpected course_effects_raw_prior shape: {_course_effects_raw_prior.shape}')
    # Compute standard deviations from Cholesky
    # SD for effect 0 (finish time): a
    # SD for effect 1 (DNF): sqrt(b^2 + c^2)
    _n_samples = min(len(_course_effect_sds_flat), len(course_effects_raw_var))
    _course_effect_sds_flat = _course_effect_sds_flat[:_n_samples]  # Finish time SD
    course_effects_raw_var = course_effects_raw_var[:_n_samples]  # DNF SD
    print(f'Plotting {_n_samples} samples')  # Shape: (samples, 2)
    _fig, _axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
    # Get raw course effects
    effect_names = ['Finish Time', 'DNF']  # xarray
    colors = ['#4a4a4a', '#8B0000']  # Convert to numpy
    for _idx, (effect_name, _color) in enumerate(zip(effect_names, colors)):
    # Compute std of raw parameters across ALL courses (for each MCMC draw and effect type)
        _ax = _axes[_idx]
        _ax.set_facecolor('white')  # Shape: (chains, draws, n_courses, 2) -> compute std across courses -> (chains*draws, 2)
        _ax.scatter(_course_effect_sds_flat[:, _idx], course_effects_raw_var[:, _idx], alpha=0.3, s=10, color=_color)
        _ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Expected variance (σ=1 for non-centered)')
        _ax.set_xlabel(f'course_effect_sds ({effect_name})', fontsize=12, fontweight='bold')  # Shape: (samples, n_courses, 2) -> compute std across courses -> (samples, 2)
        _ax.set_ylabel(f'Std(course_effects_raw) across all courses', fontsize=12, fontweight='bold')
        _ax.set_title(f'Global Funnel Check: {effect_name} Effects\nHyperprior vs. Aggregate Course Variance', fontsize=13, fontweight='bold')
        _ax.legend(fontsize=11)
        _ax.grid(True, color='lightgray', alpha=0.7, linewidth=0.5)
    # Ensure both arrays have the same number of samples
    plt.tight_layout()
    # Create 2 subplots: one for finish time effects, one for DNF effects
    plt.show()  # Add reference line
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
    """)
    return


@app.cell
def _(
    NUTS_SAMPLER,
    PLATFORM_CONFIG,
    alpha,
    az,
    beta,
    build_mcmc_summary_string,
    jax,
    model,
    model_data,
    n_courses,
    notify_mcmc_complete,
    notify_mcmc_error,
    notify_mcmc_start,
    os,
    pm,
    subset_dir,
    time,
):
    TUNE = 500
    DRAWS = 1000
    TARGET_ACCEPT = 0.95
    cache_file = f'{subset_dir}/tune{TUNE}_draws{DRAWS}_accept{TARGET_ACCEPT}.nc'
    hyperparam_vars = ['pace_marathon', 'pace_distance_effect', 'finish_time_noise', 'dnf_rate_marathon', 'dnf_distance_multiplier']
    if os.path.exists(cache_file):
        print(f'Loading cached trace from {cache_file}')
        trace = az.from_netcdf(cache_file)
    else:
        start_message = f"Configuration: tune={TUNE}, draws={DRAWS}, target_accept={TARGET_ACCEPT}\nPlatform: {PLATFORM_CONFIG}\nK-core subset: alpha={alpha}, beta={beta}\nData: total results={len(model_data):,}, courses={n_courses:,}, runners={model_data['participant_id'].nunique():,}"
        print(start_message)
        notify_mcmc_start(model_name='Model 2 (Observed DNFs)', message=start_message)
        _start_time = time.time()
        try:
            with model:
                trace = pm.sample(draws=DRAWS, tune=TUNE, chains=4, cores=4, target_accept=TARGET_ACCEPT, random_seed=37, return_inferencedata=True, idata_kwargs={'log_likelihood': False}, nuts_sampler=NUTS_SAMPLER)
            jax.clear_caches()
            _elapsed_time = time.time() - _start_time
            summary_text = build_mcmc_summary_string(trace, hyperparam_vars)
            print('\n' + '=' * 60)
            print('MCMC SAMPLING COMPLETE')
            print('=' * 60)
            print(summary_text)
            print('=' * 60 + '\n')
            notify_mcmc_complete(model_name='Model 2 (Observed DNFs)', elapsed_time=_elapsed_time, summary_text=summary_text)
            print(f'Saving trace to {cache_file}')
            trace.to_netcdf(cache_file)
        except Exception as e:
            notify_mcmc_error(model_name='Model 2 (Observed DNFs)', error_msg=str(e))
            raise
    az.summary(trace)
    return DRAWS, TARGET_ACCEPT, TUNE, hyperparam_vars, trace


@app.cell
def _(np, trace):
    # Reconstruct hyperparameters that were removed for performance
    # Do this immediately after loading trace so hyperparam_vars will work
    if 'course_effect_sds' not in trace.posterior or 'course_correlation' not in trace.posterior:
    # The course_chol from LKJCholeskyCov is often not fully saved to trace
    # We can reconstruct hyperparameters from the course_effects_centered instead
        print('Reconstructing course_effect_sds and course_correlation...')
        effects = trace.posterior['course_effects_centered'].values
        n_chains, n_draws, n_courses_1, _ = effects.shape
        print(f'  course_effects_centered shape: {effects.shape}')
        course_effect_sds = np.zeros((n_chains, n_draws, 2))  # Reconstruct from the centered effects using empirical covariance
        course_correlation = np.zeros((n_chains, n_draws))  # (chains, draws, courses, 2)
        for chain in range(n_chains):
            for draw in range(n_draws):
                sample_effects = effects[chain, draw, :, :]
                cov = np.cov(sample_effects.T)
                course_effect_sds[chain, draw, 0] = np.sqrt(cov[0, 0])  # For each chain/draw, compute the empirical covariance
                course_effect_sds[chain, draw, 1] = np.sqrt(cov[1, 1])
                course_correlation[chain, draw] = cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]))
        import xarray as xr
        trace.posterior['course_effect_sds'] = xr.DataArray(course_effect_sds, dims=['chain', 'draw', 'course_effect_type'], coords={'chain': trace.posterior.chain, 'draw': trace.posterior.draw, 'course_effect_type': ['finish_time_total', 'dnf_total']})
        trace.posterior['course_correlation'] = xr.DataArray(course_correlation, dims=['chain', 'draw'], coords={'chain': trace.posterior.chain, 'draw': trace.posterior.draw})
        print(f'✅ Reconstructed course_effect_sds and course_correlation')  # Get effects for this sample: (n_courses, 2)
        print(f'   course_effect_sds shape: {course_effect_sds.shape}')
        print(f'   course_correlation shape: {course_correlation.shape}')
    else:  # Compute covariance
        print('✓ course_effect_sds and course_correlation already in trace')  # (2, 2)  # Extract stds and correlation  # Add to trace
    return (n_courses_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Traceplot
    """)
    return


@app.cell
def _(
    DRAWS,
    TARGET_ACCEPT,
    TUNE,
    az,
    hyperparam_vars,
    np,
    plt,
    subset_dir,
    trace,
):
    # traceplot for hyperparameters
    _axes = az.plot_trace(trace, var_names=hyperparam_vars, compact=True, figsize=(12, 10))
    _fig = _axes.ravel()[0].figure
    _fig.patch.set_facecolor('white')
    for _ax in _axes.ravel():
        _ax.set_facecolor('white')
        if _ax.get_xlabel() or _ax.get_ylabel():  # Only add grid to plots with axes
            _ax.grid(True, color='lightgray', alpha=0.7, linewidth=0.5)
    plt.suptitle('MCMC Traces: Course-Level Difficulty Model Hyperparameters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    traceplot_file = f'{subset_dir}/tune{TUNE}_draws{DRAWS}_accept{TARGET_ACCEPT}_traceplot.png'
    # save traceplot
    plt.savefig(traceplot_file, dpi=300, bbox_inches='tight')
    print(f'Saved traceplot to {traceplot_file}')
    plt.show()
    course_coords = list(trace.posterior.coords['course'].values)
    sample_course_ids = np.random.choice(course_coords, size=min(20, len(course_coords)), replace=False)
    _axes = az.plot_trace(trace, var_names=['course_effects_centered'], coords={'course': sample_course_ids}, compact=True, figsize=(12, 20))
    # traceplot for sample of course difficulty parameters
    _fig = _axes.ravel()[0].figure
    _fig.patch.set_facecolor('white')
    for _ax in _axes.ravel():
        _ax.set_facecolor('white')
        if _ax.get_xlabel() or _ax.get_ylabel():
            _ax.grid(True, color='lightgray', alpha=0.7, linewidth=0.5)  # Use centered effects (finish + DNF)
    plt.suptitle('MCMC Traces: Sample Course Effects (Centered)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()  # Only add grid to plots with axes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Posterior Diagnostics
    """)
    return


@app.cell
def _(
    DRAWS,
    TARGET_ACCEPT,
    TUNE,
    hyperparam_vars,
    plot_posterior_diagnostics,
    subset_dir,
    trace,
):
    plot_posterior_diagnostics(trace, hyperparam_vars, subset_dir, TUNE, DRAWS, TARGET_ACCEPT)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Posterior Predictive Check
    """)
    return


@app.cell
def _(model, pm, trace):
    with model:
        post_pred = pm.sample_posterior_predictive(
            trace, 
            random_seed=37,
            progressbar=True
        )

    # add posterior predictive to trace
    trace.extend(post_pred)
    return


@app.cell
def _(
    model_data_finishers,
    plot_finish_time_ppc,
    plt,
    results_2,
    standard_distances,
    trace,
):
    _fig = plot_finish_time_ppc(inference_data=trace, results=results_2, model_data=model_data_finishers, distances=standard_distances, title='Posterior Predictive Check: Finish Times by Gender and Distance')
    plt.show()
    return


@app.cell
def _(
    REFERENCE_DISTANCE,
    kcore_data,
    plot_dnf_rate_ppc,
    plt,
    results_2,
    standard_distances,
    trace,
):
    _fig = plot_dnf_rate_ppc(inference_data=trace, results=results_2, model_data=kcore_data, distances=standard_distances, title='Posterior Predictive Check: DNF Rates by Distance', reference_distance=REFERENCE_DISTANCE)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parameter Visualizations
    """)
    return


@app.cell
def _(np, plt, reference_distance, standard_distances, trace, unique_genders):
    # Pace-Distance Curves by Gender
    # Extract posterior samples for pace_marathon and pace_distance_effect
    pace_marathon_samples = trace.posterior['pace_marathon'].values  # Shape: (chains, draws, n_genders)
    pace_distance_effect_samples = trace.posterior['pace_distance_effect'].values  # Shape: (chains, draws, n_genders)
    distances = np.linspace(5, 100, 200)
    # Create distance range for visualization
    _fig, _ax = plt.subplots(1, 1, figsize=(14, 8), facecolor='white')  # 5 to 100 miles
    gender_colors = {'M': 'steelblue', 'F': 'pink'}
    # Plot both genders on the same figure
    gender_labels = {'M': 'Male', 'F': 'Female'}
    all_pace_medians = {}
    # Define colors for each gender
    for _gender_idx, _gender in enumerate(unique_genders):
        _pace_marathon_flat = pace_marathon_samples[:, :, _gender_idx].flatten()
        _pace_distance_effect_flat = pace_distance_effect_samples[:, :, _gender_idx].flatten()
    # First pass: calculate all median curves to determine y-limits
        log_pace_curves = []
        for pace_base, dist_effect in zip(_pace_marathon_flat, _pace_distance_effect_flat):
            log_pace = pace_base + dist_effect * np.log(distances / reference_distance)
            log_pace_curves.append(np.exp(log_pace))
        log_pace_curves = np.array(log_pace_curves)
        all_pace_medians[_gender] = {'median': np.percentile(log_pace_curves, 50, axis=0), 'p25': np.percentile(log_pace_curves, 25, axis=0), 'p75': np.percentile(log_pace_curves, 75, axis=0)}
    all_median_values = np.concatenate([all_pace_medians[g]['median'] for g in unique_genders])
    y_min = all_median_values.min() - 0.5
    y_max = all_median_values.max() + 1.0
    _ax.set_ylim(y_min, y_max)
    for _gender_idx, _gender in enumerate(unique_genders):
        pace_data = all_pace_medians[_gender]
        pace_median = pace_data['median']
        pace_25 = pace_data['p25']
        pace_75 = pace_data['p75']
        _color = gender_colors[_gender]
        _ax.fill_between(distances, pace_25, pace_75, alpha=0.25, color=_color)
    # Determine y-limits from all medians
        _ax.plot(distances, pace_median, color=_color, linewidth=2.5)
        for dist_value, name, _ in standard_distances:
            _idx = np.argmin(np.abs(distances - dist_value))
            pace = pace_median[_idx]
            _ax.plot(dist_value, pace, 'o', color=_color, markersize=6, zorder=5, alpha=0.8)
    # Second pass: plot curves and markers
            y_offset = 0.15 if _gender == 'M' else -0.4
            label_text = f'{pace:.1f}'
            _ax.text(dist_value, pace + y_offset, label_text, fontsize=7, ha='center', va='bottom' if _gender == 'M' else 'top', fontweight='bold', color=_color)
    y_label_position = y_max - 0.2
    for dist_value, name, _ in standard_distances:
        _ax.axvline(x=dist_value, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        _ax.text(dist_value, y_label_position, name, fontsize=8, ha='center', va='top', rotation=0, alpha=0.6)
    _ax.set_xlabel('Distance (miles)', fontsize=13, fontweight='bold')
    _ax.set_ylabel('Pace (min/mile)', fontsize=13, fontweight='bold')  # Fill between for uncertainty
    _ax.set_title('Learned Pace-Distance Curves by Gender (Population Average)', fontsize=15, fontweight='bold', pad=20)
    _ax.set_facecolor('white')
    _ax.grid(True, color='lightgray', alpha=0.7, linewidth=0.5, linestyle='--')  # Median curve
    _ax.set_xlim(5, 100)
    plt.tight_layout()
    # Add distance labels at the top
    plt.show()  # Add markers and annotations for ALL standard distances  # Find closest index in distances array  # Plot marker  # Label with pace value (offset vertically by gender to avoid overlap)  # Fixed offset from top
    return


@app.cell
def _(np, plt, reference_distance, standard_distances, trace):
    # Plot learned DNF rate as a function of course distance
    # This shows the distance baseline (without course-specific effects)
    dnf_rate_marathon_samples = trace.posterior['dnf_rate_marathon'].values.flatten()
    # Get the learned distance baseline DNF effects from the posterior
    # This is dnf_rate_marathon + dnf_distance_multiplier * log(distance / marathon_distance)
    dnf_distance_multiplier_samples = trace.posterior['dnf_distance_multiplier'].values.flatten()
    _distance_range_miles = np.linspace(10, 100, 200)
    log_distance_ratios = np.log(_distance_range_miles / reference_distance)
    # Create a range of distances to plot
    _dnf_logit_curves = dnf_rate_marathon_samples[:, np.newaxis] + dnf_distance_multiplier_samples[:, np.newaxis] * log_distance_ratios

    def logit_to_prob(logit):
    # Calculate DNF logit for each distance (across all posterior samples)
        return 1 / (1 + np.exp(-logit))
    _dnf_prob_curves = logit_to_prob(_dnf_logit_curves)
    dnf_prob_median = np.median(_dnf_prob_curves, axis=0)
    # Convert from logit to probability
    dnf_prob_q05 = np.percentile(_dnf_prob_curves, 5, axis=0)
    dnf_prob_q95 = np.percentile(_dnf_prob_curves, 95, axis=0)
    _fig, _ax = plt.subplots(figsize=(12, 7))
    _ax.plot(_distance_range_miles, dnf_prob_median * 100, linewidth=3, color='#000000', label='Median learned DNF rate')
    _ax.fill_between(_distance_range_miles, dnf_prob_q05 * 100, dnf_prob_q95 * 100, alpha=0.3, color='#000000', label='90% Credible Interval')
    # Calculate median and credible intervals
    for _dist_miles, _dist_label, _ in standard_distances:
        if 10 <= _dist_miles <= 100:
            _ax.axvline(_dist_miles, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            log_ratio = np.log(_dist_miles / reference_distance)
    # Create the plot
            dnf_logit_at_dist = dnf_rate_marathon_samples + dnf_distance_multiplier_samples * log_ratio
            dnf_prob_at_dist = logit_to_prob(dnf_logit_at_dist)
    # Plot the median curve
            median_dnf = np.median(dnf_prob_at_dist) * 100
            _ax.text(_dist_miles, median_dnf + 2, _dist_label, ha='center', va='bottom', fontsize=9, color='gray')
    _ax.set_xlabel('Race Distance (miles)', fontsize=12, fontweight='bold')
    # Plot the credible interval
    _ax.set_ylabel('DNF Rate (%)', fontsize=12, fontweight='bold')
    _ax.set_title('Learned DNF Rate as a Function of Distance\n(Distance Baseline Effect Only)', fontsize=14, fontweight='bold')
    _ax.grid(alpha=0.3, linestyle=':')
    # Add vertical lines for standard distances
    _ax.legend(loc='upper left', fontsize=11)
    _ax.set_xlim(10, 100)
    _ax.set_ylim(0, max(dnf_prob_q95 * 100) + 5)
    plt.tight_layout()  # Calculate DNF rate at this distance
    plt.show()
    return


@app.cell
def _(az, np, plt, trace):
    # NEW DIAGNOSTIC: Correlation Posterior
    # Visualize the learned correlation between finish time and DNF course effects
    course_correlation_posterior = trace.posterior['course_correlation'].values.flatten()
    hdi_94 = az.hdi(trace, var_names=['course_correlation'], hdi_prob=0.94)
    # Compute HDI
    hdi_lower = float(hdi_94['course_correlation'].values[0])
    hdi_upper = float(hdi_94['course_correlation'].values[1])
    _mean_corr = course_correlation_posterior.mean()
    median_corr = np.median(course_correlation_posterior)
    # Compute summary statistics
    _fig, _ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='white')
    _ax.hist(course_correlation_posterior, bins=60, alpha=0.7, color='#666666', density=True, label='Posterior')
    _ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Independence', alpha=0.7)
    # Create figure
    _ax.axvline(x=_mean_corr, color='black', linestyle='-', linewidth=2.5, label=f'Mean ({_mean_corr:+.3f})')
    _ax.axvline(x=median_corr, color='darkblue', linestyle='-', linewidth=2, label=f'Median ({median_corr:+.3f})', alpha=0.8)
    # Plot posterior distribution
    _ax.axvspan(hdi_lower, hdi_upper, alpha=0.2, color='green', label=f'94% HDI [{hdi_lower:+.3f}, {hdi_upper:+.3f}]')
    _ax.set_xlabel('Correlation Coefficient', fontsize=13, fontweight='bold')
    # Add vertical lines for key statistics
    _ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    _ax.set_title('Posterior: Correlation between Course Finish Time and DNF Effects\n(Measures Selection Bias Strength)', fontsize=14, fontweight='bold')
    _ax.legend(loc='upper left', fontsize=11)
    _ax.set_facecolor('white')
    # Shade HDI region
    _ax.grid(True, color='lightgray', alpha=0.7, linewidth=0.5, axis='y')
    _ax.set_xlim(-1, 1)
    # Formatting
    if _mean_corr < -0.2:
        interp = '✅ NEGATIVE CORRELATION DETECTED\nHarder courses → More DNFs (Selection Bias)'
        box_color = 'lightcoral'
    elif _mean_corr > 0.2:
        interp = '⚠️ POSITIVE CORRELATION\nHarder courses → Fewer DNFs? (Unexpected)'
        box_color = 'lightyellow'
    else:
        interp = '⚪ WEAK/NO CORRELATION\nFinish time and DNF effects are independent'
        box_color = 'lightgray'
    # Add interpretation text box
    # NOTE: After fix, NEGATIVE correlation = harder courses → MORE DNFs (selection bias)
    # This is because logit_p_dnf is now correctly negated in the model
    _ax.text(0.98, 0.97, interp, transform=_ax.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute Derived Quantities

    Since we removed `pm.Deterministic` for performance (15-25% speedup), we need to manually compute the total effects from fundamental parameters + data arrays.
    """)
    return


@app.cell
def _(np, trace, unique_log_distance_ratios):
    # Reconstruct course total effects from fundamental parameters + data arrays
    # These were removed as Deterministic variables for MCMC performance optimization
    pace_marathon_post = trace.posterior['pace_marathon']
    # Extract fundamental parameters from posterior
    pace_distance_effect_post = trace.posterior['pace_distance_effect']  # Shape: (chains, draws, n_genders)
    dnf_rate_marathon_post = trace.posterior['dnf_rate_marathon']  # Shape: (chains, draws, n_genders)
    dnf_distance_multiplier_post = trace.posterior['dnf_distance_multiplier']  # Shape: (chains, draws)
    course_effects_centered_post = trace.posterior['course_effects_centered']  # Shape: (chains, draws)
    pace_marathon_avg_post = pace_marathon_post.mean(dim='gender')  # Shape: (chains, draws, n_courses, 2)
    pace_distance_effect_avg_post = pace_distance_effect_post.mean(dim='gender')
    # Compute population averages (these were intermediate variables in model, not saved to trace)
    course_distance_finish_baseline_post = pace_marathon_avg_post.values[..., np.newaxis] + pace_distance_effect_avg_post.values[..., np.newaxis] * unique_log_distance_ratios  # Shape: (chains, draws)
    course_distance_dnf_baseline_post = dnf_rate_marathon_post.values[..., np.newaxis] + dnf_distance_multiplier_post.values[..., np.newaxis] * unique_log_distance_ratios  # Shape: (chains, draws)
    course_total_finish_effect_post = course_distance_finish_baseline_post + course_effects_centered_post.sel(course_effect_type='finish_time_total').values
    # Reconstruct course_distance_finish_baseline from fundamentals + data
    # Need to expand dims for broadcasting: (chains, draws) + (chains, draws) * (n_courses,)
    # Result shape: (chains, draws, n_courses)
    course_total_dnf_effect_post = course_distance_dnf_baseline_post + course_effects_centered_post.sel(course_effect_type='dnf_total').values
    finish_flat = course_total_finish_effect_post.reshape(-1, course_total_finish_effect_post.shape[-1])
    dnf_flat = course_total_dnf_effect_post.reshape(-1, course_total_dnf_effect_post.shape[-1])
    mcmc_finish_effect_medians = np.median(finish_flat, axis=0)
    mcmc_finish_effect_q05 = np.percentile(finish_flat, 5, axis=0)
    # Reconstruct course_distance_dnf_baseline from fundamentals + data
    # Shape broadcasting: (chains, draws) + (chains, draws) * (n_courses,) -> (chains, draws, n_courses)
    mcmc_finish_effect_q25 = np.percentile(finish_flat, 25, axis=0)
    mcmc_finish_effect_q75 = np.percentile(finish_flat, 75, axis=0)
    mcmc_finish_effect_q95 = np.percentile(finish_flat, 95, axis=0)
    mcmc_dnf_effect_medians = np.median(dnf_flat, axis=0)
    mcmc_dnf_effect_q05 = np.percentile(dnf_flat, 5, axis=0)
    # Compute total finish effect: baseline + centered_effect[0] (no clipping)
    mcmc_dnf_effect_q25 = np.percentile(dnf_flat, 25, axis=0)
    mcmc_dnf_effect_q75 = np.percentile(dnf_flat, 75, axis=0)
    mcmc_dnf_effect_q95 = np.percentile(dnf_flat, 95, axis=0)
    dnf_rate_marathon_samples_1 = trace.posterior['dnf_rate_marathon'].values.flatten()
    dnf_distance_multiplier_samples_1 = trace.posterior['dnf_distance_multiplier'].values.flatten()
    correlation_samples = trace.posterior['course_correlation'].values.flatten()
    # Compute total DNF effect: baseline + centered_effect[1]
    print('✅ Reconstructed course total effects from fundamental parameters')
    print(f'   course_total_finish_effect: shape {course_total_finish_effect_post.shape}')
    print(f'   course_total_dnf_effect: shape {course_total_dnf_effect_post.shape}')
    print(f'   Extracted quantiles for {len(mcmc_finish_effect_medians):,} courses')
    # Extract quantiles for plotting (flattening chains and draws)
    # Reshape to (chains*draws, n_courses) for quantile computation
    # Also create flattened samples for DNF rate calculations and bias estimation
    print(f'   correlation_samples: {len(correlation_samples):,} samples')
    return (
        correlation_samples,
        course_effects_centered_post,
        course_total_dnf_effect_post,
        dnf_distance_multiplier_samples_1,
        dnf_rate_marathon_samples_1,
        mcmc_dnf_effect_medians,
        mcmc_dnf_effect_q05,
        mcmc_dnf_effect_q25,
        mcmc_dnf_effect_q75,
        mcmc_dnf_effect_q95,
        mcmc_finish_effect_medians,
        mcmc_finish_effect_q05,
        mcmc_finish_effect_q25,
        mcmc_finish_effect_q75,
        mcmc_finish_effect_q95,
    )


@app.cell
def _(
    correlation_samples,
    dnf_distance_multiplier_samples_1,
    dnf_rate_marathon_samples_1,
    np,
    plt,
    reference_distance,
    standard_distances,
):
    _distance_range_miles = np.linspace(10, 100, 200)
    log_ratios = np.log(_distance_range_miles / reference_distance)
    _dnf_logit_curves = dnf_rate_marathon_samples_1[:, np.newaxis] + dnf_distance_multiplier_samples_1[:, np.newaxis] * log_ratios
    _dnf_prob_curves = 1 / (1 + np.exp(-_dnf_logit_curves))
    bias_curves = _dnf_prob_curves * correlation_samples[:, np.newaxis] * 100
    bias_median = np.median(bias_curves, axis=0)
    bias_q05 = np.percentile(bias_curves, 5, axis=0)
    bias_q95 = np.percentile(bias_curves, 95, axis=0)
    median_correlation = np.median(correlation_samples)
    _fig, _ax = plt.subplots(figsize=(10, 6), facecolor='white')
    _ax.fill_between(_distance_range_miles, bias_q05, bias_q95, alpha=0.3, color='#4a4a4a', label='90% Credible Interval')
    _ax.plot(_distance_range_miles, bias_median, linewidth=2.5, color='#2a2a2a', label='Median bias')
    for _dist_miles, _dist_label, _ in standard_distances:
        if 10 <= _dist_miles <= 100:
            _ax.axvline(_dist_miles, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            y_pos = _ax.get_ylim()[1] * 0.95
            _ax.text(_dist_miles, y_pos, _dist_label, ha='center', va='top', fontsize=9, color='gray')
    _ax.set_xlabel('Race Distance (miles)', fontsize=12)
    _ax.set_ylabel('Finish Time Bias (%)', fontsize=12)
    _ax.set_title('Finish Time Bias Due to DNF Exclusion\n(Observed times faster than if all runners finished)', fontsize=14, fontweight='bold')
    _ax.legend(fontsize=11, loc='upper left')
    _ax.set_facecolor('white')
    _ax.grid(True, color='lightgray', alpha=0.7, linewidth=0.5)
    _ax.set_xlim(10, 100)
    plt.tight_layout()
    plt.show()
    print(f'Median correlation: {median_correlation:.3f}')
    print(f'Bias at 100mi: {bias_median[-1]:.2f}% (median), [{bias_q05[-1]:.2f}%, {bias_q95[-1]:.2f}%]')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MAP Inference

    Extend inference to the full dataset using learned hyperparameters from k-core MCMC.

    The k-core subsetting approach provides high-quality estimates of **population-level hyperparameters** (`pace_marathon`, `pace_distance_effect`, `course_finish_time_multiplier_std`, `finish_time_noise`, `dnf_rate_marathon`, `dnf_distance_multiplier`, `course_dnf_multiplier_std`) by focusing on densely-connected runners and courses. However, many courses appear only in the sparse closure (runners or courses with few observations).

    We use a **two-stage strategy**:
    1. **MCMC on k-core**: Learn hyperparameters from dense subgraph (already completed above)
    2. **MAP on full dataset**: Fix hyperparameters and estimate only course-level parameters for all courses

    This approach is efficient because:
    - Hyperparameters generalize from dense core to sparse closure
    - Course multipliers are entity-specific adjustments (don't need full posterior)
    - MAP optimization is fast (seconds/minutes vs. hours for full MCMC)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Preparation
    """)
    return


@app.cell
def _(np, sigma_finish_time_noise, trace):
    # Extract posterior medians from k-core MCMC trace
    # These hyperparameters will be FIXED in the MAP model
    pace_marathon_fixed = trace.posterior['pace_marathon'].median(dim=['chain', 'draw']).values
    pace_distance_effect_fixed = trace.posterior['pace_distance_effect'].median(dim=['chain', 'draw']).values
    _chol_flat = trace.posterior['course_chol'].median(dim=['chain', 'draw']).values
    # Extract course effect standard deviations and correlation from chol matrix
    # (These were removed as pm.Deterministic for MCMC performance)
    _a, _b, _c = _chol_flat
    # Cholesky is stored as flattened lower triangular: [L[0,0], L[1,0], L[1,1]]
    # Reconstruct the 2x2 matrix: L = [[a, 0], [b, c]]
    chol_fixed = np.array([[_a, 0], [_b, _c]])
    course_effect_sds_fixed = np.array([chol_fixed[0, 0], np.sqrt(chol_fixed[1, 0] ** 2 + chol_fixed[1, 1] ** 2)])
    course_correlation_fixed = float(chol_fixed[1, 0] / (chol_fixed[0, 0] * course_effect_sds_fixed[1]))
    # Covariance: C = L @ L.T = [[a^2, ab], [ab, b^2 + c^2]]
    # Stds: [a, sqrt(b^2 + c^2)], Correlation: b / sqrt(a^2 * (b^2 + c^2))
    dnf_rate_marathon_fixed = float(trace.posterior['dnf_rate_marathon'].median(dim=['chain', 'draw']).values)
    dnf_distance_multiplier_fixed = float(trace.posterior['dnf_distance_multiplier'].median(dim=['chain', 'draw']).values)  # Finish time SD
    finish_time_noise_fixed = np.array([sigma_finish_time_noise, sigma_finish_time_noise])  # DNF SD
    print('Fixed hyperparameters from k-core MCMC:')
    print('=' * 60)
    print(f'Finish Time Model:')
    print(f'  pace_marathon (M, F):                    [{pace_marathon_fixed[0]:.4f}, {pace_marathon_fixed[1]:.4f}]')
    print(f'  pace_distance_effect (M, F):             [{pace_distance_effect_fixed[0]:.4f}, {pace_distance_effect_fixed[1]:.4f}]')
    # DNF model hyperparameters
    print(f'  finish_time_noise (M, F):                [{finish_time_noise_fixed[0]:.4f}, {finish_time_noise_fixed[1]:.4f}] (fixed)')
    print(f'\nDNF Model:')
    print(f'  dnf_rate_marathon:                       {dnf_rate_marathon_fixed:.4f}')
    print(f'  dnf_distance_multiplier:                 {dnf_distance_multiplier_fixed:.4f}')
    print(f'\nCorrelated Course Effects:')
    print(f'  course_effect_sds [finish, dnf]:         [{course_effect_sds_fixed[0]:.4f}, {course_effect_sds_fixed[1]:.4f}]')
    print(f'  course_correlation:                      {course_correlation_fixed:.4f}')
    # Finish time noise is fixed in the model (not estimated)
    print('=' * 60)
    return (
        course_correlation_fixed,
        course_effect_sds_fixed,
        dnf_distance_multiplier_fixed,
        dnf_rate_marathon_fixed,
        finish_time_noise_fixed,
        pace_distance_effect_fixed,
        pace_marathon_fixed,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Specification

    Build a PyMC model with the same structure as the MCMC model, but with hyperparameters fixed to their posterior medians. Only course-level parameters will be estimated.
    """)
    return


@app.cell
def _(REFERENCE_DISTANCE, filter_races_with_dnfs, n_courses_1, np, results_2):
    results_full = filter_races_with_dnfs(results_2)
    finishers_full = results_full[results_full['finished'] & (results_full['time_ms'] > 0)].copy()
    unique_courses_full = results_full['name'].unique()
    course_to_idx_full = {course: _idx for _idx, course in enumerate(unique_courses_full)}
    gender_to_idx_full = {'M': 0, 'F': 1}
    course_indices_finishers_full = finishers_full['name'].map(course_to_idx_full).values
    gender_indices_finishers_full = finishers_full['gender'].map(gender_to_idx_full).values
    distances_finishers_full = finishers_full['distance_miles'].values
    observed_times_finishers_full = finishers_full['time_ms'].values / 60000
    course_indices_results_full = results_full['name'].map(course_to_idx_full).values
    distances_results_full = results_full['distance_miles'].values
    did_finish_full = results_full['finished'].values.astype(int)
    n_courses_full = len(unique_courses_full)
    unique_course_distances_full = np.array([distances_finishers_full[course_indices_finishers_full == _i].mean() if (course_indices_finishers_full == _i).any() else distances_results_full[course_indices_results_full == _i].mean() if (course_indices_results_full == _i).any() else REFERENCE_DISTANCE for _i in range(n_courses_full)])
    unique_log_distance_ratios_full = np.log(unique_course_distances_full / REFERENCE_DISTANCE)
    print(f'Full dataset prepared for MAP estimation:')
    print(f'  Total courses: {n_courses_full:,}')
    print(f'  K-core courses (for comparison): {n_courses_1:,}')
    print(f'  Total results (finishers + DNFs): {len(did_finish_full):,}')
    print(f'  Finisher observations: {len(observed_times_finishers_full):,}')
    return (
        did_finish_full,
        n_courses_full,
        observed_times_finishers_full,
        results_full,
        unique_courses_full,
        unique_log_distance_ratios_full,
    )


@app.cell
def _(
    course_correlation_fixed,
    course_effect_sds_fixed,
    define_model,
    dnf_distance_multiplier_fixed,
    dnf_rate_marathon_fixed,
    np,
    pace_distance_effect_fixed,
    pace_marathon_fixed,
    pm,
    results_full,
):
    corr_matrix = np.array([[1.0, course_correlation_fixed], [course_correlation_fixed, 1.0]])
    std_diag = np.diag(course_effect_sds_fixed)
    cov_matrix = std_diag @ corr_matrix @ std_diag
    chol_fixed_1 = np.linalg.cholesky(cov_matrix)
    model_map = define_model(results_full, fixed_params={'pace_marathon': pace_marathon_fixed, 'pace_distance_effect': pace_distance_effect_fixed, 'dnf_rate_marathon': dnf_rate_marathon_fixed, 'dnf_distance_multiplier': dnf_distance_multiplier_fixed, 'chol': chol_fixed_1})
    _graph = pm.model_to_graphviz(model_map)
    _graph
    return chol_fixed_1, model_map


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Estimation
    """)
    return


@app.cell
def _(
    az,
    course_correlation_fixed,
    course_effect_sds_fixed,
    did_finish_full,
    dnf_distance_multiplier_fixed,
    dnf_rate_marathon_fixed,
    model_map,
    n_courses_full,
    np,
    observed_times_finishers_full,
    os,
    pace_distance_effect_fixed,
    pace_marathon_fixed,
    pm,
    subset_dir,
    time,
    trace,
    unique_courses_full,
    unique_genders,
    unique_log_distance_ratios_full,
):
    map_cache_file = f'{subset_dir}/map_full_dataset.nc'
    if os.path.exists(map_cache_file):
        print(f'✅ Loading cached MAP estimate from {map_cache_file}')
        map_trace = az.from_netcdf(map_cache_file)
        print(f"   Loaded {len(map_trace.posterior.coords['course']):,} course effects")
    else:
        print('🚀 Running MAP estimation on full dataset...')
        print(f'   Fixed hyperparameters from k-core MCMC (8 values)')
        print(f'   Estimating course_effects_raw for {n_courses_full:,} courses (2 effects each)')
        print(f'   Total free parameters: {n_courses_full * 2:,}')
        print('\n📊 Initializing course effects from k-core MCMC...')
        mcmc_courses = np.array(trace.posterior.coords['course'].values)
        mcmc_course_effects_raw = trace.posterior['course_effects_raw'].median(dim=['chain', 'draw']).values
        _init_course_effects_raw = np.zeros((n_courses_full, 2))
        n_initialized = 0
        for _i, _course_name in enumerate(unique_courses_full):
            if _course_name in mcmc_courses:
                _mcmc_idx = np.where(mcmc_courses == _course_name)[0][0]
                _init_course_effects_raw[_i, :] = mcmc_course_effects_raw[_mcmc_idx, :]
                n_initialized = n_initialized + 1
        print(f'   Initialized {n_initialized:,} k-core courses from MCMC posteriors')
        print(f'   Initialized {n_courses_full - n_initialized:,} non-k-core courses to prior mean (0)')
        print()
        _start_time = time.time()
        with model_map:
            map_estimate = pm.find_MAP(method='L-BFGS-B', start={'course_effects_raw': _init_course_effects_raw}, progressbar=True)
        _elapsed_time = time.time() - _start_time
        print(f'\n✅ MAP estimation completed in {_elapsed_time:.2f} seconds')
        print(f'   ({_elapsed_time / 60:.1f} minutes)')
        _pace_marathon_avg_map = pace_marathon_fixed.mean()
        _pace_distance_effect_avg_map = pace_distance_effect_fixed.mean()
        _course_distance_finish_baseline_map = _pace_marathon_avg_map + _pace_distance_effect_avg_map * unique_log_distance_ratios_full
        course_distance_dnf_baseline_map = dnf_rate_marathon_fixed + dnf_distance_multiplier_fixed * unique_log_distance_ratios_full
        course_total_finish_effect_map = _course_distance_finish_baseline_map + map_estimate['course_effects_centered'][:, 0]
        course_total_dnf_effect_map = course_distance_dnf_baseline_map + map_estimate['course_effects_centered'][:, 1]
        map_trace = az.from_dict(posterior={'course_effects_raw': map_estimate['course_effects_raw'][np.newaxis, np.newaxis, :, :], 'course_effects_centered': map_estimate['course_effects_centered'][np.newaxis, np.newaxis, :, :], 'course_total_finish_effect': course_total_finish_effect_map[np.newaxis, np.newaxis, :], 'course_total_dnf_effect': course_total_dnf_effect_map[np.newaxis, np.newaxis, :], 'pace_marathon': pace_marathon_fixed[np.newaxis, np.newaxis, :], 'pace_distance_effect': pace_distance_effect_fixed[np.newaxis, np.newaxis, :], 'dnf_rate_marathon': np.array([[dnf_rate_marathon_fixed]]), 'dnf_distance_multiplier': np.array([[dnf_distance_multiplier_fixed]]), 'course_effect_sds': course_effect_sds_fixed[np.newaxis, np.newaxis, :], 'course_correlation': np.array([[course_correlation_fixed]])}, coords={'course': unique_courses_full, 'gender': unique_genders, 'course_effect_type': ['finish_time_total', 'dnf_total']}, dims={'course_effects_raw': ['course', 'course_effect_type'], 'course_effects_centered': ['course', 'course_effect_type'], 'course_total_finish_effect': ['course'], 'course_total_dnf_effect': ['course'], 'pace_marathon': ['gender'], 'pace_distance_effect': ['gender'], 'course_effect_sds': ['course_effect_type']})
        map_trace.to_netcdf(map_cache_file)
        print(f'💾 Saved MAP trace to {map_cache_file}')
    print()
    print('MAP estimation summary:')
    print(f'  Course effects estimated: {n_courses_full:,}')
    print(f'  Finisher observations: {len(observed_times_finishers_full):,}')
    print(f'  Total observations (incl. DNFs): {len(did_finish_full):,}')
    return (map_trace,)


@app.cell
def _(map_trace, unique_log_distance_ratios_full):
    pace_marathon_map = map_trace.posterior['pace_marathon'].values[0, 0, :]
    pace_distance_effect_map = map_trace.posterior['pace_distance_effect'].values[0, 0, :]
    dnf_rate_marathon_map = map_trace.posterior['dnf_rate_marathon'].values[0, 0]
    dnf_distance_multiplier_map = map_trace.posterior['dnf_distance_multiplier'].values[0, 0]
    course_effects_centered_map = map_trace.posterior['course_effects_centered'].values[0, 0, :, :]
    map_courses = map_trace.posterior.coords['course'].values
    _pace_marathon_avg_map = (pace_marathon_map[0] + pace_marathon_map[1]) / 2
    _pace_distance_effect_avg_map = (pace_distance_effect_map[0] + pace_distance_effect_map[1]) / 2
    _course_distance_finish_baseline_map = _pace_marathon_avg_map + _pace_distance_effect_avg_map * unique_log_distance_ratios_full
    course_distance_dnf_baseline_map_1 = dnf_rate_marathon_map + dnf_distance_multiplier_map * unique_log_distance_ratios_full
    map_finish_effect_estimates = _course_distance_finish_baseline_map + course_effects_centered_map[:, 0]
    map_dnf_effect_estimates = course_distance_dnf_baseline_map_1 + course_effects_centered_map[:, 1]
    return (
        course_distance_dnf_baseline_map_1,
        course_effects_centered_map,
        map_courses,
        map_dnf_effect_estimates,
        map_finish_effect_estimates,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### K-Core Validation

    Validate MAP estimates by comparing them to MCMC posteriors for courses in the k-core subset. This checks whether the MAP optimization produces reasonable point estimates that align with the full Bayesian posterior.
    """)
    return


@app.cell
def _(
    map_courses,
    map_dnf_effect_estimates,
    map_finish_effect_estimates,
    mcmc_dnf_effect_medians,
    mcmc_dnf_effect_q05,
    mcmc_dnf_effect_q25,
    mcmc_dnf_effect_q75,
    mcmc_dnf_effect_q95,
    mcmc_finish_effect_medians,
    mcmc_finish_effect_q05,
    mcmc_finish_effect_q25,
    mcmc_finish_effect_q75,
    mcmc_finish_effect_q95,
    np,
    trace,
):
    mcmc_courses_1 = trace.posterior.coords['course'].values
    n_mcmc_courses = len(mcmc_courses_1)
    map_course_to_idx = {course: _idx for _idx, course in enumerate(map_courses)}
    matched_map_indices = []
    matched_mcmc_indices = []
    for _mcmc_idx, _course_name in enumerate(mcmc_courses_1):
        if _course_name in map_course_to_idx:
            matched_map_indices.append(map_course_to_idx[_course_name])
            matched_mcmc_indices.append(_mcmc_idx)
    matched_map_indices = np.array(matched_map_indices)
    matched_mcmc_indices = np.array(matched_mcmc_indices)
    map_matched_finish = map_finish_effect_estimates[matched_map_indices]
    map_matched_dnf = map_dnf_effect_estimates[matched_map_indices]
    mcmc_matched_finish_medians = mcmc_finish_effect_medians[matched_mcmc_indices]
    mcmc_matched_finish_q05 = mcmc_finish_effect_q05[matched_mcmc_indices]
    mcmc_matched_finish_q25 = mcmc_finish_effect_q25[matched_mcmc_indices]
    mcmc_matched_finish_q75 = mcmc_finish_effect_q75[matched_mcmc_indices]
    mcmc_matched_finish_q95 = mcmc_finish_effect_q95[matched_mcmc_indices]
    mcmc_matched_dnf_medians = mcmc_dnf_effect_medians[matched_mcmc_indices]
    mcmc_matched_dnf_q05 = mcmc_dnf_effect_q05[matched_mcmc_indices]
    mcmc_matched_dnf_q25 = mcmc_dnf_effect_q25[matched_mcmc_indices]
    mcmc_matched_dnf_q75 = mcmc_dnf_effect_q75[matched_mcmc_indices]
    mcmc_matched_dnf_q95 = mcmc_dnf_effect_q95[matched_mcmc_indices]
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    finish_correlation = pearsonr(map_matched_finish, mcmc_matched_finish_medians)[0]
    finish_mae = mean_absolute_error(mcmc_matched_finish_medians, map_matched_finish)
    finish_rmse = np.sqrt(mean_squared_error(mcmc_matched_finish_medians, map_matched_finish))
    dnf_correlation = pearsonr(map_matched_dnf, mcmc_matched_dnf_medians)[0]
    dnf_mae = mean_absolute_error(mcmc_matched_dnf_medians, map_matched_dnf)
    dnf_rmse = np.sqrt(mean_squared_error(mcmc_matched_dnf_medians, map_matched_dnf))
    return (
        map_matched_dnf,
        map_matched_finish,
        matched_map_indices,
        mcmc_courses_1,
        mcmc_matched_dnf_medians,
        mcmc_matched_dnf_q05,
        mcmc_matched_dnf_q25,
        mcmc_matched_dnf_q75,
        mcmc_matched_dnf_q95,
        mcmc_matched_finish_medians,
        mcmc_matched_finish_q05,
        mcmc_matched_finish_q25,
        mcmc_matched_finish_q75,
        mcmc_matched_finish_q95,
    )


@app.cell
def _(
    map_matched_dnf,
    map_matched_finish,
    matched_map_indices,
    mcmc_matched_dnf_medians,
    mcmc_matched_dnf_q05,
    mcmc_matched_dnf_q25,
    mcmc_matched_dnf_q75,
    mcmc_matched_dnf_q95,
    mcmc_matched_finish_medians,
    mcmc_matched_finish_q05,
    mcmc_matched_finish_q25,
    mcmc_matched_finish_q75,
    mcmc_matched_finish_q95,
    plt,
):
    # K-Core Validation: Scatter plots comparing MAP vs MCMC for matched courses
    # Uses the matched indices already computed in previous cell
    _fig, (_ax_course, _ax_runner) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    # Create scatter plots with error bars
    for _i in range(len(matched_map_indices)):
        _ax_course.plot([map_matched_finish[_i], map_matched_finish[_i]], [mcmc_matched_finish_q05[_i], mcmc_matched_finish_q95[_i]], color='steelblue', alpha=0.2, linewidth=1)
    # Finish time effects scatter
        _ax_course.plot([map_matched_finish[_i], map_matched_finish[_i]], [mcmc_matched_finish_q25[_i], mcmc_matched_finish_q75[_i]], color='steelblue', alpha=0.4, linewidth=2)
    _ax_course.scatter(map_matched_finish, mcmc_matched_finish_medians, alpha=0.6, s=20, color='darkblue')  # 90% CI (lighter)
    lims_finish = [min(map_matched_finish.min(), mcmc_matched_finish_q05.min()), max(map_matched_finish.max(), mcmc_matched_finish_q95.max())]
    _ax_course.plot(lims_finish, lims_finish, 'r--', alpha=0.7, linewidth=2, label='Perfect Agreement')
    _ax_course.set_xlabel('MAP Estimate (log minutes)', fontsize=12, fontweight='bold')
    _ax_course.set_ylabel('MCMC Posterior Median (log minutes)', fontsize=12, fontweight='bold')  # 50% CI (darker)
    _ax_course.set_title('Finish Time Effects: MAP vs MCMC', fontsize=14, fontweight='bold')
    _ax_course.set_facecolor('white')
    _ax_course.grid(True, color='lightgray', alpha=0.7, linewidth=0.5)
    _ax_course.legend()
    for _i in range(len(matched_map_indices)):
        _ax_runner.plot([map_matched_dnf[_i], map_matched_dnf[_i]], [mcmc_matched_dnf_q05[_i], mcmc_matched_dnf_q95[_i]], color='steelblue', alpha=0.2, linewidth=1)
        _ax_runner.plot([map_matched_dnf[_i], map_matched_dnf[_i]], [mcmc_matched_dnf_q25[_i], mcmc_matched_dnf_q75[_i]], color='steelblue', alpha=0.4, linewidth=2)
    _ax_runner.scatter(map_matched_dnf, mcmc_matched_dnf_medians, alpha=0.6, s=20, color='darkblue')
    lims_dnf = [min(map_matched_dnf.min(), mcmc_matched_dnf_q05.min()), max(map_matched_dnf.max(), mcmc_matched_dnf_q95.max())]
    _ax_runner.plot(lims_dnf, lims_dnf, 'r--', alpha=0.7, linewidth=2, label='Perfect Agreement')
    _ax_runner.set_xlabel('MAP Estimate (log odds)', fontsize=12, fontweight='bold')
    _ax_runner.set_ylabel('MCMC Posterior Median (log odds)', fontsize=12, fontweight='bold')
    _ax_runner.set_title('DNF Effects: MAP vs MCMC', fontsize=14, fontweight='bold')
    _ax_runner.set_facecolor('white')
    _ax_runner.grid(True, color='lightgray', alpha=0.7, linewidth=0.5)
    # DNF effects scatter
    _ax_runner.legend()
    plt.tight_layout()  # 90% CI (lighter)
    plt.show()  # 50% CI (darker)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parameter Visualizations
    """)
    return


@app.cell
def _(map_trace, mcmc_courses_1, np, plt):
    def plot_course_forest(map_values, mcmc_medians, mcmc_q05, mcmc_q95, xlabel, title, reference_line=None, n_show=50, sort_ascending=True):
        """
        Create a forest plot showing top/bottom courses with MAP estimates and MCMC posteriors.

        Parameters
        ----------
        map_values : np.ndarray
            MAP estimates for all courses (shape: n_courses)
        mcmc_medians : list or None
            MCMC median values for each selected course (None entries for courses without MCMC)
        mcmc_q05 : list or None
            MCMC 5th percentile for each selected course
        mcmc_q95 : list or None
            MCMC 95th percentile for each selected course
        xlabel : str
            X-axis label
        title : str
            Plot title
        reference_line : float or None
            Value at which to draw a vertical reference line
        n_show : int
            Number of courses to show from top and bottom (default: 50)
        sort_ascending : bool
            If True, sort ascending (lowest first); if False, descending (highest first)

        Returns
        -------
        fig, ax : matplotlib figure and axes objects
        """
        from matplotlib.lines import Line2D
        all_course_names = list(map_trace.posterior.coords['course'].values)
        sorted_indices = np.argsort(map_values)
        if not sort_ascending:
            sorted_indices = sorted_indices[::-1]
        top_indices = sorted_indices[:n_show]
        bottom_indices = sorted_indices[-n_show:]
        selected_indices = np.concatenate([top_indices, bottom_indices])
        _fig, _ax = plt.subplots(1, 1, figsize=(10, 20), facecolor='white')
        _ax.set_facecolor('white')
        selected_map_values = map_values[selected_indices]
        selected_names = [all_course_names[_i] for _i in selected_indices]
        mcmc_set = set(mcmc_courses_1)
        selected_in_mcmc = [all_course_names[_i] in mcmc_set for _i in selected_indices]
        y_positions = np.arange(len(selected_map_values))
        selected_mcmc_medians = []
        selected_mcmc_q05_vals = []
        selected_mcmc_q95_vals = []
        for _course_name in selected_names:
            if _course_name in mcmc_courses_1:
                _idx = np.where(mcmc_courses_1 == _course_name)[0][0]
                selected_mcmc_medians.append(mcmc_medians[_idx] if mcmc_medians is not None else None)
                selected_mcmc_q05_vals.append(mcmc_q05[_idx] if mcmc_q05 is not None else None)
                selected_mcmc_q95_vals.append(mcmc_q95[_idx] if mcmc_q95 is not None else None)
            else:
                selected_mcmc_medians.append(None)
                selected_mcmc_q05_vals.append(None)
                selected_mcmc_q95_vals.append(None)
        for _i, (map_val, in_mcmc) in enumerate(zip(selected_map_values, selected_in_mcmc)):
            _ax.plot(map_val, y_positions[_i], 's', color='orange', markersize=6, alpha=0.8, zorder=5)
            if in_mcmc and selected_mcmc_medians[_i] is not None:
                mcmc_median = selected_mcmc_medians[_i]
                mcmc_q05_val = selected_mcmc_q05_vals[_i]
                mcmc_q95_val = selected_mcmc_q95_vals[_i]
                _ax.plot([mcmc_q05_val, mcmc_q95_val], [y_positions[_i], y_positions[_i]], color='steelblue', linewidth=1.5, alpha=0.5, zorder=3)
                _ax.plot(mcmc_median, y_positions[_i], 'o', color='steelblue', markersize=5, alpha=0.7, zorder=4)
        if reference_line is not None:
            _ax.axvline(x=reference_line, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Reference ({reference_line})')
        _ax.set_yticks(y_positions)
        _ax.set_yticklabels([name[:60] for name in selected_names], fontsize=8)
        _ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        _ax.set_title(title, fontsize=14, fontweight='bold')
        _ax.grid(axis='x', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
        legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=8, label='MAP Estimate'), Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='MCMC Median'), Line2D([0], [0], color='steelblue', linewidth=2, alpha=0.5, label='MCMC 90% CI')]
        if reference_line is not None:
            legend_elements.append(Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label=f'Reference ({reference_line})'))
        _ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        _ax.axhline(y=n_show - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.tight_layout()
        return (_fig, _ax)

    return (plot_course_forest,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot 2: Relative Finish Time Multiplier (Course-Specific Deviation)
    """)
    return


@app.cell
def _(
    course_effects_centered_map,
    course_effects_centered_post,
    np,
    plot_course_forest,
    plt,
):
    # Plot 2: Relative Finish Time Multiplier (course-specific deviation from distance baseline)
    # MAP values: exp(course_effects_centered[:, 0])
    map_values_plot2 = np.exp(course_effects_centered_map[:, 0])

    # MCMC posteriors: exp(course_effects_centered_post[:, :, :, 0])
    # Shape: (chains, draws, n_courses, 2) -> select dimension 0 for finish time
    mcmc_rel_finish = np.exp(course_effects_centered_post.values[:, :, :, 0])
    mcmc_rel_finish_flat = mcmc_rel_finish.reshape(-1, mcmc_rel_finish.shape[-1])

    # Compute medians and quantiles
    mcmc_rel_finish_medians = np.median(mcmc_rel_finish_flat, axis=0)
    mcmc_rel_finish_q05 = np.percentile(mcmc_rel_finish_flat, 5, axis=0)
    mcmc_rel_finish_q95 = np.percentile(mcmc_rel_finish_flat, 95, axis=0)

    # Reference at 1.0 (multiplicative identity - average course)
    plot_course_forest(
        map_values=map_values_plot2,
        mcmc_medians=mcmc_rel_finish_medians,
        mcmc_q05=mcmc_rel_finish_q05,
        mcmc_q95=mcmc_rel_finish_q95,
        xlabel='Pace multiplier (relative to average course at this distance)',
        title='Relative Finish Time Multiplier',
        reference_line=1.0,
        n_show=50,
        sort_ascending=True
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot 4: Relative DNF Probability (Course-Specific Deviation)
    """)
    return


@app.cell
def _(
    course_distance_dnf_baseline_map_1,
    course_effects_centered_map,
    course_total_dnf_effect_post,
    np,
    plot_course_forest,
    plt,
):
    # Plot 4: DNF Odds Ratios (AFTER accounting for distance)
    # Using exp(distance_baseline + centered_effect) to get full odds ratios
    # OR = 1.0 means average DNF rate for that distance
    # OR > 1.0 means higher DNF odds (harder course)
    # OR < 1.0 means lower DNF odds (easier course)
    map_values_plot4 = np.exp(course_distance_dnf_baseline_map_1 + course_effects_centered_map[:, 1])
    # MAP values: exp(course_total_dnf_effect)
    # course_total_dnf_effect = course_distance_dnf_baseline + course_effects_centered[:, 1]
    mcmc_full_dnf_odds = np.exp(course_total_dnf_effect_post)
    mcmc_full_dnf_odds_flat = mcmc_full_dnf_odds.reshape(-1, mcmc_full_dnf_odds.shape[-1])
    # MCMC posteriors: exp(course_total_dnf_effect)
    # course_total_dnf_effect_post is already a numpy array
    mcmc_full_dnf_medians = np.median(mcmc_full_dnf_odds_flat, axis=0)
    mcmc_full_dnf_q05 = np.percentile(mcmc_full_dnf_odds_flat, 5, axis=0)
    mcmc_full_dnf_q95 = np.percentile(mcmc_full_dnf_odds_flat, 95, axis=0)
    # Compute medians and quantiles
    plot_course_forest(map_values=map_values_plot4, mcmc_medians=mcmc_full_dnf_medians, mcmc_q05=mcmc_full_dnf_q05, mcmc_q95=mcmc_full_dnf_q95, xlabel='DNF Odds Ratio (after accounting for distance)', title='DNF Odds Ratios After Accounting for Distance\n(OR=1.0 = average, OR>1.0 = higher DNF odds, OR<1.0 = lower DNF odds)', reference_line=1.0, n_show=50, sort_ascending=True)
    # Reference at 1.0 (average course at distance)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MAP Inference with Runner Parameters

    Extend the model to include individual runner random effects (~100K+ runners) estimated via MAP optimization. Each runner gets:
    - **Baseline pace adjustment**: Deviation from their gender's average marathon pace
    - **Distance effect adjustment**: Deviation from their gender's average distance scaling

    These effects are modeled hierarchically with gender-specific shrinkage, so runners with few races shrink toward their gender baseline while frequent racers get individualized estimates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Runner Effect Hyperparameters from K-Core Residuals

    Calculate empirical priors for runner-level variation using residuals from the k-core MCMC model.
    """)
    return


@app.cell
def _(REFERENCE_DISTANCE, model_data_finishers, np, pd, trace, unique_courses):
    # Calculate runner-level variance components from k-core MCMC residuals
    print('📊 Calculating runner effect hyperparameters from k-core residuals...')
    print('=' * 80)
    pace_marathon_mcmc = trace.posterior['pace_marathon'].median(dim=['chain', 'draw']).values
    # Extract MCMC posterior medians for hyperparameters
    pace_distance_effect_mcmc = trace.posterior['pace_distance_effect'].median(dim=['chain', 'draw']).values
    course_effects_centered_mcmc = trace.posterior['course_effects_centered'].median(dim=['chain', 'draw']).values
    kcore_finishers = model_data_finishers.copy()
    kcore_course_names = unique_courses
    # Get k-core data with course indices
    kcore_course_to_idx = {course: _idx for _idx, course in enumerate(kcore_course_names)}
    observed_log_pace = np.log(kcore_finishers['time_ms'].values / 60000)
    _gender_idx = kcore_finishers['gender'].map({'M': 0, 'F': 1}).values
    course_idx = kcore_finishers['name'].map(kcore_course_to_idx).values
    # Calculate expected log pace for each finisher observation
    log_dist_ratio = np.log(kcore_finishers['distance_miles'].values / REFERENCE_DISTANCE)  # Convert to log minutes
    expected_log_pace = pace_marathon_mcmc[_gender_idx] + pace_distance_effect_mcmc[_gender_idx] * log_dist_ratio + course_effects_centered_mcmc[course_idx, 0]
    residuals = observed_log_pace - expected_log_pace
    runner_residuals = kcore_finishers.copy()
    runner_residuals['residual'] = residuals
    # Expected pace from gender + distance + course effects
    runner_residuals['log_dist_ratio'] = log_dist_ratio
    runner_stats = []
    for runner_id, group in runner_residuals.groupby('participant_id'):
        if len(group) >= 2:
            X = group['log_dist_ratio'].values
            y = group['residual'].values
    # Residuals = observed - expected (what's left to explain by runner effects)
            X_mean = X.mean()
            X_centered = X - X_mean
    # Group by runner and calculate within-runner statistics
            if X_centered.std() > 1e-06:
                slope = np.cov(X_centered, y)[0, 1] / np.var(X_centered)
                intercept = y.mean() - slope * X_mean
            else:
    # For each runner, compute intercept and slope of residuals vs distance
                slope = 0.0
                intercept = y.mean()
            runner_stats.append({'participant_id': runner_id, 'gender': group['gender'].iloc[0], 'n_races': len(group), 'intercept': intercept, 'slope': slope})  # Need at least 2 races to estimate both parameters
    runner_stats_df = pd.DataFrame(runner_stats)  # Simple linear regression: residual ~ log_dist_ratio
    sigma_runner_pace = runner_stats_df['intercept'].std()
    sigma_runner_distance = runner_stats_df['slope'].std()
    runner_correlation = runner_stats_df[['intercept', 'slope']].corr().iloc[0, 1]
    print(f'\nRunner Effect Standard Deviations (from k-core residuals):')  # Mean-center X for stability
    print(f'  sigma_runner_pace:     {sigma_runner_pace:.4f} (baseline pace variation)')
    print(f'  sigma_runner_distance: {sigma_runner_distance:.4f} (distance effect variation)')
    print(f'\nCorrelation between runner baseline and distance effects: {runner_correlation:.4f}')
    print(f"  Interpretation: {('Fast runners slow down MORE with distance' if runner_correlation < 0 else 'Fast runners slow down LESS with distance')}")  # Estimate slope and intercept
    print(f'\nBy Gender:')  # Check for variation in distances
    for _gender in ['M', 'F']:
        gender_df = runner_stats_df[runner_stats_df['gender'] == _gender]
        print(f'\n  {_gender}:')
        print(f'    N runners:             {len(gender_df):,}')  # All same distance - only intercept is identifiable
        print(f"    Intercept SD:          {gender_df['intercept'].std():.4f}")
        print(f"    Distance effect SD:    {gender_df['slope'].std():.4f}")
        print(f"    Correlation:           {gender_df[['intercept', 'slope']].corr().iloc[0, 1]:.4f}")
    # Calculate standard deviations across runners (between-runner variation)
    # These become our priors for runner effects
    # Calculate correlation between baseline pace and distance effect
    # By gender breakdown
    print('\n' + '=' * 80)
    return runner_correlation, sigma_runner_distance, sigma_runner_pace


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Definition with Runner Effects

    Extend `define_model()` to include runner-level random effects for baseline pace and distance scaling.
    """)
    return


@app.cell
def _(REFERENCE_DISTANCE, np, pm, pt):
    def define_model_with_runners(model_data, fixed_params, runner_hyperparams):
        """
        Define hierarchical model with runner-level random effects for pace.

        Extends define_model() by adding individual runner effects for:
        - Baseline pace (deviation from gender average)
        - Distance effect (deviation from gender average distance scaling)

        Parameters
        ----------
        model_data : pd.DataFrame
            Race results data
        fixed_params : dict
            Fixed hyperparameters from k-core MCMC (same as define_model)
        runner_hyperparams : dict
            Runner effect hyperparameters with keys:
            - 'sigma_runner_pace': SD of runner baseline pace effects
            - 'sigma_runner_distance': SD of runner distance effects
            - 'runner_correlation': Correlation between the two

        Returns
        -------
        pm.Model
            PyMC model ready for MAP optimization
        """
        unique_courses = model_data['name'].unique()
        unique_genders = np.array(['M', 'F'])  # === PREPARE DATA ARRAYS (same as original) ===
        unique_runners = model_data['participant_id'].unique()
        course_to_idx = {course: _idx for _idx, course in enumerate(unique_courses)}
        gender_to_idx = {'M': 0, 'F': 1}
        runner_to_idx = {runner: _idx for _idx, runner in enumerate(unique_runners)}
        finishers = model_data[model_data['finished'] & (model_data['time_ms'] > 0)].copy()
        distances_finishers = finishers['distance_miles'].values
        gender_indices_finishers = finishers['gender'].map(gender_to_idx).values
        course_indices_finishers = finishers['name'].map(course_to_idx).values
        runner_indices_finishers = finishers['participant_id'].map(runner_to_idx).values
        observed_times_finishers = finishers['time_ms'].values / 60000  # Finisher data
        log_distance_ratio_finishers = np.log(distances_finishers / REFERENCE_DISTANCE)
        distances_results = model_data['distance_miles'].values
        course_indices_results = model_data['name'].map(course_to_idx).values
        did_finish = (~model_data['finished']).values.astype(int)
        unique_course_distances = np.array([distances_finishers[course_indices_finishers == _i].mean() if (course_indices_finishers == _i).any() else distances_results[course_indices_results == _i].mean() if (course_indices_results == _i).any() else REFERENCE_DISTANCE for _i in range(len(unique_courses))])
        unique_log_distance_ratios = np.log(unique_course_distances / REFERENCE_DISTANCE)
        coords = {'course': unique_courses, 'gender': unique_genders, 'runner': unique_runners, 'course_effect_type': ['finish_time_total', 'dnf_total'], 'runner_effect_type': ['pace_baseline', 'pace_distance_effect'], 'finishers': np.arange(len(observed_times_finishers)), 'obs_results': np.arange(len(did_finish))}
        model = pm.Model(coords=coords)
        with model:  # DNF data
            pace_marathon = pm.Data('pace_marathon', fixed_params['pace_marathon'], dims='gender')
            pace_distance_effect = pm.Data('pace_distance_effect', fixed_params['pace_distance_effect'], dims='gender')
            finish_time_noise = fixed_params['finish_time_noise']
            dnf_rate_marathon = pm.Data('dnf_rate_marathon', fixed_params['dnf_rate_marathon'])
            dnf_distance_multiplier = pm.Data('dnf_distance_multiplier', fixed_params['dnf_distance_multiplier'])  # Pre-compute course distances
            chol = pt.as_tensor_variable(fixed_params['chol'])
            pace_marathon_avg = (pace_marathon[0] + pace_marathon[1]) / 2
            pace_distance_effect_avg = (pace_distance_effect[0] + pace_distance_effect[1]) / 2
            course_distance_finish_baseline = pace_marathon_avg + pace_distance_effect_avg * unique_log_distance_ratios
            course_distance_dnf_baseline = dnf_rate_marathon + dnf_distance_multiplier * unique_log_distance_ratios
            course_effects_raw = pm.Normal('course_effects_raw', mu=0, sigma=1, dims=('course', 'course_effect_type'))
            course_effects_centered = pm.Deterministic('course_effects_centered', pm.math.dot(course_effects_raw, chol), dims=('course', 'course_effect_type'))
            course_total_finish_effect = course_distance_finish_baseline + course_effects_centered[:, 0]
            course_total_dnf_effect = course_distance_dnf_baseline + course_effects_centered[:, 1]
            sigma_pace = runner_hyperparams['sigma_runner_pace']
            sigma_dist = runner_hyperparams['sigma_runner_distance']  # Coordinates
            rho = runner_hyperparams['runner_correlation']
            runner_chol = np.array([[sigma_pace, 0.0], [rho * sigma_dist, np.sqrt(1 - rho ** 2) * sigma_dist]])
            runner_chol_tensor = pt.as_tensor_variable(runner_chol)
            runner_effects_raw = pm.Normal('runner_effects_raw', mu=0, sigma=1, dims=('runner', 'runner_effect_type'))
            runner_effects_centered = pm.Deterministic('runner_effects_centered', pm.math.dot(runner_effects_raw, runner_chol_tensor), dims=('runner', 'runner_effect_type'))
            runner_baseline_effect = runner_effects_centered[runner_indices_finishers, 0]
            runner_distance_effect = runner_effects_centered[runner_indices_finishers, 1]
            expected_log_pace = pace_marathon[gender_indices_finishers] + pace_distance_effect[gender_indices_finishers] * log_distance_ratio_finishers + course_effects_centered[course_indices_finishers, 0] + runner_baseline_effect + runner_distance_effect * log_distance_ratio_finishers
            pm.Normal('finisher_times', mu=expected_log_pace, sigma=finish_time_noise[gender_indices_finishers], observed=np.log(observed_times_finishers), dims='finishers')
            logit_p_dnf = course_total_dnf_effect[course_indices_results]
            pm.Bernoulli('did_finish_obs', p=pm.math.invlogit(logit_p_dnf), observed=did_finish, dims='obs_results')  # === BUILD MODEL ===
        return model  # --- FIXED HYPERPARAMETERS (from k-core MCMC) ---  # Array, not Data  # --- DISTANCE BASELINES (course-level) ---  # --- COURSE EFFECTS (same as original) ---  # --- RUNNER EFFECTS (NEW!) ---  # Build Cholesky matrix for runner effects from hyperparameters  # Cholesky decomposition of correlation matrix  # L = [[a, 0], [b, c]] where a = sigma_pace, b = rho * sigma_dist, c = sqrt(1 - rho^2) * sigma_dist  # Non-centered parameterization for runner effects  # --- LIKELIHOOD ---  # FINISH TIME MODEL with runner effects  # Gender-specific shrinkage: runner effects are deviations from gender baseline  # Extract runner effects for each finisher  # Build expected pace from components:  # 1. Gender-specific baseline pace at marathon distance  # 2. Gender-specific distance scaling effect  # 3. Course difficulty (deviation from population average)  # 4. Runner baseline effect (deviation from their gender's baseline)  # 5. Runner distance effect (deviation from their gender's distance scaling)  # DNF MODEL (unchanged - no runner effects)

    return (define_model_with_runners,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build MAP Model with Runners
    """)
    return


@app.cell
def _(n_courses_full, results_2):
    print('📊 Preparing full dataset with runner indices...')
    print('=' * 80)
    unique_runners_full = results_2['participant_id'].unique()
    n_runners_full = len(unique_runners_full)
    print(f'\nFull Dataset Statistics:')
    print(f'  Total results:     {len(results_2):,}')
    print(f"  Finishers:         {results_2['finished'].sum():,}")
    print(f"  DNFs:              {(~results_2['finished']).sum():,}")
    print(f'  Unique courses:    {n_courses_full:,}')
    print(f'  Unique runners:    {n_runners_full:,}')
    print(f'\nRunners by Gender:')
    for _gender in ['M', 'F']:
        n_gender = (results_2[results_2['finished']].groupby('participant_id')['gender'].first() == _gender).sum()
        print(f'  {_gender}: {n_gender:,} ({100 * n_gender / n_runners_full:.1f}%)')
    races_per_runner = results_2.groupby('participant_id').size()
    print(f'\nRaces per Runner Distribution:')
    print(f'  Mean:   {races_per_runner.mean():.2f}')
    print(f'  Median: {races_per_runner.median():.0f}')
    print(f'  Min:    {races_per_runner.min()}')
    print(f'  Max:    {races_per_runner.max()}')
    print(f'\n  Percentiles:')
    for pct in [25, 50, 75, 90, 95, 99]:
        val = races_per_runner.quantile(pct / 100)
        print(f'    {pct}th: {val:.0f}')
    print(f'\nRunners by Race Count:')
    for min_races in [1, 2, 3, 5, 10, 20]:
        _n_runners = (races_per_runner >= min_races).sum()
        pct = 100 * _n_runners / n_runners_full
        print(f'  ≥{min_races:2d} races: {_n_runners:7,} ({pct:5.1f}%)')
    print('\n' + '=' * 80)
    return (n_runners_full,)


@app.cell
def _(
    chol_fixed_1,
    define_model_with_runners,
    dnf_distance_multiplier_fixed,
    dnf_rate_marathon_fixed,
    finish_time_noise_fixed,
    n_courses_full,
    n_runners_full,
    pace_distance_effect_fixed,
    pace_marathon_fixed,
    pm,
    results_full,
    runner_correlation,
    sigma_runner_distance,
    sigma_runner_pace,
):
    print('🔧 Building MAP model with runner parameters...')
    print('=' * 80)
    fixed_params_with_runners = {'pace_marathon': pace_marathon_fixed, 'pace_distance_effect': pace_distance_effect_fixed, 'finish_time_noise': finish_time_noise_fixed, 'dnf_rate_marathon': dnf_rate_marathon_fixed, 'dnf_distance_multiplier': dnf_distance_multiplier_fixed, 'chol': chol_fixed_1}
    runner_hyperparams = {'sigma_runner_pace': sigma_runner_pace, 'sigma_runner_distance': sigma_runner_distance, 'runner_correlation': runner_correlation}
    model_with_runners = define_model_with_runners(results_full, fixed_params_with_runners, runner_hyperparams)
    print(f'\nModel with Runner Parameters:')
    print(f'  Fixed hyperparameters:     8 (from k-core MCMC)')
    print(f'  Course effects:            {n_courses_full:,} × 2 = {n_courses_full * 2:,}')
    print(f'  Runner effects:            {n_runners_full:,} × 2 = {n_runners_full * 2:,}')
    print(f'  Total free parameters:     {(n_courses_full + n_runners_full) * 2:,}')
    print(f'\nObservations:')
    print(f"  Finisher observations:     {results_full['finished'].sum():,}")
    print(f"  DNF observations:          {(~results_full['finished']).sum():,}")
    print(f'  Total observations:        {len(results_full):,}')
    print('\n📊 Model structure:')
    _graph = pm.model_to_graphviz(model_with_runners)
    _graph
    return (model_with_runners,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run MAP Optimization

    Estimate all ~200K+ course and runner parameters via L-BFGS-B optimization.
    """)
    return


@app.cell
def _(
    alpha,
    az,
    beta,
    map_trace,
    model_dir,
    model_with_runners,
    n_courses_full,
    n_runners_full,
    np,
    pace_distance_effect_fixed,
    pace_marathon_fixed,
    pm,
    results_full,
    time,
    unique_genders,
):
    # Cache file for MAP with runners
    map_runners_cache_file = model_dir / f'map_with_runners_alpha{alpha}_beta{beta}.nc'
    if map_runners_cache_file.exists():
        print(f'📂 Loading cached MAP estimates from {map_runners_cache_file}...')
        map_runners_trace = az.from_netcdf(map_runners_cache_file)
        print(f"   Loaded {len(map_runners_trace.posterior.coords['course']):,} course effects")
        print(f"   Loaded {len(map_runners_trace.posterior.coords['runner']):,} runner effects")
    else:
        print('🚀 Running MAP estimation with runner parameters...')
        print(f'   This may take 15-45 minutes depending on dataset size')
        print(f'   Estimating {(n_courses_full + n_runners_full) * 2:,} parameters...')
        print()
        _init_course_effects_raw = map_trace.posterior['course_effects_raw'].values[0, 0, :, :]
        init_runner_effects_raw = np.zeros((n_runners_full, 2))
        _start_time = time.time()  # Initialize course effects from MAP course-only estimates
        with model_with_runners:
            map_runners_estimate = pm.find_MAP(method='L-BFGS-B', start={'course_effects_raw': _init_course_effects_raw, 'runner_effects_raw': init_runner_effects_raw}, progressbar=True)
        _elapsed_time = time.time() - _start_time  # Initialize runner effects at zero (prior mean)
        print(f'\n✅ MAP estimation completed in {_elapsed_time:.2f} seconds')
        print(f'   ({_elapsed_time / 60:.1f} minutes)')
        pace_marathon_avg_runners = pace_marathon_fixed.mean()
        pace_distance_effect_avg_runners = pace_distance_effect_fixed.mean()
        unique_runners_arr = np.array(sorted(results_full['participant_id'].unique()))
        unique_courses_arr = np.array(sorted(results_full['name'].unique()))
        map_runners_trace = az.from_dict(posterior={'course_effects_raw': map_runners_estimate['course_effects_raw'][np.newaxis, np.newaxis, :, :], 'course_effects_centered': map_runners_estimate['course_effects_centered'][np.newaxis, np.newaxis, :, :], 'runner_effects_raw': map_runners_estimate['runner_effects_raw'][np.newaxis, np.newaxis, :, :], 'runner_effects_centered': map_runners_estimate['runner_effects_centered'][np.newaxis, np.newaxis, :, :], 'pace_marathon': pace_marathon_fixed[np.newaxis, np.newaxis, :], 'pace_distance_effect': pace_distance_effect_fixed[np.newaxis, np.newaxis, :]}, coords={'course': unique_courses_arr, 'runner': unique_runners_arr, 'gender': unique_genders, 'course_effect_type': ['finish_time_total', 'dnf_total'], 'runner_effect_type': ['pace_baseline', 'pace_distance_effect']}, dims={'course_effects_raw': ['course', 'course_effect_type'], 'course_effects_centered': ['course', 'course_effect_type'], 'runner_effects_raw': ['runner', 'runner_effect_type'], 'runner_effects_centered': ['runner', 'runner_effect_type'], 'pace_marathon': ['gender'], 'pace_distance_effect': ['gender']})
        map_runners_trace.to_netcdf(map_runners_cache_file)
        print(f'💾 Saved MAP trace to {map_runners_cache_file}')
    print()
    print('MAP estimation with runners summary:')
    print(f"  Course effects: {len(map_runners_trace.posterior.coords['course']):,}")
    print(f"  Runner effects: {len(map_runners_trace.posterior.coords['runner']):,}")  # Reconstruct derived quantities  # Convert to InferenceData  # Save to cache
    return (map_runners_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Extract Runner Effects for Visualization
    """)
    return


@app.cell
def _(map_runners_trace, np, pd, results_2):
    runner_effects_centered_map = map_runners_trace.posterior['runner_effects_centered'].values[0, 0, :, :]
    map_runners_arr = map_runners_trace.posterior.coords['runner'].values
    runner_baseline_pace = runner_effects_centered_map[:, 0]
    runner_distance_effect = runner_effects_centered_map[:, 1]
    runner_gender = results_2.groupby('participant_id')['gender'].first().reindex(map_runners_arr).values
    runner_race_counts = results_2.groupby('participant_id').size().reindex(map_runners_arr).values
    runner_names_df = results_2.groupby('participant_id').agg({'firstname': 'first', 'lastname': 'first'}).reindex(map_runners_arr)
    runner_full_names = []
    for _idx, pid in enumerate(map_runners_arr):
        firstname = runner_names_df.iloc[_idx]['firstname']
        lastname = runner_names_df.iloc[_idx]['lastname']
        if pd.notna(firstname) and pd.notna(lastname):
            runner_full_names.append(f'{firstname} {lastname}')
        else:
            runner_full_names.append(f'Runner {pid}')
    runner_full_names = np.array(runner_full_names)
    print('✅ Extracted runner effects:')
    print(f'  N runners: {len(runner_baseline_pace):,}')
    print(f'  Baseline pace range: [{runner_baseline_pace.min():.3f}, {runner_baseline_pace.max():.3f}]')
    print(f'  Distance effect range: [{runner_distance_effect.min():.3f}, {runner_distance_effect.max():.3f}]')
    print(f'  Sample names: {runner_full_names[:5]}')
    return (
        map_runners_arr,
        runner_baseline_pace,
        runner_distance_effect,
        runner_full_names,
        runner_gender,
        runner_race_counts,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Runner Parameter Visualizations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot 1 & 2: Runner Effect Distributions by Gender
    """)
    return


@app.cell
def _(plt, runner_baseline_pace, runner_distance_effect, runner_gender, sns):
    # Plot 1 & 2: Distribution of runner effects by gender
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
    for _gender, _color in [('M', 'steelblue'), ('F', '#FF69B4')]:
    # Plot 1: Baseline Pace Distribution
        _mask = runner_gender == _gender
        sns.kdeplot(data=runner_baseline_pace[_mask], ax=_ax1, color=_color, linewidth=2.5, fill=False, label=f'{_gender} (n={_mask.sum():,})')
    _ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Gender Average')
    _ax1.set_xlabel('Runner Baseline Pace Adjustment (log minutes)', fontsize=12, fontweight='bold')
    _ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    _ax1.set_title('Runner Baseline Pace Effects by Gender\nDeviation from Gender Average', fontsize=14, fontweight='bold')
    _ax1.legend(fontsize=11)
    _ax1.grid(alpha=0.3, color='gray')
    _ax1.set_facecolor('white')
    for _gender, _color in [('M', 'steelblue'), ('F', '#FF69B4')]:
        _mask = runner_gender == _gender
        sns.kdeplot(data=runner_distance_effect[_mask], ax=_ax2, color=_color, linewidth=2.5, fill=False, label=f'{_gender} (n={_mask.sum():,})')
    _ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Gender Average')
    _ax2.set_xlabel('Runner Distance Scaling Adjustment', fontsize=12, fontweight='bold')
    _ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    _ax2.set_title('Runner Distance Effects by Gender\nDeviation from Gender Average Slowdown', fontsize=14, fontweight='bold')
    _ax2.legend(fontsize=11)
    _ax2.grid(alpha=0.3, color='gray')
    _ax2.set_facecolor('white')
    plt.tight_layout()
    # Plot 2: Distance Effect Distribution
    plt.show()
    print('\nRunner Effect Summary Statistics:')
    print('=' * 80)
    for _gender in ['M', 'F']:
        _mask = runner_gender == _gender
        print(f'\n{_gender} Runners (n={_mask.sum():,}):')
        print(f'  Baseline Pace:')
        print(f'    Mean:   {runner_baseline_pace[_mask].mean():+.4f}')
        print(f'    Std:    {runner_baseline_pace[_mask].std():.4f}')
        print(f'    Range:  [{runner_baseline_pace[_mask].min():+.4f}, {runner_baseline_pace[_mask].max():+.4f}]')
        print(f'  Distance Effect:')
        print(f'    Mean:   {runner_distance_effect[_mask].mean():+.4f}')
        print(f'    Std:    {runner_distance_effect[_mask].std():.4f}')
        print(f'    Range:  [{runner_distance_effect[_mask].min():+.4f}, {runner_distance_effect[_mask].max():+.4f}]')
    # Summary statistics
    print('=' * 80)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot 3: Runner Effects Correlation Scatter
    """)
    return


@app.cell
def _(np, plt, runner_baseline_pace, runner_distance_effect, runner_gender):
    # Plot 3: Correlation between runner baseline and distance effects
    _fig, _ax = plt.subplots(1, 1, figsize=(12, 10), facecolor='white')
    _n_plot = min(10000, len(runner_baseline_pace))
    # Subsample for visibility (plot max 10K points)
    _plot_indices = np.random.choice(len(runner_baseline_pace), _n_plot, replace=False)
    for _gender, _color in [('M', 'steelblue'), ('F', '#FF69B4')]:
        _mask = runner_gender[_plot_indices] == _gender
    # Plot by gender
        _ax.scatter(runner_baseline_pace[_plot_indices][_mask], runner_distance_effect[_plot_indices][_mask], alpha=0.3, s=20, color=_color, label=f'{_gender} (n={_mask.sum():,} plotted)')
    _ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    _ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    xlim = _ax.get_xlim()
    ylim = _ax.get_ylim()
    _ax.text(xlim[0] * 0.8, ylim[1] * 0.8, 'Slow but\nSteady', fontsize=11, alpha=0.6, ha='left', va='top', style='italic')
    _ax.text(xlim[1] * 0.8, ylim[1] * 0.8, 'Fast but\nFade', fontsize=11, alpha=0.6, ha='right', va='top', style='italic')
    _ax.text(xlim[0] * 0.8, ylim[0] * 0.8, 'Slow and\nFade', fontsize=11, alpha=0.6, ha='left', va='bottom', style='italic')
    _ax.text(xlim[1] * 0.8, ylim[0] * 0.8, 'Fast at\nAll Distances', fontsize=11, alpha=0.6, ha='right', va='bottom', style='italic')
    _ax.set_xlabel('Runner Baseline Pace Adjustment (log minutes)', fontsize=12, fontweight='bold')
    _ax.set_ylabel('Runner Distance Scaling Adjustment', fontsize=12, fontweight='bold')
    # Add reference lines
    _ax.set_title('Runner Baseline vs Distance Effects\nColored by Gender', fontsize=14, fontweight='bold')
    _ax.legend(fontsize=11, loc='upper left')
    _ax.grid(alpha=0.3, color='gray')
    # Add quadrant labels
    _ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    print(f'\nCorrelation between baseline pace and distance effect: {np.corrcoef(runner_baseline_pace, runner_distance_effect)[0, 1]:.4f}')
    # Calculate correlation
    print(f"Interpretation: {('Fast runners slow down MORE with distance' if np.corrcoef(runner_baseline_pace, runner_distance_effect)[0, 1] < 0 else 'Fast runners slow down LESS with distance')}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Forest Plots: Top/Bottom Runners
    """)
    return


@app.cell
def _(np, plt):
    def plot_runner_forest(runner_values, runner_ids, runner_genders, runner_races, xlabel, title, reference_line=0.0, n_show=50, sort_ascending=True, min_races=1, runner_names=None):
        """
        Create forest plot showing top/bottom runners.

        Parameters
        ----------
        runner_values : np.ndarray
            Runner effect values
        runner_ids : np.ndarray
            Runner IDs
        runner_genders : np.ndarray
            Runner genders
        runner_races : np.ndarray
            Number of races per runner
        xlabel : str
            X-axis label
        title : str
            Plot title
        reference_line : float
            Value for vertical reference line
        n_show : int
            Number to show from each end
        sort_ascending : bool
            Sort direction
        min_races : int
            Minimum races to include
        runner_names : np.ndarray, optional
            Runner names (if None, uses runner_ids)

        Returns
        -------
        fig, ax
        """
        from matplotlib.lines import Line2D
        _mask = runner_races >= min_races
        filtered_values = runner_values[_mask]
        filtered_ids = runner_ids[_mask]
        filtered_genders = runner_genders[_mask]  # Filter by minimum races
        filtered_races = runner_races[_mask]
        if runner_names is not None:
            filtered_names = runner_names[_mask]
        sorted_indices = np.argsort(filtered_values)
        if not sort_ascending:
            sorted_indices = sorted_indices[::-1]
        top_indices = sorted_indices[:n_show]  # Filter names if provided
        bottom_indices = sorted_indices[-n_show:]
        selected_indices = np.concatenate([top_indices, bottom_indices])
        _fig, _ax = plt.subplots(1, 1, figsize=(10, 20), facecolor='white')
        _ax.set_facecolor('white')  # Sort
        selected_values = filtered_values[selected_indices]
        selected_ids = filtered_ids[selected_indices]
        selected_genders = filtered_genders[selected_indices]
        selected_races = filtered_races[selected_indices]
        if runner_names is not None:  # Select top and bottom
            selected_names = filtered_names[selected_indices]
        y_positions = np.arange(len(selected_values))
        for _i, (val, _gender) in enumerate(zip(selected_values, selected_genders)):
            _color = 'steelblue' if _gender == 'M' else '#FF69B4'
            _ax.plot(val, y_positions[_i], 's', color=_color, markersize=6, alpha=0.8, zorder=5)  # Create figure
        if reference_line is not None:
            _ax.axvline(x=reference_line, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Reference ({reference_line})')
        if runner_names is not None:
            y_labels = [f'{name} ({g}, {nr} races)' for name, g, nr in zip(selected_names, selected_genders, selected_races)]  # Extract for selected
        else:
            y_labels = [f'Runner {rid} ({g}, {nr} races)' for rid, g, nr in zip(selected_ids, selected_genders, selected_races)]
        _ax.set_yticks(y_positions)
        _ax.set_yticklabels(y_labels, fontsize=7)
        _ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        _ax.set_title(title, fontsize=14, fontweight='bold')
        _ax.grid(axis='x', alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
        legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor='steelblue', markersize=8, label='Male'), Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF69B4', markersize=8, label='Female'), Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label=f'Reference ({reference_line})')]
        _ax.legend(handles=legend_elements, loc='lower right', fontsize=10)  # Y positions
        _ax.axhline(y=n_show - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.tight_layout()
        return (_fig, _ax)  # Plot points with gender colors  # Reference line  # Y labels with names (if provided) or IDs  # Grid  # Legend  # Dividing line

    return (plot_runner_forest,)


@app.cell
def _(
    map_runners_arr,
    plot_runner_forest,
    plt,
    runner_baseline_pace,
    runner_full_names,
    runner_gender,
    runner_race_counts,
):
    # Plot 4 & 5: Fastest and Slowest Runners (Baseline Pace)
    _fig, _ax = plot_runner_forest(runner_baseline_pace, map_runners_arr, runner_gender, runner_race_counts, xlabel='Baseline Pace Adjustment (log minutes)', title='Top 50 Fastest & Bottom 50 Slowest Runners (Baseline Pace)\nMAP Estimates (≥5 races)', reference_line=0.0, n_show=50, sort_ascending=True, min_races=5, runner_names=runner_full_names)
    plt.show()
    return


@app.cell
def _(
    map_runners_arr,
    plot_runner_forest,
    plt,
    runner_distance_effect,
    runner_full_names,
    runner_gender,
    runner_race_counts,
):
    # Plot 6 & 7: Best and Worst Distance Scalers
    # Filter to runners with multiple races for better identification
    _fig, _ax = plot_runner_forest(runner_distance_effect, map_runners_arr, runner_gender, runner_race_counts, xlabel='Distance Scaling Adjustment', title='Top 50 Best & Bottom 50 Worst Distance Scalers\nMAP Estimates (≥3 races)', reference_line=0.0, n_show=50, sort_ascending=True, min_races=3, runner_names=runner_full_names)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot 8: Hierarchical Shrinkage Demonstration
    """)
    return


@app.cell
def _(
    np,
    plt,
    runner_baseline_pace,
    runner_distance_effect,
    runner_race_counts,
):
    # Plot 8: Shrinkage - Runner Effects vs Number of Races
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
    _n_plot = min(10000, len(runner_baseline_pace))
    # Subsample for plotting
    _plot_indices = np.random.choice(len(runner_baseline_pace), _n_plot, replace=False)
    _ax1.scatter(runner_race_counts[_plot_indices], np.abs(runner_baseline_pace[_plot_indices]), alpha=0.3, s=10, color='steelblue')
    from scipy.ndimage import gaussian_filter1d
    # Left: Baseline pace magnitude vs races
    race_bins = np.arange(1, min(50, runner_race_counts.max()) + 1)
    bin_means = []
    for nb in race_bins:
        _mask = runner_race_counts[_plot_indices] == nb
        if _mask.sum() > 10:
            bin_means.append(np.abs(runner_baseline_pace[_plot_indices][_mask]).mean())
        else:
            bin_means.append(np.nan)
    # Add smoothed trend line
    valid_mask = ~np.isnan(bin_means)
    if valid_mask.sum() > 3:
        smoothed = gaussian_filter1d(np.array(bin_means)[valid_mask], sigma=1.5)
        _ax1.plot(race_bins[valid_mask], smoothed, color='red', linewidth=3, label='Smoothed Trend', zorder=10)
    _ax1.set_xlabel('Number of Races', fontsize=12, fontweight='bold')
    _ax1.set_ylabel('|Baseline Pace Effect|', fontsize=12, fontweight='bold')
    _ax1.set_title('Hierarchical Shrinkage: Baseline Pace\nRunners with Few Races → Gender Baseline', fontsize=14, fontweight='bold')
    _ax1.legend(fontsize=11)
    _ax1.grid(alpha=0.3, color='gray')
    _ax1.set_facecolor('white')
    _ax1.set_xlim(left=0)
    _ax2.scatter(runner_race_counts[_plot_indices], np.abs(runner_distance_effect[_plot_indices]), alpha=0.3, s=10, color='#FF69B4')
    bin_means_dist = []
    for nb in race_bins:
        _mask = runner_race_counts[_plot_indices] == nb
        if _mask.sum() > 10:
            bin_means_dist.append(np.abs(runner_distance_effect[_plot_indices][_mask]).mean())
        else:
            bin_means_dist.append(np.nan)
    valid_mask_dist = ~np.isnan(bin_means_dist)
    if valid_mask_dist.sum() > 3:
        smoothed_dist = gaussian_filter1d(np.array(bin_means_dist)[valid_mask_dist], sigma=1.5)
        _ax2.plot(race_bins[valid_mask_dist], smoothed_dist, color='red', linewidth=3, label='Smoothed Trend', zorder=10)
    _ax2.set_xlabel('Number of Races', fontsize=12, fontweight='bold')
    _ax2.set_ylabel('|Distance Effect|', fontsize=12, fontweight='bold')
    # Right: Distance effect magnitude vs races
    _ax2.set_title('Hierarchical Shrinkage: Distance Effect\nRunners with Few Races → Gender Baseline', fontsize=14, fontweight='bold')
    _ax2.legend(fontsize=11)
    _ax2.grid(alpha=0.3, color='gray')
    _ax2.set_facecolor('white')
    _ax2.set_xlim(left=0)
    plt.tight_layout()
    plt.show()
    print('\nShrinkage Demonstration:')
    print('=' * 80)
    print('Runners with few races have effects closer to 0 (gender baseline)')
    print('Runners with many races get individualized estimates away from 0')
    print('This is hierarchical shrinkage in action!')
    print('=' * 80)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Comparison Summary

    Compare course-only vs runner-augmented models to quantify improvement from modeling individual runner heterogeneity.
    """)
    return


@app.cell
def _(n_courses_full, n_runners_full):
    # Model Comparison: Course-Only vs Course + Runners
    print("📊 Model Comparison: Course-Only vs Course + Runners")
    print("=" * 80)

    print("\nParameter Counts:")
    print(f"  Course-Only Model:")
    print(f"    Hyperparameters (fixed):  8")
    print(f"    Course effects:           {n_courses_full * 2:,}")
    print(f"    Total parameters:         {n_courses_full * 2:,}")
    print()
    print(f"  Course + Runner Model:")
    print(f"    Hyperparameters (fixed):  8")
    print(f"    Course effects:           {n_courses_full * 2:,}")
    print(f"    Runner effects:           {n_runners_full * 2:,}")
    print(f"    Total parameters:         {(n_courses_full + n_runners_full) * 2:,}")
    print()
    print(f"  Parameter increase:         {n_runners_full * 2:,} (+{100 * n_runners_full * 2 / (n_courses_full * 2):.1f}%)")

    print("\n" + "=" * 80)
    print("\nKey Insights:")
    print("  1. Runner effects capture individual ability and distance scaling")
    print("  2. Gender-specific shrinkage ensures runners shrink toward their gender baseline")
    print("  3. Hierarchical structure means low-race runners → gender average")
    print("  4. High-race runners get individualized estimates")
    print("  5. This enables runner-specific predictions and counterfactuals")
    print("=" * 80)
    return


if __name__ == "__main__":
    app.run()
