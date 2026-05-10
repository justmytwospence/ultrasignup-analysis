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
    # Finish Time Model

    **Gender-Specific Power Law with Course Effects**

    This model estimates ultrarunning race pace using gender-specific power law relationships between distance and pace, accounting for course-specific difficulty. The model pools data across multiple years of the same race course to improve parameter estimation.

    **Model Data:** All finishers from races that report DNFs

    **Data Subsetting:** Well connected k-core subset + closure courses/participants up to `max_entities`

    **Hyperparameters**

    - `course_finish_time_multiplier_std`: Variation in course difficulty (finish times)

    **Runner Parameters**

    - `pace_distance_effect[gender]`: Distance exponent (how pace degrades with distance, per gender)
    - `finish_time_noise[gender]`: Observation noise (gender-specific)
    - `pace_marathon[gender]`: Average pace at marathon distance (per gender)

    **Course Parameters (n_courses)**

    - `course_finish_time_multiplier`: Course-specific pace adjustment (shared across genders)
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
    # libraries
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import pymc as pm
    import numpy as np
    import arviz as az
    from pathlib import Path
    from plotly.subplots import make_subplots
    from matplotlib import pyplot as plt
    from scipy.stats import pearsonr
    import os
    import time

    # local functions
    from utils.data_processing import load_results, process_results, filter_races_with_dnfs
    from utils.kcore_subsetting import subset_kcore_data
    from utils.mcmc_notifications import notify_mcmc_start, notify_mcmc_complete, notify_mcmc_error
    from utils.empirical_priors import (
        calculate_gender_marathon_pace,
        calculate_gender_distance_exponent,
        calculate_course_variation
    )
    from utils.utils import plot_posterior_diagnostics

    return (
        Path,
        az,
        calculate_course_variation,
        calculate_gender_distance_exponent,
        calculate_gender_marathon_pace,
        filter_races_with_dnfs,
        load_results,
        notify_mcmc_complete,
        notify_mcmc_error,
        notify_mcmc_start,
        np,
        os,
        plot_posterior_diagnostics,
        plt,
        pm,
        process_results,
        subset_kcore_data,
        time,
    )


@app.cell
def _(Path, os):
    # define variables

    # Standard race distances for analysis and visualization
    # Format: (distance_miles, label, tolerance_for_binning)
    standard_distances = [
        (6.21371, '10K', 0.3),
        (10.0, '10mi', 0.3),
        (13.1, 'Half', 0.3),
        (26.2, 'Marathon', 0.5),
        (31.0686, '50K', 0.5),
        (50.0, '50mi', 1.0),
        (62.1371, '100K', 1.0),
        (100.0, '100mi', 1.0)
    ]

    reference_distance = 26.2  # Marathon

    # K-core parameters for dev iteration.
    # Cached traces at (3, 840) and (3, 423) tune=500 were fit with broken time units
    # (observed_times /1000 instead of /60000). Bumping to tune=1000/draws=1000 forces
    # a fresh fit with corrected units. (3, 840) keeps the dataset small (~11 courses,
    # ~22K entities) so the fit completes in single-digit minutes for diagnostics.
    alpha = 3  # Minimum courses per k-core runner
    beta = 840  # Minimum runners per k-core course

    tune = 2000
    draws = 1000
    target_accept = 0.9

    # Path is anchored to the notebook file so caches resolve regardless of
    # marimo's CWD (which is the dir you ran 'uv run marimo' from, not notebooks/)
    model_dir = Path(__file__).resolve().parent.parent / 'data' / 'cache' / 'model_1'
    os.makedirs(model_dir, exist_ok=True)

    subset_dir = f'{model_dir}/alpha{alpha}_beta{beta}'
    os.makedirs(subset_dir, exist_ok=True)
    return (
        alpha,
        beta,
        draws,
        model_dir,
        reference_distance,
        standard_distances,
        subset_dir,
        target_accept,
        tune,
    )


@app.cell
def _(load_results, process_results):
    # load and process data

    results = load_results()
    results = process_results(results)
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Filtering

    Filter the dataset to the results relevant to this model
    """)
    return


@app.cell
def _(filter_races_with_dnfs, results):
    # this model does not attempt to model DNF reporting or DNFs themselves
    results_1 = filter_races_with_dnfs(results)
    results_1 = results_1[results_1.finished == 1]
    return (results_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Subsetting

    Subset the data to a well-connected subgraph for computational efficiency
    """)
    return


@app.cell
def _(alpha, beta, results_1, subset_kcore_data):
    # Apply k-core flagging with course-completion closure
    results_2 = subset_kcore_data(results_1, alpha=alpha, beta=beta)
    model_data = results_2[results_2['in_kcore'] | results_2['in_closure']]
    print(f'\nK-core subset (α={alpha}, β={beta}):')
    print(f"  K-core results: {results_2['in_kcore'].sum():,}")
    print(f"  Closure results: {results_2['in_closure'].sum():,}")
    print(f'  Total for modeling: {len(model_data):,}')
    print(f"  K-core courses: {results_2[results_2['in_kcore']]['name'].nunique():,}")
    print(f"  K-core runners: {results_2[results_2['in_kcore']]['participant_id'].nunique():,}")
    return model_data, results_2


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

    Compute empirical priors on all the results relevant to this model
    """)
    return


@app.cell
def _(
    calculate_course_variation,
    calculate_gender_distance_exponent,
    calculate_gender_marathon_pace,
    reference_distance,
    results_2,
):
    # Calculate empirical priors using simple, focused functions
    # Each function computes ONE specific prior from the data
    mu_pace_m = calculate_gender_marathon_pace(results_2, 'M', reference_distance)
    # Gender-specific marathon pace (mu_pace prior)
    mu_pace_f = calculate_gender_marathon_pace(results_2, 'F', reference_distance)
    beta_m = calculate_gender_distance_exponent(results_2, 'M', reference_distance)
    beta_f = calculate_gender_distance_exponent(results_2, 'F', reference_distance)
    # Gender-specific distance exponent (beta prior)
    # Course-level variation (sigma_course prior)
    sigma_course_prior = calculate_course_variation(results_2, reference_distance)
    return beta_f, beta_m, mu_pace_f, mu_pace_m, sigma_course_prior


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generative Model
    """)
    return


@app.cell
def _(
    beta_f,
    beta_m,
    mo,
    model_data,
    model_dir,
    mu_pace_f,
    mu_pace_m,
    pm,
    sigma_course_prior,
):
    unique_courses = model_data['name'].unique()
    course_to_idx = {course: _idx for _idx, course in enumerate(unique_courses)}
    model_data['course_idx'] = model_data['name'].map(course_to_idx)
    unique_genders = ['M', 'F']
    gender_to_idx = {'M': 0, 'F': 1}
    model_data['gender_idx'] = model_data['gender'].map(gender_to_idx)
    n_courses = len(unique_courses)
    n_genders = len(unique_genders)
    n_observations = len(model_data)
    course_indices = model_data['course_idx'].values
    gender_indices = model_data['gender_idx'].values
    race_distances = model_data['distance_miles'].values
    observed_times = model_data['time_ms'].values / 60000  # minutes (matches expected_time units)
    reference_distance_1 = 26.2
    coords = {'course': unique_courses, 'gender': unique_genders, 'finishers': range(n_observations)}
    with pm.Model(coords=coords) as model:
        # Loosened from sigma=0.1 → 0.3 so the prior doesn't fight the likelihood at
        # the tails of the empirical pace distribution.
        _pace_marathon = pm.Normal('pace_marathon', mu=[mu_pace_m, mu_pace_f], sigma=0.3, dims='gender')
        _pace_distance_effect = pm.Normal('pace_distance_effect', mu=[beta_m, beta_f], sigma=0.05, dims='gender')
        _course_finish_time_multiplier_std = pm.HalfNormal('course_finish_time_multiplier_std', sigma=sigma_course_prior)
        # finish_time_noise is now sigma on log(time) — same prior scale (~0.15) gives
        # ~16% CV per finisher, matching the previous Normal-on-time model's effective
        # dispersion but with proper positive-support, multiplicative noise.
        _finish_time_noise = pm.HalfNormal('finish_time_noise', sigma=0.15, dims='gender')
        _course_finish_time_multiplier_raw = pm.Normal('course_finish_time_multiplier_raw', mu=0, sigma=1, dims='course')
        _course_finish_time_multiplier = pm.Deterministic('course_finish_time_multiplier', _course_finish_time_multiplier_std * _course_finish_time_multiplier_raw, dims='course')
        _log_distance_ratio = pm.math.log(race_distances / reference_distance_1)
        _expected_log_pace = _pace_marathon[gender_indices] + _pace_distance_effect[gender_indices] * _log_distance_ratio + _course_finish_time_multiplier[course_indices]
        _expected_pace = pm.Deterministic('expected_pace', pm.math.exp(_expected_log_pace), dims='finishers')
        _expected_log_time = _expected_log_pace + pm.math.log(race_distances)
        pm.LogNormal('finish_times', mu=_expected_log_time, sigma=_finish_time_noise[gender_indices], observed=observed_times, dims='finishers')
    _graph = pm.model_to_graphviz(model)
    graph_file = f'{model_dir}/model_structure'
    _graph.graph_attr['dpi'] = '72'
    _graph.render(graph_file, format='png', cleanup=True)
    print(f'Saved model structure to {graph_file}.png')
    # Wrap SVG in a scrollable container so wide DAGs don't overflow the cell.
    mo.Html(
        f'<div style="max-width:100%; overflow-x:auto">{_graph.pipe(format="svg").decode()}</div>'
    )
    return (
        gender_indices,
        gender_to_idx,
        model,
        n_courses,
        n_genders,
        n_observations,
        observed_times,
        race_distances,
        reference_distance_1,
        unique_genders,
    )


@app.cell
def _(model, pm):
    with model:
        prior_pred = pm.sample_prior_predictive(samples=500, random_seed=37)
    return (prior_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior Visualization
    """)
    return


@app.cell
def _(
    beta_f,
    beta_m,
    mu_pace_f,
    mu_pace_m,
    n_genders,
    np,
    plt,
    prior_pred,
    sigma_course_prior,
    unique_genders,
):
    # Prior Distribution Visualizations
    # Extract prior samples from prior_pred and plot as KDEs to show theoretical distributions
    _pace_marathon_prior = prior_pred.prior['pace_marathon'].values  # Shape: (chains, draws, n_genders)
    _pace_distance_effect_prior = prior_pred.prior['pace_distance_effect'].values
    course_finish_time_multiplier_std_prior = prior_pred.prior['course_finish_time_multiplier_std'].values
    _finish_time_noise_prior = prior_pred.prior['finish_time_noise'].values
    _pace_marathon_flat = _pace_marathon_prior.reshape(-1, n_genders)
    # Flatten to (samples,) for scalar parameters or (samples, dim) for vector parameters
    _pace_distance_effect_flat = _pace_distance_effect_prior.reshape(-1, n_genders)
    course_finish_time_multiplier_std_flat = course_finish_time_multiplier_std_prior.flatten()
    finish_time_noise_flat = _finish_time_noise_prior.reshape(-1, n_genders)
    pace_marathon_interpretable = np.exp(_pace_marathon_flat)
    pace_distance_effect_interpretable = _pace_distance_effect_flat
    # Convert to interpretable units
    # pace_marathon: exp(log pace) = pace in min/mile
    course_mult_std_interpretable = np.exp(course_finish_time_multiplier_std_flat)
    finish_time_noise_interpretable = np.exp(finish_time_noise_flat)
    # pace_distance_effect: already interpretable (unitless exponent)
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))
    _axes = _axes.flatten()
    # course_finish_time_multiplier_std: convert to time multiplier via exp(std)
    gender_colors_plot = {'M': 'steelblue', 'F': 'coral'}
    gender_labels_plot = {'M': 'Male', 'F': 'Female'}
    # finish_time_noise: convert to time multiplier via exp(std)
    _ax = _axes[0]
    for _gender_idx, _gender in enumerate(unique_genders):
    # ============================================================================
    # PLOT 1: Hyperparameter Priors (only gender and population-level parameters)
        import seaborn as sns
        sns.kdeplot(data=pace_marathon_interpretable[:, _gender_idx], ax=_ax, color=gender_colors_plot[_gender], label=gender_labels_plot[_gender], linewidth=2.5, fill=False)
        empirical_value = np.exp(mu_pace_m if _gender == 'M' else mu_pace_f)
        _ax.axvline(x=empirical_value, color=gender_colors_plot[_gender], linestyle='--', linewidth=2, alpha=0.8)
    _ax.set_xlabel('Pace (min/mile)', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    _ax.set_title('pace_marathon\nAverage running pace at marathon distance', fontsize=12, fontweight='bold')
    # 1. pace_marathon (gender-specific) - in min/mile
    _ax.legend()
    _ax.grid(alpha=0.3)
    _ax = _axes[1]  # Use seaborn kdeplot for smooth KDE curves
    for _gender_idx, _gender in enumerate(unique_genders):
        sns.kdeplot(data=pace_distance_effect_interpretable[:, _gender_idx], ax=_ax, color=gender_colors_plot[_gender], label=gender_labels_plot[_gender], linewidth=2.5, fill=False)
        empirical_value = beta_m if _gender == 'M' else beta_f
        _ax.axvline(x=empirical_value, color=gender_colors_plot[_gender], linestyle='--', linewidth=2, alpha=0.8)
    _ax.set_xlabel('Exponent (unitless)', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')  # Add empirical prior as vertical line
    _ax.set_title('pace_distance_effect\nHow pace slows as distance doubles', fontsize=12, fontweight='bold')
    _ax.legend()
    _ax.grid(alpha=0.3)
    _ax = _axes[2]
    sns.kdeplot(data=course_mult_std_interpretable, ax=_ax, color='#4a4a4a', linewidth=2.5, fill=False)
    empirical_multiplier = np.exp(sigma_course_prior)
    _ax.axvline(x=empirical_multiplier, color='#4a4a4a', linestyle='--', linewidth=2, alpha=0.8, label='Empirical prior')
    _ax.axvline(x=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='No variation')
    _ax.set_xlabel('Time Multiplier (×)', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    # 2. pace_distance_effect (gender-specific)
    _ax.set_title('course_finish_time_multiplier_std\nTypical course-to-course time variation', fontsize=12, fontweight='bold')
    _ax.legend()
    _ax.grid(alpha=0.3)
    _ax = _axes[3]
    for _gender_idx, _gender in enumerate(unique_genders):
        sns.kdeplot(data=finish_time_noise_interpretable[:, _gender_idx], ax=_ax, color=gender_colors_plot[_gender], label=gender_labels_plot[_gender], linewidth=2.5, fill=False)
    _ax.axvline(x=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='No variation')  # Add empirical prior as vertical line
    _ax.set_xlabel('Time Multiplier (×)', fontsize=11, fontweight='bold')
    _ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    _ax.set_title('finish_time_noise\nWithin-person variability in finish times', fontsize=12, fontweight='bold')
    _ax.legend()
    _ax.grid(alpha=0.3)
    plt.suptitle('Prior Distributions: Hyperparameters', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    # 3. course_finish_time_multiplier_std (scalar) - as time multiplier
    # 4. finish_time_noise (gender-specific) - as time multiplier
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior Predictive Check

    Check that the data generated by the model matches the observed data
    """)
    return


@app.cell
def _(
    gender_to_idx,
    mo,
    n_courses,
    n_genders,
    np,
    plt,
    prior_pred,
    reference_distance_1,
    results_2,
    standard_distances,
    unique_genders,
):
    _pace_marathon_prior = prior_pred.prior['pace_marathon'].values
    _pace_distance_effect_prior = prior_pred.prior['pace_distance_effect'].values
    course_finish_time_multiplier_prior = prior_pred.prior['course_finish_time_multiplier'].values
    baseline_flat = course_finish_time_multiplier_prior.reshape(-1, course_finish_time_multiplier_prior.shape[-1])
    _finish_time_noise_prior = prior_pred.prior['finish_time_noise'].values.flatten()
    _gender_display_names = {'M': 'Male', 'F': 'Female'}
    _full_observed_times = results_2['time_ms'].values / 60000
    _full_gender_indices = results_2['gender'].map(gender_to_idx).values
    _full_race_distances = results_2['distance_miles'].values
    _figs_priorppc = []
    for _gender_idx, _gender in enumerate(unique_genders):
        _fig, _axes = plt.subplots(2, 4, figsize=(16, 8))
        _axes = _axes.flatten()
        _gender_mask = _full_gender_indices == _gender_idx
        for col_idx, _distance_tuple in enumerate(standard_distances):
            _dist_value, _name, _tolerance = _distance_tuple
            _ax = _axes[col_idx]
            _obs_mask = np.abs(_full_race_distances - _dist_value) < _tolerance
            _combined_mask = _obs_mask & _gender_mask
            if not _combined_mask.any():
                _ax.axis('off')
                _ax.set_title(f'{_name}\nNo data', fontsize=10)
                continue
            _obs_times_binned = _full_observed_times[_combined_mask]
            n_prior_samples = min(len(_obs_times_binned) * 2, 5000)
            random_draws = np.random.choice(_pace_marathon_prior.shape[0] * _pace_marathon_prior.shape[1], n_prior_samples)
            random_courses = np.random.choice(n_courses, n_prior_samples)
            _pace_marathon_flat = _pace_marathon_prior.reshape(-1, n_genders)[:, _gender_idx]
            _pace_distance_effect_flat = _pace_distance_effect_prior.reshape(-1, n_genders)[:, _gender_idx]
            _log_distance_ratio = np.log(_dist_value / reference_distance_1)
            log_prior_pace = _pace_marathon_flat[random_draws] + _pace_distance_effect_flat[random_draws] * _log_distance_ratio + baseline_flat[random_draws, random_courses]
            log_prior_pace_with_noise = np.random.normal(log_prior_pace, _finish_time_noise_prior[random_draws])
            prior_times = np.exp(log_prior_pace_with_noise) * _dist_value
            _combined_data = np.concatenate([_obs_times_binned, prior_times])
            _xlim_max = np.percentile(_combined_data, 99)
            if col_idx == 0:
                _ax.hist(_obs_times_binned, bins='auto', histtype='step', linewidth=2, label='Observed', density=True, color='steelblue')
                _ax.hist(prior_times, bins='auto', histtype='step', linewidth=2, label='Prior', density=True, color='orange')
            else:
                _ax.hist(_obs_times_binned, bins='auto', histtype='step', linewidth=2, density=True, color='steelblue')
                _ax.hist(prior_times, bins='auto', histtype='step', linewidth=2, density=True, color='orange')
            _ax.set_xlabel('Time (min)', fontsize=10)
            _ax.set_ylabel('Density', fontsize=10)
            _ax.set_title(f'{_name} ({_dist_value:.1f}mi)\n{len(_obs_times_binned):,} finishers', fontsize=10)
            _ax.tick_params(labelsize=9)
            _ax.set_xlim(0, _xlim_max)
        _handles, _labels = _axes[0].get_legend_handles_labels()
        _fig.legend(_handles, _labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=10, frameon=False)
        _fig.suptitle(f'Prior Predictive Check: {_gender_display_names[_gender]} Finish Times', fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        _figs_priorppc.append(_fig)
    mo.vstack(_figs_priorppc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Geometry Check
    """)
    return


@app.cell
def _(plt, prior_pred):
    # PRE-SAMPLING GEOMETRY DIAGNOSTIC
    # Comprehensive check for sampling issues using ArviZ
    # Detects: funnels, high correlations, poor prior specification
    course_finish_time_multiplier_std_prior_samples = prior_pred.prior['course_finish_time_multiplier_std'].values.flatten()
    # ============================================================================
    # GLOBAL FUNNEL CHECK: Hyperprior vs Aggregate Course Variance
    course_finish_time_multiplier_raw_prior = prior_pred.prior['course_finish_time_multiplier_raw'].values
    # Check if the scale parameter (hyperprior) correlates with the variance of ALL raw course parameters
    # This is the RIGHT way to check for funnel geometry in hierarchical models
    course_finish_time_multiplier_raw_var = course_finish_time_multiplier_raw_prior.std(axis=-1).flatten()
    _fig, _ax = plt.subplots(1, 1, figsize=(10, 10))
    _ax.scatter(course_finish_time_multiplier_std_prior_samples, course_finish_time_multiplier_raw_var, alpha=0.3, s=10, color='#4a4a4a')
    _ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Expected variance (σ=1 for non-centered)')
    # Compute variance of raw parameters across ALL courses (for each MCMC draw)
    # Shape: (chains, draws, n_courses) -> compute std across courses -> (chains*draws,)
    _ax.set_xlabel('course_finish_time_multiplier_std (hyperprior σ)', fontsize=12, fontweight='bold')
    _ax.set_ylabel('Std(course_finish_time_multiplier_raw) across all courses', fontsize=12, fontweight='bold')
    # Scatter plot: Hyperprior σ vs. Realized variance of raw parameters
    _ax.set_title('Global Funnel Check: Hyperprior vs. Aggregate Course Variance', fontsize=14, fontweight='bold')
    _ax.legend(fontsize=11)
    _ax.grid(alpha=0.3)
    plt.tight_layout()
    # Add reference lines
    _fig
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
    az,
    draws,
    model,
    n_observations,
    notify_mcmc_complete,
    notify_mcmc_error,
    notify_mcmc_start,
    os,
    pm,
    subset_dir,
    target_accept,
    time,
    tune,
):
    # cache path
    cache_file = f'{subset_dir}/tune{tune}_draws{draws}_accept{target_accept}.nc'
    hyperparam_vars = ['pace_marathon', 'pace_distance_effect', 'course_finish_time_multiplier_std', 'finish_time_noise']
    # Hyperparameters to monitor (finish time model)
    if os.path.exists(cache_file):
        print(f'Loading cached trace from {cache_file}')
        trace = az.from_netcdf(cache_file)
        print(az.summary(trace, var_names=hyperparam_vars))
    else:
        print(f'Configuration: tune={tune}, draws={draws}, target_accept={target_accept}')
        print(f'K-core subset: n_observations={n_observations}')
        notify_mcmc_start(model_name='Model 1', n_results=n_observations, draws=draws, target_accept=target_accept)
        _start_time = time.time()
        try:
            with model:
                trace = pm.sample(draws=draws, tune=tune, chains=4, cores=4, target_accept=target_accept, random_seed=42, return_inferencedata=True, idata_kwargs={'log_likelihood': False}, nuts_sampler='nutpie', progressbar=True, compute_convergence_checks=False)
            _elapsed_time = time.time() - _start_time
            print(az.summary(trace, var_names=hyperparam_vars))
            print(f'Saving trace to {cache_file}...')
            trace.to_netcdf(cache_file)
            effective_draws = trace.posterior.dims['draw']
            divergences = trace.sample_stats.diverging.sum().values
            notify_mcmc_complete(model_name='Model 1', elapsed_time=_elapsed_time, n_results=n_observations, effective_draws=effective_draws, divergences=divergences, n_chains=4)
            print(f'MCMC completed in {_elapsed_time / 60:.2f} minutes')
        except Exception as e:
            _elapsed_time = time.time() - _start_time
            notify_mcmc_error(model_name='Model 1', error_msg=str(e), elapsed_time=_elapsed_time)
            print(f'❌ MCMC failed after {_elapsed_time / 60:.2f} minutes')
            raise
    return hyperparam_vars, trace


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Traceplot
    """)
    return


@app.cell
def _(
    az,
    draws,
    hyperparam_vars,
    mo,
    np,
    plt,
    subset_dir,
    target_accept,
    trace,
    tune,
):
    # traceplot for hyperparameters
    az.plot_trace(trace, var_names=hyperparam_vars, compact=True, figsize=(12, 10))
    plt.suptitle('MCMC Traces: Course-Level Difficulty Model Hyperparameters', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # save traceplot
    traceplot_file = f'{subset_dir}/tune{tune}_draws{draws}_accept{target_accept}_traceplot.png'
    plt.savefig(traceplot_file, dpi=300, bbox_inches='tight')
    print(f"Saved traceplot to {traceplot_file}")

    _hp_trace_fig = plt.gcf()

    # traceplot for sample of course difficulty parameters
    course_coords = list(trace.posterior.coords['course'].values)
    sample_course_ids = np.random.choice(course_coords, size=min(20, len(course_coords)), replace=False)

    az.plot_trace(
        trace,
        var_names=['course_finish_time_multiplier'],
        coords={'course': sample_course_ids},
        compact=True,
        figsize=(12, 20)
    )
    plt.suptitle('MCMC Traces: Sample Course Finish Time Multiplier Parameters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _course_trace_fig = plt.gcf()
    mo.vstack([_hp_trace_fig, _course_trace_fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Posterior Diagnostics

    Comprehensive diagnostic plots to check if the model converged
    """)
    return


@app.cell
def _(
    draws,
    hyperparam_vars,
    plot_posterior_diagnostics,
    subset_dir,
    target_accept,
    trace,
    tune,
):
    plot_posterior_diagnostics(trace, hyperparam_vars, subset_dir, tune, draws, target_accept)
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
    return (post_pred,)


@app.cell
def _(
    gender_indices,
    gender_to_idx,
    mo,
    np,
    observed_times,
    plt,
    post_pred,
    race_distances,
    results_2,
    standard_distances,
    unique_genders,
):
    post_pred_times = post_pred.posterior_predictive['finish_times'].values
    post_pred_times_flat = post_pred_times.reshape(-1, post_pred_times.shape[-1])
    _gender_display_names = {'M': 'Male', 'F': 'Female'}
    _full_observed_times = results_2['time_ms'].values / 60000
    _full_gender_indices = results_2['gender'].map(gender_to_idx).values
    _full_race_distances = results_2['distance_miles'].values
    kcore_observed_times = observed_times
    kcore_gender_indices = gender_indices
    kcore_race_distances = race_distances
    _figs_postppc = []
    for _gender_idx, _gender in enumerate(unique_genders):
        _fig, _axes = plt.subplots(2, 4, figsize=(16, 8))
        _axes = _axes.flatten()
        _gender_mask = _full_gender_indices == _gender_idx
        kcore_gender_mask = kcore_gender_indices == _gender_idx
        for _idx, _distance_tuple in enumerate(standard_distances):
            _dist_value, _name, _tolerance = _distance_tuple
            _obs_mask = np.abs(_full_race_distances - _dist_value) < _tolerance
            _combined_mask = _obs_mask & _gender_mask
            if not _combined_mask.any():
                _axes[_idx].axis('off')
                _axes[_idx].set_title(f'{_name}\nNo data', fontsize=9)
                continue
            _obs_times_binned = _full_observed_times[_combined_mask]
            kcore_obs_mask = np.abs(kcore_race_distances - _dist_value) < _tolerance
            kcore_combined_mask = kcore_obs_mask & kcore_gender_mask
            kcore_obs_times_binned = kcore_observed_times[kcore_combined_mask]
            model_obs_mask = np.abs(race_distances - _dist_value) < _tolerance
            model_gender_mask = gender_indices == _gender_idx
            model_combined_mask = model_obs_mask & model_gender_mask
            if not model_combined_mask.any():
                pred_times_flat = np.array([])
            else:
                pred_times_binned = post_pred_times_flat[:, model_combined_mask]
                pred_times_flat = pred_times_binned.flatten()
            if len(pred_times_flat) > 0:
                _combined_data = np.concatenate([_obs_times_binned, pred_times_flat, kcore_obs_times_binned])
            else:
                _combined_data = np.concatenate([_obs_times_binned, kcore_obs_times_binned])
            _xlim_max = np.percentile(_combined_data, 99)
            if _idx == 0:
                _axes[_idx].hist(_obs_times_binned, bins='auto', histtype='step', linewidth=2, label=f'Observed (All, n={len(_obs_times_binned):,})', density=True, color='steelblue')
                _axes[_idx].hist(kcore_obs_times_binned, bins='auto', histtype='step', linewidth=2, label=f'Observed (K-Core, n={len(kcore_obs_times_binned):,})', density=True, color='green')
                if len(pred_times_flat) > 0:
                    _axes[_idx].hist(pred_times_flat, bins='auto', histtype='step', linewidth=2, label='Predicted', density=True, color='orange')
            else:
                _axes[_idx].hist(_obs_times_binned, bins='auto', histtype='step', linewidth=2, density=True, color='steelblue')
                _axes[_idx].hist(kcore_obs_times_binned, bins='auto', histtype='step', linewidth=2, density=True, color='green')
                if len(pred_times_flat) > 0:
                    _axes[_idx].hist(pred_times_flat, bins='auto', histtype='step', linewidth=2, density=True, color='orange')
            _axes[_idx].set_xlabel('Time (min)', fontsize=9)
            _axes[_idx].set_ylabel('Density', fontsize=9)
            _axes[_idx].set_title(f'{_name} ({_dist_value:.1f}mi)\nAll: {len(_obs_times_binned):,} | K-Core: {len(kcore_obs_times_binned):,}', fontsize=9)
            _axes[_idx].tick_params(labelsize=8)
            _axes[_idx].set_xlim(0, _xlim_max)
        _handles, _labels = _axes[0].get_legend_handles_labels()
        _fig.legend(_handles, _labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=10, frameon=False)
        _fig.suptitle(f'Posterior Predictive Check: {_gender_display_names[_gender]}', fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        _figs_postppc.append(_fig)
    mo.vstack(_figs_postppc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parameter Visualizations
    """)
    return


@app.cell
def _(
    np,
    plt,
    reference_distance_1,
    standard_distances,
    trace,
    unique_genders,
):
    pace_marathon_samples = trace.posterior['pace_marathon'].values
    pace_distance_effect_samples = trace.posterior['pace_distance_effect'].values
    distances = np.linspace(5, 100, 200)
    _fig, _ax = plt.subplots(1, 1, figsize=(14, 8))
    gender_colors = {'M': 'steelblue', 'F': 'coral'}
    gender_labels = {'M': 'Male', 'F': 'Female'}
    all_pace_medians = {}
    for _gender_idx, _gender in enumerate(unique_genders):
        _pace_marathon_flat = pace_marathon_samples[:, :, _gender_idx].flatten()
        _pace_distance_effect_flat = pace_distance_effect_samples[:, :, _gender_idx].flatten()
        log_pace_curves = []
        for pace_base, dist_effect in zip(_pace_marathon_flat, _pace_distance_effect_flat):
            log_pace = pace_base + dist_effect * np.log(distances / reference_distance_1)
            log_pace_curves.append(np.exp(log_pace))
        log_pace_curves = np.array(log_pace_curves)
        all_pace_medians[_gender] = {'median': np.percentile(log_pace_curves, 50, axis=0), 'p5': np.percentile(log_pace_curves, 5, axis=0), 'p25': np.percentile(log_pace_curves, 25, axis=0), 'p75': np.percentile(log_pace_curves, 75, axis=0), 'p95': np.percentile(log_pace_curves, 95, axis=0)}
    all_median_values = np.concatenate([all_pace_medians[g]['median'] for g in unique_genders])
    y_min = all_median_values.min() - 0.5
    y_max = all_median_values.max() + 1.0
    _ax.set_ylim(y_min, y_max)
    for _gender_idx, _gender in enumerate(unique_genders):
        pace_data = all_pace_medians[_gender]
        pace_median = pace_data['median']
        pace_5 = pace_data['p5']
        pace_25 = pace_data['p25']
        pace_75 = pace_data['p75']
        pace_95 = pace_data['p95']
        color = gender_colors[_gender]
        _ax.fill_between(distances, pace_5, pace_95, alpha=0.15, color=color)
        _ax.fill_between(distances, pace_25, pace_75, alpha=0.25, color=color)
        _ax.plot(distances, pace_median, color=color, linewidth=2.5)
        for _dist_value, _name, _ in standard_distances:
            _idx = np.argmin(np.abs(distances - _dist_value))
            pace = pace_median[_idx]
            _ax.plot(_dist_value, pace, 'o', color=color, markersize=6, zorder=5, alpha=0.8)
            y_offset = 0.15 if _gender == 'M' else -0.4
            label_text = f'{pace:.1f}'
            _ax.text(_dist_value, pace + y_offset, label_text, fontsize=7, ha='center', va='bottom' if _gender == 'M' else 'top', fontweight='bold', color=color)
    y_label_position = y_max - 0.2
    for _dist_value, _name, _ in standard_distances:
        _ax.axvline(x=_dist_value, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        _ax.text(_dist_value, y_label_position, _name, fontsize=8, ha='center', va='top', rotation=0, alpha=0.6)
    _ax.set_xlabel('Distance (miles)', fontsize=13, fontweight='bold')
    _ax.set_ylabel('Pace (min/mile)', fontsize=13, fontweight='bold')
    _ax.set_title('Learned Pace-Distance Curves by Gender (Population Average)', fontsize=15, fontweight='bold', pad=20)
    _ax.grid(alpha=0.3, linestyle='--')
    _ax.set_xlim(5, 100)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(az, np, plt, trace):
    # Forest plot of course difficulty (ordered by median)
    # Extract course finish time multiplier values and compute medians for sorting
    course_finish_time_multiplier_samples = trace.posterior['course_finish_time_multiplier'].values  # Shape: (chains, draws, n_courses)
    course_names = list(trace.posterior.coords['course'].values)
    medians = np.median(course_finish_time_multiplier_samples.reshape(-1, len(course_names)), axis=0)
    # Calculate median for each course
    _sorted_indices = np.argsort(medians)[::-1]
    _n_show = 20
    # Create sorted indices (most difficult = highest positive value)
    _top_indices = _sorted_indices[:_n_show]  # Descending order
    _bottom_indices = _sorted_indices[-_n_show:]
    # Show top 20 hardest and bottom 30 easiest courses
    _selected_indices = np.concatenate([_top_indices, _bottom_indices])
    selected_courses = [course_names[i] for i in _selected_indices]
    _fig = az.plot_forest(trace, var_names=['course_finish_time_multiplier'], coords={'course': selected_courses}, combined=True, figsize=(10, 12), ess=False, r_hat=False)
    plt.axvline(x=0, linestyle='--', linewidth=1.5, alpha=0.5, label='Average Difficulty')
    plt.xlabel('Course Difficulty (log pace adjustment)', fontsize=12, fontweight='bold')
    plt.title(f'Course Difficulty Rankings\nTop {_n_show} Hardest & Bottom {_n_show} Easiest', fontsize=14, fontweight='bold', pad=15)
    # Create forest plot
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    # Add zero reference line
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MAP Inference

    Extend inference to the full dataset using learned hyperparameters from k-core MCMC.

    The k-core subsetting approach provides high-quality estimates of **population-level hyperparameters** by focusing on densely-connected runners and courses. However, many courses appear only in the sparse closure (runners or courses with few observations).

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

    Extract fixed hyperparameters from k-core MCMC and prepare the full dataset for MAP estimation.
    """)
    return


@app.cell
def _(trace):
    # Extract posterior medians from k-core MCMC trace
    # These hyperparameters will be FIXED in the MAP model
    pace_marathon_fixed = trace.posterior["pace_marathon"].median(dim=["chain", "draw"]).values
    pace_distance_effect_fixed = trace.posterior["pace_distance_effect"].median(dim=["chain", "draw"]).values
    course_finish_time_multiplier_std_fixed = float(
        trace.posterior["course_finish_time_multiplier_std"].median(dim=["chain", "draw"]).values
    )
    finish_time_noise_fixed = trace.posterior["finish_time_noise"].median(dim=["chain", "draw"]).values

    print("Fixed hyperparameters from k-core MCMC:")
    print("=" * 60)
    print(f"Finish Time Model:")
    print(f"  pace_marathon (M, F):                    [{pace_marathon_fixed[0]:.4f}, {pace_marathon_fixed[1]:.4f}]")
    print(f"  pace_distance_effect (M, F):             [{pace_distance_effect_fixed[0]:.4f}, {pace_distance_effect_fixed[1]:.4f}]")
    print(f"  course_finish_time_multiplier_std:       {course_finish_time_multiplier_std_fixed:.4f}")
    print(f"  finish_time_noise (M, F):                [{finish_time_noise_fixed[0]:.4f}, {finish_time_noise_fixed[1]:.4f}]")
    print("=" * 60)
    return (
        course_finish_time_multiplier_std_fixed,
        finish_time_noise_fixed,
        pace_distance_effect_fixed,
        pace_marathon_fixed,
    )


@app.cell
def _(gender_to_idx, n_courses, n_observations, results_2):
    unique_courses_full = results_2['name'].unique()
    course_to_idx_full = {course: _idx for _idx, course in enumerate(unique_courses_full)}
    results_2['course_idx_full'] = results_2['name'].map(course_to_idx_full)
    results_2['gender_idx'] = results_2['gender'].map(gender_to_idx)
    n_courses_full = len(unique_courses_full)
    n_observations_full = len(results_2)
    course_indices_full = results_2['course_idx_full'].values
    gender_indices_full = results_2['gender_idx'].values
    race_distances_full = results_2['distance_miles'].values
    observed_times_full = results_2['time_ms'].values / 60000
    print('\nDataset comparison:')
    print('=' * 60)
    print(f'K-core subset (MCMC):')
    print(f'  Observations: {n_observations:,}')
    print(f'  Courses:      {n_courses:,}')
    print(f'\nFull dataset (MAP):')
    print(f'  Observations: {n_observations_full:,}')
    print(f'  Courses:      {n_courses_full:,}')
    print(f'\nClosure (new entities):')
    print(f'  Observations: {n_observations_full - n_observations:,} (+{100 * (n_observations_full - n_observations) / n_observations:.1f}%)')
    print(f'  Courses:      {n_courses_full - n_courses:,} (+{100 * (n_courses_full - n_courses) / n_courses:.1f}%)')
    print('=' * 60)
    return (
        course_indices_full,
        gender_indices_full,
        n_courses_full,
        n_observations_full,
        observed_times_full,
        race_distances_full,
        unique_courses_full,
    )


@app.cell
def _(model_data):
    # Verify k-core flags exist in model_data
    # These flags were created by subset_kcore_data() function
    print("K-core subsetting flags:")
    print("=" * 60)
    print(f"'in_kcore' column exists:   {'in_kcore' in model_data.columns}")
    print(f"'in_closure' column exists: {'in_closure' in model_data.columns}")

    if 'in_kcore' in model_data.columns and 'in_closure' in model_data.columns:
        print(f"\nK-core composition in MCMC subset:")
        print(f"  K-core results:   {model_data['in_kcore'].sum():,} ({100*model_data['in_kcore'].mean():.1f}%)")
        print(f"  Closure results:  {model_data['in_closure'].sum():,} ({100*model_data['in_closure'].mean():.1f}%)")
        print(f"\nNote: K-core means BOTH runner AND course are in dense core")
        print(f"      Closure means at least ONE entity is from sparse periphery")
    print("=" * 60)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Specification

    Build a PyMC model with the same structure as the MCMC model, but with hyperparameters fixed to their posterior medians. Only course-level parameters will be estimated.
    """)
    return


@app.cell
def _(
    course_finish_time_multiplier_std_fixed,
    course_indices_full,
    finish_time_noise_fixed,
    gender_indices_full,
    mo,
    n_courses,
    n_courses_full,
    n_observations_full,
    observed_times_full,
    pace_distance_effect_fixed,
    pace_marathon_fixed,
    pm,
    race_distances_full,
    reference_distance_1,
    unique_courses_full,
    unique_genders,
):
    coords_full = {'course': unique_courses_full, 'gender': unique_genders, 'finishers': range(n_observations_full)}
    with pm.Model(coords=coords_full) as model_map:
        _pace_marathon = pm.Data('pace_marathon', pace_marathon_fixed, dims='gender')
        _pace_distance_effect = pm.Data('pace_distance_effect', pace_distance_effect_fixed, dims='gender')
        _course_finish_time_multiplier_std = pm.Data('course_finish_time_multiplier_std', course_finish_time_multiplier_std_fixed)
        _finish_time_noise = pm.Data('finish_time_noise', finish_time_noise_fixed, dims='gender')
        _course_finish_time_multiplier_raw = pm.Normal('course_finish_time_multiplier_raw', mu=0, sigma=1, dims='course')
        _course_finish_time_multiplier = pm.Deterministic('course_finish_time_multiplier', _course_finish_time_multiplier_std * _course_finish_time_multiplier_raw, dims='course')
        _log_distance_ratio = pm.math.log(race_distances_full / reference_distance_1)
        _expected_log_pace = _pace_marathon[gender_indices_full] + _pace_distance_effect[gender_indices_full] * _log_distance_ratio + _course_finish_time_multiplier[course_indices_full]
        _expected_pace = pm.Deterministic('expected_pace', pm.math.exp(_expected_log_pace), dims='finishers')
        _expected_time = _expected_pace * race_distances_full
        pm.Normal('finish_times', mu=_expected_time, sigma=_finish_time_noise[gender_indices_full] * race_distances_full, observed=observed_times_full, dims='finishers')
    print('\nMAP Model Summary:')
    print('=' * 60)
    print(f'Total courses:        {n_courses_full:,}')
    print(f'  K-core courses:     {n_courses:,}')
    print(f'  Closure courses:    {n_courses_full - n_courses:,}')
    print(f'\nFixed hyperparameters (from k-core MCMC):')
    print(f'  pace_marathon, pace_distance_effect, course_finish_time_multiplier_std, finish_time_noise')
    print(f'\nFree parameters (to be estimated):')
    print(f'  course_finish_time_multiplier_raw (N = {n_courses_full:,})')
    _graph = pm.model_to_graphviz(model_map)
    mo.Html(
        f'<div style="max-width:100%; overflow-x:auto">{_graph.pipe(format="svg").decode()}</div>'
    )
    return (model_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Estimation

    Run fast optimization to find the mode of the posterior distribution (Maximum A Posteriori estimate). Unlike MCMC sampling, this produces point estimates only, but completes in seconds/minutes rather than hours.
    """)
    return


@app.cell
def _(az, model_map, n_courses_full, np, pm, time, unique_courses_full):
    print('🚀 Running MAP estimation on full dataset...')
    print(f'   Fixed hyperparameters from k-core MCMC')
    print(f'   Estimating course_finish_time_multiplier for {n_courses_full:,} courses')
    print()
    _start_time = time.time()
    with model_map:
        map_estimate = pm.find_MAP(method='L-BFGS-B', progressbar=True)
    _elapsed_time = time.time() - _start_time
    print(f'\n✅ MAP estimation completed in {_elapsed_time:.2f} seconds')
    # Convert MAP point estimate to InferenceData format for consistency
    # Create a single-draw posterior (shape: [1 chain, 1 draw, n_courses])
    map_trace = az.from_dict(posterior={'course_finish_time_multiplier': map_estimate['course_finish_time_multiplier'][np.newaxis, np.newaxis, :], 'course_finish_time_multiplier_raw': map_estimate['course_finish_time_multiplier_raw'][np.newaxis, np.newaxis, :]}, coords={'course': unique_courses_full}, dims={'course_finish_time_multiplier': ['course'], 'course_finish_time_multiplier_raw': ['course']})
    return (map_trace,)


@app.cell
def _(map_trace):
    # Extract MAP estimates and display summary statistics
    map_course_finish_time_multiplier = map_trace.posterior['course_finish_time_multiplier'].values[0, 0, :]
    return (map_course_finish_time_multiplier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### K-Core Validation

    Validate MAP estimates by comparing them to MCMC posteriors for courses in the k-core subset. This checks whether the MAP optimization produces reasonable point estimates that align with the full Bayesian posterior.
    """)
    return


@app.cell
def _(map_trace, model_data, np, trace):
    # Extract course finish time multipliers from both traces
    # ALL courses in MCMC trace (both k-core and closure added up to max_entities=6000)
    all_mcmc_courses = list(trace.posterior.coords['course'].values)
    map_course_names_full = list(map_trace.posterior.coords['course'].values)
    # MAP estimates for all courses in MCMC trace
    mcmc_indices_in_map = [map_course_names_full.index(c) for c in all_mcmc_courses]
    map_all_estimates = map_trace.posterior['course_finish_time_multiplier'].values[0, 0, mcmc_indices_in_map]
    mcmc_medians = trace.posterior['course_finish_time_multiplier'].median(dim=['chain', 'draw']).values
    mcmc_q05 = trace.posterior['course_finish_time_multiplier'].quantile(0.05, dim=['chain', 'draw']).values
    # Get MCMC posteriors for all courses
    mcmc_q95 = trace.posterior['course_finish_time_multiplier'].quantile(0.95, dim=['chain', 'draw']).values
    mcmc_q25 = trace.posterior['course_finish_time_multiplier'].quantile(0.25, dim=['chain', 'draw']).values
    mcmc_q75 = trace.posterior['course_finish_time_multiplier'].quantile(0.75, dim=['chain', 'draw']).values
    kcore_course_names_from_data = model_data[model_data['in_kcore']]['name'].unique().tolist()
    closure_course_names_from_data = model_data[model_data['in_closure']]['name'].unique().tolist()
    is_kcore_course = np.array([_name in kcore_course_names_from_data for _name in all_mcmc_courses])
    # CORRECTED: Identify k-core vs closure-only courses from model_data
    # K-core = dense core (152 courses)
    # Closure-only = added entities not in k-core (288 courses)
    is_closure_only_course = ~is_kcore_course
    map_kcore_estimates = map_all_estimates[is_kcore_course]
    map_closure_only_estimates = map_all_estimates[is_closure_only_course]
    # Separate MCMC courses: k-core (in k-core) vs closure-only (not in k-core)
    _kcore_indices = np.where(is_kcore_course)[0]
    kcore_correlation = np.corrcoef(map_kcore_estimates, mcmc_medians[_kcore_indices])[0, 1]  # Closure entities NOT in k-core
    kcore_mae = np.mean(np.abs(map_kcore_estimates - mcmc_medians[_kcore_indices]))
    kcore_rmse = np.sqrt(np.mean((map_kcore_estimates - mcmc_medians[_kcore_indices]) ** 2))
    kcore_within_90 = np.sum((map_kcore_estimates >= mcmc_q05[_kcore_indices]) & (map_kcore_estimates <= mcmc_q95[_kcore_indices]))
    kcore_within_50 = np.sum((map_kcore_estimates >= mcmc_q25[_kcore_indices]) & (map_kcore_estimates <= mcmc_q75[_kcore_indices]))
    # K-Core validation metrics (dense core)
    _closure_only_indices = np.where(is_closure_only_course)[0]
    closure_only_correlation = np.corrcoef(map_closure_only_estimates, mcmc_medians[_closure_only_indices])[0, 1]
    closure_only_mae = np.mean(np.abs(map_closure_only_estimates - mcmc_medians[_closure_only_indices]))
    closure_only_rmse = np.sqrt(np.mean((map_closure_only_estimates - mcmc_medians[_closure_only_indices]) ** 2))
    closure_only_within_90 = np.sum((map_closure_only_estimates >= mcmc_q05[_closure_only_indices]) & (map_closure_only_estimates <= mcmc_q95[_closure_only_indices]))
    closure_only_within_50 = np.sum((map_closure_only_estimates >= mcmc_q25[_closure_only_indices]) & (map_closure_only_estimates <= mcmc_q75[_closure_only_indices]))
    print('\n' + '=' * 80)
    print('MAP vs MCMC Validation (Finish Time Multipliers)')
    print('=' * 80)
    # Closure-Only validation metrics (added entities not in k-core)
    print(f'\nK-Core Courses (Dense Core, n={len(map_kcore_estimates):,}):')
    print(f'  Correlation (r):        {kcore_correlation:.4f}')
    print(f'  MAE:                    {kcore_mae:.4f}')
    print(f'  RMSE:                   {kcore_rmse:.4f}')
    print(f'  Within 90% CI:          {kcore_within_90}/{len(map_kcore_estimates)} ({100 * kcore_within_90 / len(map_kcore_estimates):.1f}%)')
    print(f'  Within 50% CI:          {kcore_within_50}/{len(map_kcore_estimates)} ({100 * kcore_within_50 / len(map_kcore_estimates):.1f}%)')
    print(f'\nClosure-Only Courses (Added Entities, n={len(map_closure_only_estimates):,}):')
    print(f'  Correlation (r):        {closure_only_correlation:.4f}')
    print(f'  MAE:                    {closure_only_mae:.4f}')
    print(f'  RMSE:                   {closure_only_rmse:.4f}')
    print(f'  Within 90% CI:          {closure_only_within_90}/{len(map_closure_only_estimates)} ({100 * closure_only_within_90 / len(map_closure_only_estimates):.1f}%)')
    print(f'  Within 50% CI:          {closure_only_within_50}/{len(map_closure_only_estimates)} ({100 * closure_only_within_50 / len(map_closure_only_estimates):.1f}%)')
    print('=' * 80)
    return (
        all_mcmc_courses,
        closure_only_correlation,
        closure_only_mae,
        closure_only_rmse,
        is_closure_only_course,
        is_kcore_course,
        kcore_correlation,
        kcore_course_names_from_data,
        kcore_mae,
        kcore_rmse,
        map_all_estimates,
        map_closure_only_estimates,
        map_kcore_estimates,
        mcmc_medians,
        mcmc_q05,
        mcmc_q25,
        mcmc_q75,
        mcmc_q95,
    )


@app.cell
def _(
    all_mcmc_courses,
    closure_only_correlation,
    closure_only_mae,
    closure_only_rmse,
    is_closure_only_course,
    is_kcore_course,
    kcore_correlation,
    kcore_mae,
    kcore_rmse,
    map_all_estimates,
    map_closure_only_estimates,
    map_kcore_estimates,
    mcmc_medians,
    mcmc_q05,
    mcmc_q25,
    mcmc_q75,
    mcmc_q95,
    np,
    plt,
):
    # Scatter plot: MAP vs MCMC with credible intervals - COURSE FINISH TIME MULTIPLIER
    # K-core (dense core) vs Closure-only (added entities) courses
    _fig, _ax = plt.subplots(1, 1, figsize=(14, 14))
    mcmc_kcore_medians = mcmc_medians[is_kcore_course]
    kcore_q25 = mcmc_q25[is_kcore_course]
    # K-CORE COURSES: Dense core with credible intervals
    kcore_q75 = mcmc_q75[is_kcore_course]
    kcore_q05 = mcmc_q05[is_kcore_course]
    kcore_q95 = mcmc_q95[is_kcore_course]
    _ax.errorbar(map_kcore_estimates, mcmc_kcore_medians, xerr=None, yerr=[mcmc_kcore_medians - kcore_q25, kcore_q75 - mcmc_kcore_medians], fmt='o', alpha=0.4, color='steelblue', markersize=4, elinewidth=1.5, capsize=0, label='K-Core 50% CI')
    _ax.errorbar(map_kcore_estimates, mcmc_kcore_medians, xerr=None, yerr=[mcmc_kcore_medians - kcore_q05, kcore_q95 - mcmc_kcore_medians], fmt='o', alpha=0.2, color='steelblue', markersize=4, elinewidth=0.8, capsize=0, label='K-Core 90% CI')
    _ax.scatter(map_kcore_estimates, mcmc_kcore_medians, s=30, color='steelblue', alpha=0.6, zorder=5, label=f'K-Core Courses (n={sum(is_kcore_course):,})')
    mcmc_closure_only_medians = mcmc_medians[is_closure_only_course]
    closure_only_q25 = mcmc_q25[is_closure_only_course]
    closure_only_q75 = mcmc_q75[is_closure_only_course]
    closure_only_q05 = mcmc_q05[is_closure_only_course]
    closure_only_q95 = mcmc_q95[is_closure_only_course]
    _ax.errorbar(map_closure_only_estimates, mcmc_closure_only_medians, xerr=None, yerr=[mcmc_closure_only_medians - closure_only_q25, closure_only_q75 - mcmc_closure_only_medians], fmt='^', alpha=0.3, color='limegreen', markersize=3, elinewidth=1.0, capsize=0, label='Closure-Only 50% CI')
    _ax.errorbar(map_closure_only_estimates, mcmc_closure_only_medians, xerr=None, yerr=[mcmc_closure_only_medians - closure_only_q05, closure_only_q95 - mcmc_closure_only_medians], fmt='^', alpha=0.15, color='limegreen', markersize=3, elinewidth=0.6, capsize=0, label='Closure-Only 90% CI')
    _ax.scatter(map_closure_only_estimates, mcmc_closure_only_medians, s=20, color='limegreen', alpha=0.5, zorder=4, marker='^', label=f'Closure-Only Courses (n={sum(is_closure_only_course):,})')
    lims = [min(map_all_estimates.min(), mcmc_medians.min()) - 0.1, max(map_all_estimates.max(), mcmc_medians.max()) + 0.1]
    _ax.plot(lims, lims, 'r--', linewidth=2, alpha=0.7, label='Perfect Agreement (y=x)')
    from scipy.stats import linregress
    kcore_slope, kcore_intercept, _, _, _ = linregress(map_kcore_estimates, mcmc_kcore_medians)
    kcore_fit_x = np.array([map_kcore_estimates.min(), map_kcore_estimates.max()])
    kcore_fit_y = kcore_slope * kcore_fit_x + kcore_intercept
    _ax.plot(kcore_fit_x, kcore_fit_y, color='steelblue', linewidth=2, alpha=0.7, linestyle=':', label=f'K-Core Fit (slope={kcore_slope:.3f})')
    if len(map_closure_only_estimates) >= 2:
        closure_only_slope, closure_only_intercept, _, _, _ = linregress(map_closure_only_estimates, mcmc_closure_only_medians)
        closure_only_fit_x = np.array([map_closure_only_estimates.min(), map_closure_only_estimates.max()])
        closure_only_fit_y = closure_only_slope * closure_only_fit_x + closure_only_intercept
        _ax.plot(closure_only_fit_x, closure_only_fit_y, color='limegreen', linewidth=2, alpha=0.7, linestyle=':', label=f'Closure-Only Fit (slope={closure_only_slope:.3f})')
    # CLOSURE-ONLY COURSES: Added entities with credible intervals
    _ax.text(0.05, 0.95, f'K-Core:\n  r = {kcore_correlation:.4f}\n  MAE = {kcore_mae:.4f}\n  RMSE = {kcore_rmse:.4f}', transform=_ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='steelblue', alpha=0.4))
    _ax.text(0.95, 0.95, f'Closure-Only:\n  r = {closure_only_correlation:.4f}\n  MAE = {closure_only_mae:.4f}\n  RMSE = {closure_only_rmse:.4f}', transform=_ax.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='limegreen', alpha=0.4))
    _ax.set_xlabel('MAP Estimate (Full Dataset)', fontsize=12, fontweight='bold')
    _ax.set_ylabel('MCMC Posterior Median (K-Core Subset)', fontsize=12, fontweight='bold')
    _ax.set_title('Validation: MAP vs MCMC Course Finish Time Multipliers\n(K-Core Dense Core vs Closure-Only Added Entities)', fontsize=14, fontweight='bold')
    _ax.legend(loc='lower right', fontsize=9, ncol=2)
    _ax.grid(alpha=0.3)
    _ax.set_xlim(lims)
    _ax.set_ylim(lims)
    plt.tight_layout()
    kcore_outliers_mask = (map_kcore_estimates < kcore_q05) | (map_kcore_estimates > kcore_q95)
    if kcore_outliers_mask.any():
        print(f'\n⚠️  K-Core Outliers ({kcore_outliers_mask.sum()} courses where MAP is outside 90% CI):')
        print('=' * 80)
        kcore_course_names = [all_mcmc_courses[i] for i in range(len(all_mcmc_courses)) if is_kcore_course[i]]
        outlier_indices = np.where(kcore_outliers_mask)[0]
        for _idx in outlier_indices[:10]:
            _course_name = kcore_course_names[_idx]
            map_val = map_kcore_estimates[_idx]
            mcmc_val = mcmc_kcore_medians[_idx]
            ci_lower = kcore_q05[_idx]
            ci_upper = kcore_q95[_idx]
            print(f'  {_course_name[:60]:<60} MAP: {map_val:+.3f}  MCMC: {mcmc_val:+.3f} [{ci_lower:+.3f}, {ci_upper:+.3f}]')
        if kcore_outliers_mask.sum() > 10:
    # Add y=x reference line (perfect agreement)
            print(f'  ... and {kcore_outliers_mask.sum() - 10} more')
    else:
    # Fit lines for each group
    # Add correlation and MAE annotations
    # Identify outliers (MAP outside 90% credible interval) for k-core courses
        print('\n✅ No k-core outliers: All MAP estimates within MCMC 90% credible intervals!')  # Show first 10
    _fig
    return


@app.cell
def _(
    is_closure_only_course,
    is_kcore_course,
    map_closure_only_estimates,
    map_kcore_estimates,
    mcmc_medians,
    np,
    plt,
):
    from scipy import stats
    _fig, _axes = plt.subplots(1, 2, figsize=(16, 6))
    _fig.suptitle('MAP vs MCMC', fontsize=16, fontweight='bold', y=1.0)
    _ax = _axes[0]
    _kcore_indices = np.where(is_kcore_course)[0]
    kcore_map = map_kcore_estimates
    kcore_mcmc = mcmc_medians[_kcore_indices]
    kcore_differences = 100 * (kcore_map - kcore_mcmc) / kcore_mcmc
    kcore_p2_5 = np.percentile(kcore_differences, 2.5)
    kcore_p97_5 = np.percentile(kcore_differences, 97.5)
    kcore_xlim = max(abs(kcore_p2_5), abs(kcore_p97_5))
    kcore_bins = np.linspace(-kcore_xlim, kcore_xlim, 50)
    _ax.hist(kcore_differences, bins=kcore_bins, alpha=0.7, histtype='step', linewidth=2, color='steelblue', edgecolor='navy')
    _ax.axvline(x=0, linestyle='--', color='red', linewidth=2, label='Perfect Agreement', alpha=0.7)
    kcore_mean_diff = kcore_differences.mean()
    kcore_std_diff = kcore_differences.std()
    kcore_median_diff = np.median(kcore_differences)
    kcore_mae_1 = np.mean(np.abs(kcore_differences))
    stats_text = f'K-Core Stats (n={len(kcore_differences)}):\n  Mean: {kcore_mean_diff:+.2f}%\n  Median: {kcore_median_diff:+.2f}%\n  MAE: {kcore_mae_1:.2f}%\n  Std: {kcore_std_diff:.2f}%'
    _ax.text(0.98, 0.98, stats_text, transform=_ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    _ax.set_xlabel('% Difference', fontsize=12, fontweight='bold')
    _ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    _ax.set_title('K-Core', fontsize=13, fontweight='bold')
    _ax.set_xlim(-kcore_xlim, kcore_xlim)
    _ax.legend(fontsize=11)
    _ax.grid(alpha=0.3)
    _ax = _axes[1]
    _closure_only_indices = np.where(is_closure_only_course)[0]
    if len(_closure_only_indices) > 0:
        closure_only_map = map_closure_only_estimates
        closure_only_mcmc = mcmc_medians[_closure_only_indices]
        closure_only_differences = 100 * (closure_only_map - closure_only_mcmc) / closure_only_mcmc
        closure_p2_5 = np.percentile(closure_only_differences, 2.5)
        closure_p97_5 = np.percentile(closure_only_differences, 97.5)
        closure_xlim = max(abs(closure_p2_5), abs(closure_p97_5))
        closure_bins = np.linspace(-closure_xlim, closure_xlim, 50)
        _ax.hist(closure_only_differences, bins=closure_bins, alpha=0.7, histtype='step', linewidth=2, color='limegreen', edgecolor='darkgreen')
        _ax.axvline(x=0, linestyle='--', color='red', linewidth=2, label='Perfect Agreement', alpha=0.7)
        closure_mean_diff = closure_only_differences.mean()
        closure_std_diff = closure_only_differences.std()
        closure_median_diff = np.median(closure_only_differences)
        closure_mae = np.mean(np.abs(closure_only_differences))
        stats_text = f'Closure-Only Stats (n={len(closure_only_differences)}):\n  Mean: {closure_mean_diff:+.2f}%\n  Median: {closure_median_diff:+.2f}%\n  MAE: {closure_mae:.2f}%\n  Std: {closure_std_diff:.2f}%'
        _ax.text(0.98, 0.98, stats_text, transform=_ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
        _ax.set_xlim(-closure_xlim, closure_xlim)
        _ax.legend(fontsize=11)
    else:
        _ax.text(0.5, 0.5, 'No closure-only courses', transform=_ax.transAxes, ha='center', va='center', fontsize=14)
    _ax.set_xlabel('% Difference', fontsize=12, fontweight='bold')
    _ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    _ax.set_title('Closure-Only', fontsize=13, fontweight='bold')
    _ax.grid(alpha=0.3)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parameter Visualizations

    Visualize course difficulty rankings and distributions using MAP estimates for the complete dataset (k-core + closure courses).
    """)
    return


@app.cell
def _(
    all_mcmc_courses,
    kcore_course_names_from_data,
    map_course_finish_time_multiplier,
    map_trace,
    mcmc_medians,
    mcmc_q05,
    mcmc_q95,
    np,
    plt,
):
    # Forest plot: Top 50 Hardest and Bottom 50 Easiest Courses (Full Dataset)
    # Sort all courses by difficulty (course_finish_time_multiplier)
    all_course_names_map = list(map_trace.posterior.coords['course'].values)
    _sorted_indices = np.argsort(map_course_finish_time_multiplier)[::-1]  # Descending order
    _n_show = 50
    # Select top 50 hardest and bottom 50 easiest
    _top_indices = _sorted_indices[:_n_show]
    _bottom_indices = _sorted_indices[-_n_show:]
    _selected_indices = np.concatenate([_top_indices, _bottom_indices])
    _fig, _ax = plt.subplots(1, 1, figsize=(10, 20))
    selected_multipliers = map_course_finish_time_multiplier[_selected_indices]
    # Create figure manually (arviz.plot_forest doesn't support markers by group)
    selected_names = [all_course_names_map[i] for i in _selected_indices]
    kcore_set = set(kcore_course_names_from_data)
    # Extract finish time multipliers and course names for selected courses
    mcmc_set = set(all_mcmc_courses)
    selected_is_kcore = [all_course_names_map[i] in kcore_set for i in _selected_indices]
    selected_is_closure_only = [all_course_names_map[i] in mcmc_set and all_course_names_map[i] not in kcore_set for i in _selected_indices]
    # Check which courses are in MCMC trace vs MAP-only
    # Three groups:
    # 1. K-core: In MCMC dense core (from kcore_course_names_from_data)
    # 2. Closure-only: In MCMC but not k-core (from all_mcmc_courses but not k-core)
    # 3. MAP-only: NOT in MCMC at all (only in full dataset MAP)
    selected_is_map_only = [all_course_names_map[i] not in mcmc_set for i in _selected_indices]
    selected_mcmc_medians = []  # All MCMC courses
    selected_mcmc_q05 = []
    selected_mcmc_q95 = []
    for _course_name in selected_names:
        if _course_name in all_mcmc_courses:
            _idx = all_mcmc_courses.index(_course_name)
    # Get MCMC posteriors for courses that have them
            selected_mcmc_medians.append(mcmc_medians[_idx])
            selected_mcmc_q05.append(mcmc_q05[_idx])
            selected_mcmc_q95.append(mcmc_q95[_idx])
        else:
            selected_mcmc_medians.append(None)
            selected_mcmc_q05.append(None)
            selected_mcmc_q95.append(None)
    y_positions = np.arange(len(selected_multipliers))
    for i, (multiplier, is_kcore, is_closure, is_map_only) in enumerate(zip(selected_multipliers, selected_is_kcore, selected_is_closure_only, selected_is_map_only)):
        _ax.plot([multiplier], [y_positions[i]], marker='s', color='orange', markersize=8, alpha=0.7, zorder=2, markeredgecolor='darkorange', markeredgewidth=0.5)
        if not is_map_only and selected_mcmc_medians[i] is not None:
            mcmc_median = selected_mcmc_medians[i]
            mcmc_q05_val = selected_mcmc_q05[i]
            mcmc_q95_val = selected_mcmc_q95[i]
    # Plot as horizontal lines with markers
            if is_kcore:
                _ax.errorbar([mcmc_median], [y_positions[i]], xerr=[[mcmc_median - mcmc_q05_val], [mcmc_q95_val - mcmc_median]], fmt='o', color='steelblue', markersize=5, alpha=0.9, elinewidth=1.5, capsize=3, zorder=3, markeredgecolor='navy', markeredgewidth=0.5)
            elif is_closure:
                _ax.errorbar([mcmc_median], [y_positions[i]], xerr=[[mcmc_median - mcmc_q05_val], [mcmc_q95_val - mcmc_median]], fmt='^', color='limegreen', markersize=5, alpha=0.9, elinewidth=1.5, capsize=3, zorder=3, markeredgecolor='darkgreen', markeredgewidth=0.5)
    _ax.axvline(x=0, linestyle='--', color='red', linewidth=2, alpha=0.5, label='Average Difficulty')
    _ax.set_yticks(y_positions)  # Always plot orange square for MAP estimate
    _ax.set_yticklabels(selected_names, fontsize=8)
    _ax.set_xlabel('Course Difficulty (log pace adjustment)', fontsize=12, fontweight='bold')
    _ax.set_title(f'Course Difficulty Rankings (Full Dataset)\nTop {_n_show} Hardest & Bottom {_n_show} Easiest', fontsize=14, fontweight='bold', pad=15)
    _ax.grid(alpha=0.3, axis='x')  # Plot MCMC posterior with credible interval if available
    _ax.invert_yaxis()
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=8, markeredgecolor='darkorange', markeredgewidth=0.5, label=f'All Courses (MAP estimates) - {len(selected_multipliers)} shown'), Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=6, markeredgecolor='navy', markeredgewidth=0.5, label=f'K-Core Dense Core (MCMC + MAP) - {sum(selected_is_kcore)} shown'), Line2D([0], [0], marker='^', color='w', markerfacecolor='limegreen', markersize=6, markeredgecolor='darkgreen', markeredgewidth=0.5, label=f'Closure-Only (MCMC + MAP) - {sum(selected_is_closure_only)} shown'), Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=8, markeredgecolor='darkorange', markeredgewidth=0.5, label=f'MAP-Only (no MCMC) - {sum(selected_is_map_only)} shown'), Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Average Difficulty')]
    _ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    plt.tight_layout()
    # Add zero reference line
    # Configure axes
    # Add legend
    _fig  # K-core: blue circle with credible interval  # Closure-only: green triangle with credible interval  # Hardest at top
    return


if __name__ == "__main__":
    app.run()
