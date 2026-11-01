import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def marimo_import():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    # Masked-race validation

    Can we recover a race's hidden DNF count from its visible finishers and the rest of the dataset? We mask individual race occurrences, not entire calendar years or courses. Other years and sibling distances remain available.

    This is one frozen, stratified 20% holdout with actual refits—not PSIS-LOO. Masked races are known nonreporters; reporting classification is not tested. Counts refer to runners retained by the analysis filters, within the fixed core's convenience sample.
    """)
    return


@app.cell(hide_code=True)
def setup_heading(mo):
    mo.md("""
    ## Setup
    """)
    return


@app.cell(hide_code=True)
def configuration(mo):
    from pathlib import Path
    from datetime import datetime
    import hashlib
    import json
    import numpy as np
    import pandas as pd
    import xarray as xr
    import arviz as az
    import pymc as pm
    import matplotlib.pyplot as plt
    from utils import data_source
    from utils.data_processing import load_results, process_results
    from utils.sampling import sample_lean, load_store, check_memory, netcdf_safe

    ROOT = Path(__file__).resolve().parent.parent
    VALIDATION_ROOT = ROOT / 'data' / 'cache' / '2.4_dnf_validation'
    SMOKE = False
    CORE = 'core' if SMOKE else 'core_3_500'
    SEEDS = {stage: sum(map(ord, f'2.4_dnf_validation:{stage}:v1')) for stage in ['split', 'reporting', 'baseline', 'joint', 'prediction', 'prediction_check', 'bootstrap', 'prior']}
    SETTINGS = {
        'protocol': 'known-nonreporter-individual-race-v1', 'smoke': SMOKE, 'core': CORE,
        'chains': 2 if SMOKE else 4,
        'reporting': {'draws': 50 if SMOKE else 1000, 'tune': 50 if SMOKE else 1000, 'target_accept': 0.95},
        'baseline': {'draws': 50 if SMOKE else 1000, 'tune': 50 if SMOKE else 1000, 'target_accept': 0.9},
        'joint': {'draws': 50 if SMOKE else 1500, 'tune': 50 if SMOKE else 2000, 'target_accept': 0.9},
        'prediction_draws_per_chain': 10 if SMOKE else 500,
        'seeds': SEEDS,
    }
    mo.callout('Smoke execution only: these short chains must not be interpreted.' if SMOKE else 'Full validation: one frozen split; no tuning against held-out outcomes.', kind='warn' if SMOKE else 'info')
    return (
        CORE,
        ROOT,
        SEEDS,
        SETTINGS,
        SMOKE,
        VALIDATION_ROOT,
        az,
        check_memory,
        data_source,
        datetime,
        hashlib,
        load_results,
        np,
        pd,
        plt,
        pm,
        process_results,
        sample_lean,
        xr,
    )


@app.cell(hide_code=True)
def validation_helpers():
    from utils import reporting_model as rm
    from utils import dnf_model as dm
    from utils import masked_validation as mv

    return dm, mv, rm


@app.cell(hide_code=True)
def data_heading(mo):
    mo.md("""
    ## Data and masking audit
    """)
    return


@app.cell(hide_code=True)
def source_data(
    CORE,
    ROOT,
    data_source,
    hashlib,
    load_results,
    mo,
    np,
    pd,
    process_results,
):
    with data_source.connect(local=True) as connection:
        processed_results = process_results(load_results(connection))
    core_names = np.sort(pd.read_parquet(ROOT / 'data' / 'cache' / '0.1_kcore' / f'{CORE}.parquet')['name'].unique())
    with (ROOT / 'data' / 'results.duckdb').open('rb') as snapshot_file:
        snapshot_sha256 = hashlib.file_digest(snapshot_file, 'sha256').hexdigest()
    mo.md(f"Pinned local snapshot `{snapshot_sha256[:12]}`; **{len(core_names)} courses**. The fixed core is a convenience sample; this is not a test of course selection.")
    return core_names, processed_results, snapshot_sha256


@app.cell(hide_code=True)
def masking_audit(
    SEEDS,
    SETTINGS,
    VALIDATION_ROOT,
    core_names,
    data_source,
    hashlib,
    mv,
    pd,
    processed_results,
    rm,
    snapshot_sha256,
):
    with data_source.connect(local=True) as reporting_connection:
        reporting_source = rm.load_reporting_editions(reporting_connection)
    manifest = mv.make_race_manifest(processed_results, core_names, seed=SEEDS['split'])
    heldout_ids = tuple(int(v) for v in manifest.loc[manifest['selected'], 'did'])
    visible_results = mv.mask_results(processed_results, heldout_ids)
    reporting_training = reporting_source.loc[~reporting_source['did'].isin(heldout_ids)].copy()
    baseline_counts = mv.training_counts(visible_results, core_names)
    targets = manifest.loc[manifest['selected']].copy().sort_values('did')
    visible_core = visible_results.loc[visible_results['name'].isin(core_names)].copy()
    runner_order = pd.Index(pd.concat([visible_core.loc[visible_core['finished'], 'participant_id'], visible_core.loc[visible_core['reports_dnfs'], 'participant_id']]).unique())
    cache_settings = {**SETTINGS, 'model_version': {'reporting': 3, 'joint': 2, 'baseline': 1}, 'extraction_version': 1, 'prediction_version': mv.PREDICTION_VERSION,
        'sampler': {k: SETTINGS[k] for k in ['reporting', 'baseline', 'joint']},
        'coordinates': {'courses': core_names.tolist(), 'runners_sha256': hashlib.sha256(pd.util.hash_pandas_object(runner_order, index=False).to_numpy().tobytes()).hexdigest(), 'n_runners': len(runner_order)}}
    experiment_key = mv.validation_fingerprint(visible_core, reporting_training, manifest, cache_settings)
    OUT = VALIDATION_ROOT / experiment_key
    OUT.mkdir(parents=True, exist_ok=True)
    mv.atomic_write_frame(OUT / 'manifest.parquet', manifest)
    mv.atomic_json(OUT / 'provenance.json', {'snapshot_sha256': snapshot_sha256, 'experiment_key': experiment_key, 'settings': cache_settings})
    assert not reporting_training['did'].isin(heldout_ids).any()
    assert not baseline_counts['did'].isin(heldout_ids).any()
    split_audit = manifest.groupby(['distance_bin', 'finisher_bin'], observed=True).agg(eligible=('eligible', 'sum'), masked=('selected', 'sum'))
    print(f"{len(heldout_ids)} masked races; {len(reporting_training)} reporting-stage training rows; cache {experiment_key}")
    split_audit
    return (
        OUT,
        baseline_counts,
        experiment_key,
        heldout_ids,
        reporting_training,
        runner_order,
        targets,
        visible_core,
    )


@app.cell(hide_code=True)
def models_heading(mo):
    mo.md("""
    ## Models and prior predictions

    The joint model retains 2.3 v2’s fixed priors and parameterization. Its masked races contribute finish times but no DNF outcomes and no silence likelihood. The count baseline uses finisher exposure, log distance, partially pooled course effects and negative-binomial overdispersion. Neither model sees the held-out counts.
    """)
    return


@app.cell(hide_code=True)
def initial_models(
    SEEDS,
    baseline_counts,
    mv,
    np,
    pd,
    pm,
    reporting_training,
    rm,
):
    reporting_model = rm.build_reporting_model(reporting_training)
    baseline_model = mv.build_count_baseline(baseline_counts)
    with reporting_model:
        reporting_prior = pm.sample_prior_predictive(draws=100, var_names=['dnf_intercept', 'dnf_distance', 'kappa', 'report_distance', 'dns_intercept', 'dns_field'], random_seed=SEEDS['prior'])
    with baseline_model:
        baseline_prior = pm.sample_prior_predictive(draws=100, var_names=['count'], random_seed=SEEDS['prior'])
    prior_count_ratios = baseline_prior.prior_predictive['count'].values.reshape(100, -1) / baseline_counts['n_finishers'].to_numpy()
    prior_check_table = pd.DataFrame({'quantile': [0.03, 0.5, 0.97], 'baseline_DNFs_per_finisher': np.quantile(prior_count_ratios, [0.03, 0.5, 0.97])})
    prior_check_table
    return baseline_model, prior_count_ratios, reporting_model


@app.cell(hide_code=True)
def fit_heading(mo):
    mo.md("""
    ## Fit

    Reporting, baseline and joint fits run sequentially. Reporting probabilities are recomputed without the held-out races and plugged in at posterior means; upstream uncertainty is not fully propagated. Smoke and full fits have different cache keys. Full fits use the pinned local snapshot.
    """)
    return


@app.cell(hide_code=True)
def fit_helpers(
    OUT,
    SEEDS,
    SETTINGS,
    az,
    check_memory,
    datetime,
    dm,
    experiment_key,
    mo,
    mv,
    pm,
    sample_lean,
):
    def fit_validation(model, stage, *, large=False, runner_gender=None):
        """Notebook orchestration: exact-config cache, disk streaming, atomic persistence."""
        config = SETTINGS[stage]
        metadata = {'experiment': experiment_key, 'stage': stage, 'chains': SETTINGS['chains'], **config, 'seed': SEEDS[stage]}
        path = OUT / f'{stage}.nc'
        trace = mv.load_cached_trace(path, metadata, chains=SETTINGS['chains'], draws=config['draws'])
        if trace is not None:
            return trace
        mo.output.replace(mo.callout(mo.md(f"{stage.title()} fit started at {datetime.now():%H:%M}; {config['draws']} retained draws per chain. Allow several hours for the full joint fit."), kind='info'))
        if large:
            keep = [v.name for v in model.free_RVs + model.deterministics if v.name not in ('runner_effect', 'runner_dnf_effect')]
            check_memory(model, keep, chains=SETTINGS['chains'], draws=config['draws'], tune=config['tune'], on_disk=True)
            store = OUT / f'{stage}.zarr'
            scalars = [v.name for v in model.free_RVs if v.ndim == 0]
            trace = mv.recover_trace_store(store, metadata=metadata, chains=SETTINGS['chains'], draws=config['draws'], scalar_names=scalars)
            if trace is None:
                mv.prepare_trace_store(store, metadata=metadata)
                trace = sample_lean(model, keep, chains=SETTINGS['chains'], store_dir=store, random_seed=SEEDS[stage], progressbar=True, **config)
            trace = az.InferenceData(posterior=dm.derive_runner_effects(trace.posterior, runner_gender), sample_stats=trace.sample_stats)
        else:
            with model:
                trace = pm.sample(chains=SETTINGS['chains'], nuts_sampler='nutpie', random_seed=SEEDS[stage], progressbar=True, **config)
        mv.save_trace_atomic(trace, path, metadata)
        return mv.load_cached_trace(path, metadata, chains=SETTINGS['chains'], draws=config['draws'])


    return (fit_validation,)


@app.cell(hide_code=True)
def reporting_fit(az, fit_validation, reporting_model):
    reporting_trace = fit_validation(reporting_model, 'reporting')
    reporting_summary = az.summary(reporting_trace, var_names=['dnf_intercept','dnf_distance','kappa','report_distance','dns_intercept','dns_field','series_sd'], hdi_prob=0.94)
    reporting_summary
    return (reporting_trace,)


@app.cell(hide_code=True)
def baseline_fit(az, baseline_model, fit_validation):
    baseline_trace = fit_validation(baseline_model, 'baseline')
    baseline_summary = az.summary(baseline_trace, var_names=['intercept','slope','course_sd','dispersion'], hdi_prob=0.94)
    baseline_summary
    return (baseline_trace,)


@app.cell(hide_code=True)
def joint_design(
    OUT,
    core_names,
    dm,
    heldout_ids,
    mv,
    reporting_trace,
    reporting_training,
    rm,
    runner_order,
    visible_core,
):
    reporting_probabilities = rm.reporting_probabilities(reporting_trace.posterior, reporting_training.loc[reporting_training['did'].isin(visible_core['event_distance_id'].unique())])
    mv.atomic_write_frame(OUT / 'reporting_probabilities.parquet', reporting_probabilities)
    joint_data = mv.prepare_dnf_inputs(visible_core, reporting_probabilities, core_names, known_nonreporting_ids=heldout_ids)
    assert not joint_data.starts['event_distance_id'].isin(heldout_ids).any()
    assert list(joint_data.runners) == list(runner_order)
    joint_model = dm.build_unobserved_model(joint_data)
    print(f"Joint design: {len(joint_data.courses)} courses, {len(joint_data.runners):,} runners, {len(joint_data.starts):,} reporting starts; {len(heldout_ids)} races excluded from all DNF likelihood terms.")
    return joint_data, joint_model


@app.cell(hide_code=True)
def joint_prior_check(
    SEEDS,
    core_names,
    joint_model,
    np,
    plt,
    pm,
    prior_count_ratios,
):
    with joint_model:
        joint_prior = pm.sample_prior_predictive(draws=100, var_names=['course_dnf_level','omega_dnf','rho_runner','dnf_runner_pace_quadratic','tau_edition'], random_seed=SEEDS['prior'])
    prior_rng = np.random.default_rng(SEEDS['prior'])
    prior_post = joint_prior.prior.stack(sample=('chain','draw'))
    prior_z1, prior_z2 = prior_rng.standard_normal((2,100,400))
    prior_rho = prior_post['rho_runner'].values[:,None]
    prior_runner = prior_post['omega_dnf'].values[:,None] * (prior_rho*prior_z1 + np.sqrt(1-prior_rho**2)*prior_z2) + prior_post['dnf_runner_pace_quadratic'].values[:,None]*(prior_z1**2-1)
    prior_edition = prior_post['tau_edition'].values[:,None]*prior_rng.standard_normal((100,len(core_names)))
    prior_logits = prior_post['course_dnf_level'].transpose('sample','course').values + prior_edition
    prior_probabilities = np.array([np.mean(1/(1+np.exp(-(prior_logits[i,:,None]+prior_runner[i,None,:]))),axis=1) for i in range(100)])
    prior_figure, prior_axes = plt.subplots(1,2,figsize=(11,4))
    prior_axes[0].hist(prior_probabilities.ravel(),bins=30,color='#2A9D8F')
    prior_axes[0].set(xlabel='Prior DNF probability at age 40, gender midpoint',ylabel='Draws × courses')
    prior_axes[1].hist(np.log1p(prior_count_ratios.ravel()),bins=30,color='#264653')
    prior_axes[1].set(xlabel='Baseline log(1 + DNFs / finishers)',ylabel='Draws × races')
    prior_figure.tight_layout()
    prior_figure
    return (prior_probabilities,)


@app.cell(hide_code=True)
def joint_fit(
    SETTINGS,
    az,
    baseline_trace,
    dm,
    fit_validation,
    joint_data,
    joint_model,
    np,
    prior_probabilities,
):
    assert baseline_trace.posterior.sizes['draw'] == SETTINGS['baseline']['draws']
    assert np.isfinite(prior_probabilities).all()
    joint_trace = fit_validation(joint_model, 'joint', large=True, runner_gender=joint_data.runner_gender)
    joint_summary = az.summary(joint_trace, var_names=dm.FINISH_SCALARS + dm.DNF_SCALARS, hdi_prob=0.94, round_to=4)
    joint_summary
    return (joint_trace,)


@app.cell(hide_code=True)
def convergence_heading(mo):
    mo.md("""
    ## Convergence

    Check population and latent parameters separately from predictive summaries. Slow course/race-year separation is a limitation even when their sums mix well. A completed sampling job is not a convergence certificate.
    """)
    return


@app.cell(hide_code=True)
def convergence_diagnostics(
    OUT,
    az,
    baseline_trace,
    dm,
    joint_trace,
    mo,
    mv,
    np,
    pd,
    reporting_trace,
):
    convergence_rows = []
    for diagnostic_name, diagnostic_trace, diagnostic_vars in [
        ('reporting', reporting_trace, ['dnf_intercept','dnf_distance','kappa','report_distance','dns_intercept','dns_field','series_sd','year_effect','series_effect']),
        ('baseline', baseline_trace, ['intercept','slope','course_sd','dispersion','course_effect']),
        ('joint', joint_trace, dm.FINISH_SCALARS + dm.DNF_SCALARS + ['age_linear','age_quadratic','course_mean','course_dnf_level','edition_dnf_effect','runner_effect','runner_dnf_effect'])]:
        for diagnostic_var in diagnostic_vars:
            diagnostic_summary = az.summary(diagnostic_trace, var_names=[diagnostic_var], kind='diagnostics', round_to='none')
            convergence_rows.append({'model': diagnostic_name, 'parameter': diagnostic_var, 'max_rhat': diagnostic_summary['r_hat'].max(), 'min_bulk_ess': diagnostic_summary['ess_bulk'].min(), 'min_tail_ess': diagnostic_summary['ess_tail'].min(), 'rhat_over_1_01': int((diagnostic_summary['r_hat'] > 1.01).sum())})
    convergence_table = pd.DataFrame(convergence_rows)
    sampler_checks = pd.DataFrame([{'model': name, 'divergences': int(trace.sample_stats['diverging'].sum()), 'depth_hits': int(trace.sample_stats['maxdepth_reached'].sum()), 'min_bfmi': float(np.min(az.bfmi(trace)))} for name, trace in [('reporting',reporting_trace),('baseline',baseline_trace),('joint',joint_trace)]])
    mv.atomic_write_frame(OUT/'convergence.parquet',convergence_table)
    mv.atomic_write_frame(OUT/'sampler_checks.parquet',sampler_checks)
    mo.vstack([sampler_checks,convergence_table])
    return convergence_table, sampler_checks


@app.cell(hide_code=True)
def joint_rank_plot(az, joint_trace, np):
    rank_axes = az.plot_rank(joint_trace,var_names=['dnf_mean','dnf_distance','tau_dnf','tau_edition','rho_runner'],kind='vlines',figsize=(11,7))
    rank_figure = np.ravel(rank_axes)[0].figure
    rank_figure.tight_layout()
    rank_figure
    return


@app.cell(hide_code=True)
def prediction_heading(mo):
    mo.md("""
    ## Held-out predictions

    For each target, draw a fresh race-year effect and average starter risk over visible finisher composition and the remaining reporting-start reference population. Predict the hidden count conditional on the observed finisher count. These predictions assume known nonreporting, not uncertainty about whether reporting occurred.
    """)
    return


@app.cell(hide_code=True)
def heldout_predictions(
    OUT,
    SEEDS,
    SETTINGS,
    baseline_trace,
    datetime,
    experiment_key,
    heldout_ids,
    joint_data,
    joint_trace,
    mo,
    mv,
    targets,
):
    def cached_prediction(name, producer, metadata):
        path = OUT / f'{name}.nc'
        prediction = mv.load_predictions_cache(path, metadata=metadata)
        if prediction is None:
            mo.output.replace(mo.callout(mo.md(f"Predicting {name} since {datetime.now():%H:%M}; bounded batches, no held-out outcomes are read."),kind='info'))
            prediction = producer()
            mv.atomic_write_predictions(path, prediction, metadata=metadata)
        return prediction

    prediction_metadata = {'experiment': experiment_key, 'version': mv.PREDICTION_VERSION, 'draws_per_chain': SETTINGS['prediction_draws_per_chain'], 'seed': SEEDS['prediction']}
    baseline_predictions = cached_prediction('baseline_predictions', lambda: mv.predict_count_baseline(baseline_trace.posterior, targets, seed=SEEDS['prediction'],posterior_draws_per_chain=SETTINGS['prediction_draws_per_chain']), {**prediction_metadata,'model':'baseline'})
    joint_predictions = cached_prediction('joint_predictions', lambda: mv.predict_hidden_counts(joint_trace.posterior,joint_data,heldout_ids,seed=SEEDS['prediction'],posterior_draws_per_chain=SETTINGS['prediction_draws_per_chain'],nodes=400),{**prediction_metadata,'model':'joint','nodes':400})
    check_predictions = cached_prediction('joint_predictions_check', lambda: mv.predict_hidden_counts(joint_trace.posterior,joint_data,heldout_ids,seed=SEEDS['prediction_check'],posterior_draws_per_chain=SETTINGS['prediction_draws_per_chain'],nodes=800),{**prediction_metadata,'model':'joint','nodes':800,'seed':SEEDS['prediction_check']})
    prediction_artifacts = [OUT/f'{name}.nc' for name in ['baseline_predictions','joint_predictions','joint_predictions_check']]
    assert all(path.exists() for path in prediction_artifacts)
    print(f"Saved {joint_predictions.sizes['chain'] * joint_predictions.sizes['draw']} predictive count vectors for {len(heldout_ids)} held-out races.")
    return (
        baseline_predictions,
        check_predictions,
        joint_predictions,
        prediction_artifacts,
    )


@app.cell(hide_code=True)
def comparison_heading(mo):
    mo.md("""
    ## Predictive comparison

    The primary comparison is the mean paired weighted interval score (joint minus baseline; negative favors the joint model). Coverage is assessed together with interval width, bias and proper scores. Course-clustered bootstrap intervals describe evaluation uncertainty, not posterior credible intervals. Scores are based on actual held-out predictions, not PSIS-LOO.
    """)
    return


@app.cell(hide_code=True)
def heldout_scores(
    OUT,
    SEEDS,
    SMOKE,
    baseline_predictions,
    check_predictions,
    heldout_ids,
    joint_predictions,
    mo,
    mv,
    np,
    pd,
    prediction_artifacts,
    processed_results,
    targets,
):
    # Truth is first read by the evaluator, after all prediction artifacts exist.
    assert all(path.exists() for path in prediction_artifacts)
    evaluation_truth = mv.heldout_truth(processed_results, heldout_ids)
    mv.atomic_write_frame(OUT/'evaluation_truth.parquet',evaluation_truth)
    model_scores = pd.concat([mv.score_predictions(prediction,evaluation_truth,targets) for prediction in [baseline_predictions,joint_predictions]],ignore_index=True)
    score_tables = mv.summarize_scores(model_scores,seed=SEEDS['bootstrap'],bootstraps=200 if SMOKE else 2000)
    check_scores = mv.score_predictions(check_predictions,evaluation_truth,targets)
    prediction_stability = pd.DataFrame({'calculation':['400 donors, primary seed','800 donors, check seed'], 'mean_WIS':[model_scores.loc[model_scores['model']=='joint','wis'].mean(),check_scores['wis'].mean()], 'coverage_94':[model_scores.loc[model_scores['model']=='joint','coverage_94'].mean(),check_scores['coverage_94'].mean()]})
    total_count_rows=[]
    for total_name,total_prediction in [('baseline',baseline_predictions),('joint',joint_predictions)]:
        total_draws=mv.sum_predictions(total_prediction).values.ravel()
        total_count_rows.append({'model':total_name,'actual':int(evaluation_truth['n_dnf'].sum()),'predicted_mean':float(total_draws.mean()),'lower_94':float(np.quantile(total_draws,.03,method='inverted_cdf')),'upper_94':float(np.quantile(total_draws,.97,method='inverted_cdf'))})
    total_count_check=pd.DataFrame(total_count_rows)
    mv.atomic_write_frame(OUT/'scores.parquet',model_scores)
    mv.atomic_write_frame(OUT/'scores_prediction_check.parquet',check_scores)
    mv.atomic_write_frame(OUT/'total_count_check.parquet',total_count_check)
    mo.vstack([mo.callout('SMOKE ONLY: short fits and few predictive draws; do not interpret scores.',kind='warn') if SMOKE else mo.md('Actual held-out evaluation of the frozen specification.'),score_tables['comparison'],score_tables['overall'],prediction_stability,total_count_check])
    return evaluation_truth, model_scores, score_tables


@app.cell(hide_code=True)
def evaluation_status(
    SMOKE,
    az,
    convergence_table,
    integrated_diagnostics,
    joint_predictions,
    mo,
    mv,
    sampler_checks,
):
    predictive_diagnostics = az.summary(mv.target_probability_summary(joint_predictions).to_dataset(), kind='diagnostics', round_to='none')
    provisional = SMOKE or bool((sampler_checks['divergences'] > 0).any() or (sampler_checks['depth_hits'] > 0).any() or (convergence_table['max_rhat'] > 1.01).any() or (convergence_table['min_bulk_ess'] < 400).any() or (convergence_table['min_tail_ess'] < 400).any() or (integrated_diagnostics['r_hat'] > 1.01).any() or (integrated_diagnostics['ess_bulk'] < 400).any())
    mo.callout('Provisional: at least one sampling diagnostic fails the screening thresholds. Inspect the parameter-level table before using scores.' if provisional else 'Sampling screens passed. This does not establish predictive validity or transportability.',kind='warn' if provisional else 'info')
    return


@app.cell(hide_code=True)
def integrated_mean_diagnostics(
    OUT,
    SEEDS,
    SETTINGS,
    az,
    heldout_ids,
    joint_data,
    joint_trace,
    mv,
):
    integrated_moments = mv.integrated_target_moments(joint_trace.posterior,joint_data,heldout_ids,seed=SEEDS['prediction'],posterior_draws_per_chain=SETTINGS['prediction_draws_per_chain'],nodes=64,edition_nodes=8)
    integrated_diagnostics = az.summary(integrated_moments,kind='diagnostics',round_to='none')
    mv.atomic_write_frame(OUT/'integrated_mean_diagnostics.parquet',integrated_diagnostics.reset_index(names='parameter'))
    integrated_diagnostics
    return integrated_diagnostics, integrated_moments


@app.cell(hide_code=True)
def predictive_mean_traces(az, integrated_moments, np):
    worst_target_ids = integrated_moments.did.values[np.argsort(az.rhat(integrated_moments)['mean_count'].values)[-3:]]
    mean_trace_axes = az.plot_trace(integrated_moments[['mean_count']].sel(did=worst_target_ids),compact=False,figsize=(11,7))
    mean_trace_figure = np.ravel(mean_trace_axes)[0].figure
    mean_trace_figure.tight_layout()
    mean_trace_figure
    return


@app.cell(hide_code=True)
def coverage_plot(SMOKE, heldout_ids, plt, score_tables):
    coverage_figure, coverage_axes = plt.subplots(1,2,figsize=(12,4.5))
    for plot_model, plot_color in [('baseline','#264653'),('joint','#2A9D8F')]:
        coverage_rows = score_tables['coverage'].query('model == @plot_model').sort_values('level')
        coverage_axes[0].errorbar(coverage_rows['level']/100, coverage_rows['coverage'], yerr=[coverage_rows['coverage']-coverage_rows['lower_94'],coverage_rows['upper_94']-coverage_rows['coverage']], marker='o',label=plot_model,color=plot_color)
        width_rows = score_tables['overall'].set_index('model').loc[plot_model]
        coverage_axes[1].plot([.5,.8,.94],[width_rows[f'width_{level}'] for level in [50,80,94]],marker='o',label=plot_model,color=plot_color)
    coverage_axes[0].plot([.4,1],[.4,1],color='gray',linestyle='--')
    coverage_axes[0].set(xlabel='Nominal coverage',ylabel='Held-out coverage',ylim=(0,1),title='Course-clustered 94% evaluation intervals')
    coverage_axes[1].set(xlabel='Nominal coverage',ylabel='Mean interval width (DNFs)',title='Coverage must be read with width')
    coverage_axes[0].legend()
    coverage_axes[1].legend()
    coverage_figure.suptitle('SMOKE ONLY — not interpretable' if SMOKE else f'{len(heldout_ids)} individually masked races')
    coverage_figure.tight_layout()
    coverage_figure
    return


@app.cell(hide_code=True)
def conditional_scores(mo, score_tables):
    mo.vstack([score_tables['by_distance_bin'],score_tables['by_finisher_bin'],score_tables['equal_course']])
    return


@app.cell(hide_code=True)
def validation_findings(mo):
    mo.md("""
    ### Findings

    The simple count baseline performs better on this frozen holdout: mean WIS **9.03 versus 9.82** (lower is better). Paired joint-minus-baseline WIS is **+0.79**, with a **94% course-clustered bootstrap evaluation interval [0.30, 1.37]**. Equal-course weighting gives +0.72 [0.23, 1.24]. The second seed and 800-donor numerical check gives joint WIS 9.86; it does not reverse the result.

    Nominal 94% intervals cover **94.7% versus 95.8%** of races, but mean widths are **101.4 versus 127.5 DNFs**. The joint model's extra coverage comes with wider intervals and more upward bias: **11,375** predicted total DNFs versus **9,893** for the baseline and **9,680** observed. Both models overcover at 50%, at roughly 60% coverage.

    There were no retained-sample divergences or tree-depth hits. Nevertheless, five course and 30 race-effect coordinates exceed R-hat 1.01. Fresh-effect-integrated predictive means also show imperfect mixing (maximum R-hat 1.026, minimum bulk ESS 329). Treat the apparent baseline advantage as **provisional**, not as a conclusive ranking. No retuning or additional fit was selected after inspecting these outcomes.

    The 4h14m full pipeline evaluates 263 races across 114 courses within the fixed 172-course convenience sample. Counts are filtered by the existing cleaning rules. This is historical reconstruction of known nonreporters, not reporting classification, forecasting, validation of core selection, or proof of transportability to naturally silent races. The dataset already informed model development; it is not a pristine external test.
    """)
    return


@app.cell(hide_code=True)
def stability_heading(mo):
    mo.md("""
    ### Numerical stability

    These post-fit diagnostics keep the frozen score and holdout unchanged. Chain and early/late comparisons use fewer predictive draws, so their WIS values also reflect finite-quantile noise. Chain numbers across models are arbitrary, not paired MCMC trajectories. The block bootstrap resamples contiguous saved-prediction positions independently within each model and chain, using the same draw indices across all races to retain their dependence. Its variation is conditional on the sampled chains; it cannot detect an unvisited mode or replace convergence diagnostics. It does not resample races and is distinct from the course-clustered evaluation interval.

    Excluding poorly mixing targets below is an influence diagnostic only, not a replacement primary score.
    """)
    return


@app.cell(hide_code=True)
def stability_subsets(
    SETTINGS,
    baseline_predictions,
    evaluation_truth,
    integrated_diagnostics,
    joint_predictions,
    mo,
    model_scores,
    mv,
    np,
    pd,
    targets,
):
    def prediction_wis(prediction):
        return float(mv.score_predictions(prediction,evaluation_truth,targets)['wis'].mean())

    chain_score_rows=[]
    for stability_model, stability_predictions in [('baseline',baseline_predictions),('joint',joint_predictions)]:
        for stability_chain in range(stability_predictions.sizes['chain']):
            chain_score_rows.append({'model':stability_model,'chain':stability_chain,'wis':prediction_wis(stability_predictions.isel(chain=[stability_chain]))})
    chain_scores=pd.DataFrame(chain_score_rows)
    chain_cross_differences = (chain_scores.query("model == 'joint'")['wis'].to_numpy()[:,None] - chain_scores.query("model == 'baseline'")['wis'].to_numpy()[None,:])
    segment_rows=[]
    for segment_name, segment_slice in [('early',slice(0,SETTINGS['prediction_draws_per_chain']//2)),('late',slice(SETTINGS['prediction_draws_per_chain']//2,None))]:
        segment_base=prediction_wis(baseline_predictions.isel(draw=segment_slice))
        segment_joint=prediction_wis(joint_predictions.isel(draw=segment_slice))
        segment_rows.append({'segment':segment_name,'baseline_wis':segment_base,'joint_wis':segment_joint,'difference':segment_joint-segment_base})
    segment_scores=pd.DataFrame(segment_rows)
    flagged_rows=integrated_diagnostics.loc[(integrated_diagnostics['r_hat']>1.01)|(integrated_diagnostics['ess_bulk']<400)|(integrated_diagnostics['ess_tail']<400)]
    flagged_ids=set(flagged_rows.index.str.extract(r'\[(\d+)\]',expand=False).astype(int))
    race_wis_difference=model_scores.pivot(index='did',columns='model',values='wis').eval('joint - baseline')
    influence_rows=[]
    for influence_name,influence_mask in [('all',np.ones(len(race_wis_difference),dtype=bool)),('flagged',race_wis_difference.index.isin(flagged_ids)),('unflagged',~race_wis_difference.index.isin(flagged_ids))]:
        influence_values=race_wis_difference[influence_mask]
        influence_rows.append({'targets':influence_name,'n_races':len(influence_values),'mean_difference':influence_values.mean(),'sum_difference':influence_values.sum()})
    influence_table=pd.DataFrame(influence_rows)
    mo.vstack([chain_scores,segment_scores,influence_table])
    return (prediction_wis,)


@app.cell(hide_code=True)
def stability_monte_carlo(
    baseline_predictions,
    joint_predictions,
    np,
    pd,
    prediction_wis,
    xr,
):
    def block_resampled_prediction(prediction,rng,block_length):
        values=prediction.transpose('chain','draw','did').values
        n_chains,n_draws,n_targets=values.shape
        resampled=np.empty_like(values)
        for chain in range(n_chains):
            starts=rng.integers(0,n_draws,size=int(np.ceil(n_draws/block_length)))
            indices=((starts[:,None]+np.arange(block_length))%n_draws).ravel()[:n_draws]
            resampled[chain]=values[chain,indices,:]
        return xr.DataArray(resampled,dims=('chain','draw','did'),coords={'chain':np.arange(n_chains),'draw':np.arange(n_draws),'did':prediction.did.values},attrs={'model':prediction.attrs['model']})

    mc_rng=np.random.default_rng(sum(map(ord,'2.4_dnf_validation:mc_stability:v1')))
    mc_rows=[]
    for mc_block in [25,50]:
        for mc_replicate in range(200):
            mc_baseline=prediction_wis(block_resampled_prediction(baseline_predictions,mc_rng,mc_block))
            mc_joint=prediction_wis(block_resampled_prediction(joint_predictions,mc_rng,mc_block))
            mc_rows.append({'block_length':mc_block,'replicate':mc_replicate,'difference':mc_joint-mc_baseline})
    mc_replicates=pd.DataFrame(mc_rows)
    mc_summary=mc_replicates.groupby('block_length')['difference'].agg(mean='mean',bootstrap_sd='std',lower_94=lambda x:x.quantile(.03),upper_94=lambda x:x.quantile(.97),minimum='min')
    mc_summary
    return


@app.cell(hide_code=True)
def stability_findings(mo):
    mo.md("""
    ### Stability findings

    The baseline advantage is stable within the saved chains. Every one of the 16 cross-model chain comparisons favors the baseline, with joint-minus-baseline WIS from **+0.69 to +0.95**. Early and late halves give **+0.80 and +0.79**.

    Thirteen targets fail the integrated-mean R-hat/ESS screens. They contribute about **15%** of the summed WIS difference. Excluding those same races from both models leaves **+0.70** across the other 250 races, so they do not explain the overall baseline advantage. This exclusion is diagnostic only; the primary score remains +0.79 on all 263 races.

    Within-chain circular block bootstraps with block lengths 25 and 50 saved draws give a Monte Carlo standard deviation of approximately **0.038 WIS**, far smaller than the observed +0.79 gap. Their descriptive central 94% ranges are approximately **[+0.73, +0.86]** and **[+0.72, +0.86]**. These are conditional resampling ranges, not evaluation intervals, posterior credible intervals, or a bound on bias from poor mixing. The bootstrap cannot reveal an unvisited mode; no chains or observations are removed from the primary estimate.

    There is no numerical indication here that a longer fit is needed merely to decide which method to prefer on this holdout. Retain the convergence warning for individual predictions and course effects. No new posterior or predictive draws were generated, and neither model nor the holdout was changed.
    """)
    return


@app.cell(hide_code=True)
def bias_heading(mo):
    mo.md("""
    ### Count bias and spread

    This is a post-validation diagnostic of the fixed fitted model, not a scored alternative. One-node Gauss–Hermite quadrature evaluates the fresh race effect at zero; eight nodes integrate over its fitted distribution. Both calculations use the same 64 common training-start donors per race and the same posterior draws. Their difference isolates the prediction-stage effect of averaging over fresh race effects conditional on this posterior and donor approximation; it is not the result of refitting without race effects.

    The count transform is nonlinear: averaging `F × p / (1 − p)` is not the same as transforming an averaged probability. We separate that convexity contribution from the probability shift itself. Donor composition and the fitted course/runner levels remain unchanged here.
    """)
    return


@app.cell(hide_code=True)
def bias_decomposition(
    SEEDS,
    SETTINGS,
    evaluation_truth,
    heldout_ids,
    integrated_moments,
    joint_data,
    joint_trace,
    mo,
    mv,
    targets,
    xr,
):
    zero_race_moments=mv.integrated_target_moments(joint_trace.posterior,joint_data,heldout_ids,seed=SEEDS['prediction'],posterior_draws_per_chain=SETTINGS['prediction_draws_per_chain'],nodes=64,edition_nodes=1)
    finishers_by_target=xr.DataArray(targets.set_index('did').loc[integrated_moments.did.values,'n_finishers'].to_numpy(),dims='did',coords={'did':integrated_moments.did.values})
    count_at_averaged_probability=finishers_by_target*integrated_moments['mean_probability']/(1-integrated_moments['mean_probability'])
    bias_decomposition=targets.set_index('did').loc[integrated_moments.did.values,['distance_bin','finisher_bin','n_finishers']].copy()
    bias_decomposition['actual']=evaluation_truth.set_index('did').loc[bias_decomposition.index,'n_dnf']
    bias_decomposition['zero_race_effect']=zero_race_moments['mean_count'].mean(('chain','draw')).values
    bias_decomposition['average_probability_transform']=count_at_averaged_probability.mean(('chain','draw')).values
    bias_decomposition['integrated_race_effect']=integrated_moments['mean_count'].mean(('chain','draw')).values
    bias_decomposition['probability_shift']=bias_decomposition['average_probability_transform']-bias_decomposition['zero_race_effect']
    bias_decomposition['convexity_uplift']=bias_decomposition['integrated_race_effect']-bias_decomposition['average_probability_transform']
    bias_decomposition['race_effect_uplift']=bias_decomposition['integrated_race_effect']-bias_decomposition['zero_race_effect']
    bias_totals=bias_decomposition.select_dtypes('number').drop(columns=['finisher_bin'],errors='ignore').sum().to_frame('total')
    bias_by_distance=bias_decomposition.groupby('distance_bin',observed=True)[['actual','zero_race_effect','integrated_race_effect','probability_shift','convexity_uplift']].sum()
    mo.vstack([bias_totals,bias_by_distance])
    return bias_totals, finishers_by_target


@app.cell(hide_code=True)
def bias_spread(
    finishers_by_target,
    joint_predictions,
    mo,
    model_scores,
    mv,
    pd,
):
    predictive_p=mv.target_probability_summary(joint_predictions)
    conditional_count_mean=finishers_by_target*predictive_p/(1-predictive_p)
    conditional_count_variance=finishers_by_target*predictive_p/(1-predictive_p)**2
    spread_components=pd.DataFrame({'did':joint_predictions.did.values,'mean_conditional_count_variance':conditional_count_variance.mean(('chain','draw')).values,'variance_conditional_mean':conditional_count_mean.var(('chain','draw')).values,'empirical_predictive_variance':joint_predictions.var(('chain','draw')).values})
    spread_totals=spread_components.drop(columns='did').sum().to_frame('sum_of_marginal_variances')
    spread_totals['fraction_of_total_variance']=spread_totals['sum_of_marginal_variances']/(spread_totals.loc['mean_conditional_count_variance','sum_of_marginal_variances']+spread_totals.loc['variance_conditional_mean','sum_of_marginal_variances'])
    bias_score_bands=model_scores.groupby(['model','distance_bin'],observed=True).agg(actual=('n_dnf','sum'),predicted=('mean','sum'),mean_bias=('signed_mean_error','mean'),mean_width_94=('width_94','mean'),wis=('wis','mean'))
    bias_score_sizes=model_scores.groupby(['model','finisher_bin'],observed=True).agg(actual=('n_dnf','sum'),predicted=('mean','sum'),mean_bias=('signed_mean_error','mean'),mean_width_94=('width_94','mean'),wis=('wis','mean'))
    mo.vstack([spread_totals,bias_score_bands,bias_score_sizes])
    return bias_score_bands, conditional_count_mean, conditional_count_variance


@app.cell(hide_code=True)
def bias_variance_comparison(
    baseline_predictions,
    baseline_trace,
    conditional_count_mean,
    conditional_count_variance,
    finishers_by_target,
    joint_predictions,
    np,
    pd,
    targets,
    xr,
):
    baseline_diagnostic_post=baseline_trace.posterior.sel(draw=baseline_predictions.draw.values)
    baseline_target_rows=targets.set_index('did').loc[baseline_predictions.did.values]
    baseline_course_selector=xr.DataArray(baseline_target_rows['course_id'].to_numpy(),dims='did',coords={'did':baseline_predictions.did.values})
    baseline_distance=xr.DataArray(np.log(baseline_target_rows['distance_km'].to_numpy()/50),dims='did',coords={'did':baseline_predictions.did.values})
    baseline_conditional_mu=finishers_by_target*np.exp(baseline_diagnostic_post['intercept']+baseline_diagnostic_post['slope']*baseline_distance+baseline_diagnostic_post['course_effect'].sel(course=baseline_course_selector))
    baseline_conditional_variance=baseline_conditional_mu+baseline_conditional_mu**2/baseline_diagnostic_post['dispersion']
    variance_comparison=pd.DataFrame([
        {'model':'baseline','conditional_count_variance':float(baseline_conditional_variance.mean(('chain','draw')).sum()),'mixing_mean_variance':float(baseline_conditional_mu.var(('chain','draw')).sum()),'empirical_variance':float(baseline_predictions.var(('chain','draw')).sum())},
        {'model':'joint','conditional_count_variance':float(conditional_count_variance.mean(('chain','draw')).sum()),'mixing_mean_variance':float(conditional_count_mean.var(('chain','draw')).sum()),'empirical_variance':float(joint_predictions.var(('chain','draw')).sum())}])
    variance_comparison['total_moment_variance']=variance_comparison['conditional_count_variance']+variance_comparison['mixing_mean_variance']
    variance_comparison
    return


@app.cell(hide_code=True)
def bias_figure(bias_score_bands, bias_totals, evaluation_truth, plt):
    bias_figure,bias_axes=plt.subplots(1,2,figsize=(12,4.5))
    for bias_plot_model,bias_plot_color in [('baseline','#264653'),('joint','#2A9D8F')]:
        bias_plot_rows=bias_score_bands.loc[bias_plot_model].reindex(['42-60','60-90','90-120','120-165'])
        bias_axes[0].plot(bias_plot_rows.index,bias_plot_rows['predicted']/bias_plot_rows['actual'],marker='o',label=bias_plot_model,color=bias_plot_color)
    bias_axes[0].axhline(1,color='gray',linestyle='--')
    bias_axes[0].set(xlabel='Distance band (km)',ylabel='Predicted total / actual total',title='Overprediction concentrates at long distances')
    bias_axes[0].legend()
    bias_axes[1].bar(['Race effect\nset to zero','Transform of\naveraged probability','Average of\ncount transform'],[bias_totals.loc[key,'total'] for key in ['zero_race_effect','average_probability_transform','integrated_race_effect']],color=['#457B9D','#E9C46A','#2A9D8F'])
    bias_axes[1].axhline(float(evaluation_truth['n_dnf'].sum()),color='black',linestyle='--',label='Actual total')
    bias_axes[1].set(ylabel='Expected total DNFs',title='Same posterior; diagnostic integration only')
    bias_axes[1].legend()
    bias_figure.tight_layout()
    bias_figure
    return


@app.cell(hide_code=True)
def bias_findings(mo):
    mo.md("""
    ### Bias findings

    Fresh-race integration materially raises the joint model's expected counts. With the same posterior and common donor approximation, setting the new race effect to zero gives **8,823** expected DNFs; integrating it gives **11,291**, an increase of **2,468 (28%)**. About **871** of that increase comes from the shift in averaged DNF probability and **1,597** from the convexity of `p/(1−p)`. Actual held-out DNFs total **9,680**. The integrated diagnostic total differs slightly from the primary predictive total (11,375) because it uses 64 common donors and quadrature rather than the primary 400-donor simulation.

    **The convexity is not an implementation bug.** Averaging conditional count means is the appropriate expectation under this predictive construction. Transforming an averaged probability happens to land near the observed total, but replacing the expectation with that number would be an unjustified change selected against this holdout. Likewise, setting the race effect to zero is not a refit of a no-race-effect model.

    Overprediction is concentrated at longer distances: the joint model predicts **5,788 versus 4,622 actual DNFs at 120–165 km**, and **1,930 versus 1,579 at 90–120 km**. Its aggregate 42–60 km prediction is much closer (2,196 versus 2,118). The joint model has slightly better WIS in the smallest within-distance finisher-count tercile (6.22 versus 6.43), but worse WIS in the middle and largest terciles. These are exploratory subgroup descriptions, not a new selection rule.

    Across races, the sum of marginal predictive variances is approximately **968,000** for the joint model versus **506,000** for the baseline. About **98%** of the joint moment variance comes from variation in conditional means, not the final negative-binomial count draw. This variation includes fresh race effects, posterior uncertainty and donor Monte Carlo variation; the calculation does not separately identify those contributions. These are sums of race-specific variances, **not** variances of the aggregate total, and variance is not identical to central interval width.

    This identifies an important prediction-stage mechanism, not a unique root cause or a validated repair. The next modeling question is whether race-effect variability and the donor-based probabilities jointly calibrate for this reconstruction target, particularly at longer distances. Donor representativeness has not been isolated here. Any revision informed by these errors needs a new predeclared evaluation; the original scores and convergence caveats remain unchanged.
    """)
    return


@app.cell(hide_code=True)
def limitations_heading(mo):
    mo.md("""
    ## Limitations

    This tests historical reconstruction on selected reporting races after artificial suppression. It is not forecasting, external validation, reporting classification, or evidence that natural nonreporters have the same entrant population. Counts cover only runners retained by the existing cleaning rules. The fixed core and v2 model were already developed using this dataset; this is out-of-fit evaluation of a frozen specification, not a pristine external test.
    """)
    return


@app.cell(hide_code=True)
def handoff_heading(mo):
    mo.md("""
    ## Handoff

    Artifacts are isolated under `data/cache/2.4_dnf_validation/`. The production reporting export, 2.3 traces and course estimates are not overwritten by validation. No additional split or model tuning is selected from these held-out results.
    """)
    return


@app.cell(hide_code=True)
def artifact_handoff(
    OUT,
    ROOT,
    SMOKE,
    experiment_key,
    integrated_diagnostics,
    mo,
    snapshot_sha256,
):
    mo.md(f"""
    **Experiment:** `{experiment_key}`  
    **Snapshot:** `{snapshot_sha256}`  
    **Mode:** {'smoke; not for inference' if SMOKE else 'full large core'}  
    **Artifacts:** `{OUT.relative_to(ROOT)}`

    The primary score uses the predeclared 400-donor calculation; the second seed with 800 donors is a numerical stability check, not an alternative selected for a better score. The separate integrated-mean diagnostic uses common training-start donors and Gauss–Hermite integration over fresh race effects, removing predictive simulation noise. It is a numerical approximation for mixing diagnostics only, never a scored prediction. Its maximum R-hat is {integrated_diagnostics.r_hat.max():.3f} and minimum bulk ESS is {integrated_diagnostics.ess_bulk.min():.0f}, so the comparison remains provisional. No model was retuned against this holdout.
    """)
    return


if __name__ == "__main__":
    app.run()
