import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def configuration():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import xarray as xr
    import arviz as az
    import matplotlib.pyplot as plt
    import pymc as pm
    from pathlib import Path
    from dataclasses import asdict, replace
    from datetime import datetime
    from time import perf_counter
    from utils import masked_validation as cache
    from utils.sampling import sample_lean, check_memory

    ROOT=Path(__file__).resolve().parent.parent
    CACHE_ROOT=ROOT/'data'/'cache'/'2.5_dnf_field'
    MODEL_VERSION='joint-field-v1'
    SCENARIOS=['benign','race_variation','participation_selection','population_misspecification']
    SAMPLER={'chains':4,'tune':1000,'draws':500,'target_accept':.95}
    SMOKE_SAMPLER={'chains':2,'tune':50,'draws':50,'target_accept':.95}
    def stage_seed(scenario,stage):
        return sum(map(ord,f'2.5_dnf_field:{scenario}:{stage}:v1'))

    return (
        CACHE_ROOT,
        MODEL_VERSION,
        ROOT,
        SCENARIOS,
        SMOKE_SAMPLER,
        asdict,
        cache,
        check_memory,
        datetime,
        mo,
        np,
        pd,
        perf_counter,
        plt,
        sample_lean,
        stage_seed,
    )


@app.cell(hide_code=True)
def field_helpers(np, stage_seed):
    from utils import joint_field_model as jf
    PRIORS=jf.JointFieldPrior()
    PRIOR_SEEDS=np.random.SeedSequence(stage_seed('shared','prior')).spawn(200)
    SIM_DESIGN_SEED=stage_seed('shared','design')
    SIM_NUISANCE_SEED=stage_seed('shared','nuisance')
    return PRIORS, PRIOR_SEEDS, SIM_DESIGN_SEED, SIM_NUISANCE_SEED, jf


@app.cell(hide_code=True)
def scope(mo):
    mo.md(r"""
    # Joint entrant and race model

    ## Scope

    A **synthetic prototype**, not a new analysis of UltraSignup results. Known runners have an externally supplied registry and candidate races. Anonymous entrants are one-off identities. These assumptions make the observation process explicit, but they are not yet justified for the real database.

    The completed 2.4 validation remains unchanged. This notebook does not load its outcomes, traces or donor pools. Passing the numerical checks below is not evidence of transportability or identification on real data, and these fixed-truth recovery cases are not simulation-based calibration.
    """)
    return


@app.cell(hide_code=True)
def generative_process(mo):
    mo.md(r"""
    ## Generative process

    Named candidates participate with probability $\pi$, then DNF with probability $p$. Reporting races distinguish finishing, DNF and not starting. At suppressed races, a missing named record has likelihood $(1-\pi)+\pi p$: **absence is not a known nonstart**. Its conditional hidden-DNF probability is $\pi p/[(1-\pi)+\pi p]$.

    Anonymous entrants follow $\lambda\sim\mathrm{Gamma}(a,b)$ and $N\sim\mathrm{Poisson}(\lambda)$, with a jointly learned course-specific demographic distribution and integrated latent runner traits. If $Q$ is their population finish probability, anonymous hidden DNFs follow $\mathrm{NB}(a+F,(b+Q)/(b+1))$ conditional on the remaining parameters and visible finishers. This is proper at $F=0$.

    Correlated pace/DNF edition effects let observed finishing pace inform the same race's hidden-count posterior. Predictions use that conditional edition posterior, not an unrelated fresh-effect draw. Pace is conditional on finishing; this is not a counterfactual completion time for a DNF and does not identify a causal weather effect.

    The synthetic log-pace distribution is unfiltered Normal, in minutes/km. Real-data plausibility filtering and registry ascertainment require a later model-design step.
    """)
    return


@app.cell(hide_code=True)
def synthetic_heading(mo):
    mo.md("""
    ## Synthetic data

    Six courses, four editions each, and one masked occurrence per course. Candidate membership is generated independently of latent runner traits and before any outcomes.
    """)
    return


@app.cell(hide_code=True)
def synthetic_design(
    PRIORS,
    SCENARIOS,
    SIM_DESIGN_SEED,
    SIM_NUISANCE_SEED,
    jf,
    pd,
    stage_seed,
):
    simulations={scenario:jf.simulate_joint_field(scenario=scenario,seed=stage_seed(scenario,'outcomes'),design_seed=SIM_DESIGN_SEED,nuisance_seed=SIM_NUISANCE_SEED,priors=PRIORS) for scenario in SCENARIOS}
    smoke_simulation=jf.simulate_joint_field(scenario='smoke',seed=stage_seed('smoke','outcomes'),design_seed=SIM_DESIGN_SEED,nuisance_seed=SIM_NUISANCE_SEED,priors=PRIORS)
    synthetic_rows=[]
    for synthetic_name,synthetic_sim in simulations.items():
        synthetic_truth=synthetic_sim.truth['editions']
        synthetic_rows.append({'scenario':synthetic_name,'editions':len(synthetic_sim.data.editions),'suppressed':int((~synthetic_sim.data.editions['reported']).sum()),'registry':len(synthetic_sim.data.registry),'candidate_edges':len(synthetic_sim.data.candidates),'starts':int((synthetic_truth['named_starters']+synthetic_truth['anonymous_starters']).sum()),'dnfs':int(synthetic_truth['total_dnf'].sum())})
    synthetic_design_summary=pd.DataFrame(synthetic_rows)
    synthetic_design_summary
    return simulations, smoke_simulation


@app.cell(hide_code=True)
def prior_heading(mo):
    mo.md("""
    ## Prior checks

    The added priors are explicit illustrative prototype assumptions, not expert elicitation or estimates from the previous holdout. Inspect prior implications before fitting.
    """)
    return


@app.cell(hide_code=True)
def prior_simulation(PRIORS, PRIOR_SEEDS, jf, np, pd):
    prior_count_frames=[]
    prior_pace_values=[]
    prior_group_counts=np.zeros(6,dtype=np.int64)
    for prior_rep,prior_seed_sequence in enumerate(PRIOR_SEEDS):
        prior_outcome_seed,prior_design_seed,prior_nuisance_seed=map(int,prior_seed_sequence.generate_state(3))
        prior_sim=jf.simulate_joint_field(scenario='prior',seed=prior_outcome_seed,design_seed=prior_design_seed,nuisance_seed=prior_nuisance_seed,priors=PRIORS)
        prior_counts=prior_sim.truth['editions'].copy()
        prior_counts['replicate']=prior_rep
        prior_counts['starts']=prior_counts['named_starters']+prior_counts['anonymous_starters']
        prior_counts['finishers']=prior_counts['starts']-prior_counts['total_dnf']
        prior_count_frames.append(prior_counts)
        prior_complete=prior_sim.truth['complete_data']
        prior_pace_values.append(np.concatenate([prior_complete.candidates.loc[prior_complete.candidates['state']=='finish','log_pace'].to_numpy(),prior_complete.anonymous.loc[prior_complete.anonymous['state']=='finish','log_pace'].to_numpy()]))
        prior_group_counts+=np.bincount(prior_complete.anonymous['group'].to_numpy(dtype=int),minlength=6)
    prior_counts_all=pd.concat(prior_count_frames,ignore_index=True)
    prior_log_paces=np.concatenate(prior_pace_values)
    assert np.isfinite(prior_log_paces).all()
    prior_quantiles=prior_counts_all[['starts','finishers','total_dnf']].quantile([0,.03,.5,.97,1])
    prior_quantiles
    return prior_counts_all, prior_log_paces


@app.cell(hide_code=True)
def prior_plot(np, plt, prior_counts_all, prior_log_paces):
    prior_field_figure,prior_field_axes=plt.subplots(1,3,figsize=(12,3.8))
    prior_field_axes[0].hist(np.log1p(prior_counts_all['starts']),bins=35,color='#264653')
    prior_field_axes[0].set(xlabel='log(1 + starts)',ylabel='Simulated editions',title='Prior field sizes')
    prior_field_axes[1].hist(prior_counts_all.loc[prior_counts_all['starts']>0,'total_dnf']/prior_counts_all.loc[prior_counts_all['starts']>0,'starts'],bins=30,color='#2A9D8F')
    prior_field_axes[1].set(xlabel='DNF fraction',title='Prior DNF rates')
    prior_field_axes[2].hist(prior_log_paces,bins=50,color='#E9C46A')
    prior_field_axes[2].set(xlabel='Log pace (minutes/km)',title='Unfiltered synthetic finishers')
    prior_field_figure.tight_layout()
    prior_field_figure
    return


@app.cell(hide_code=True)
def prior_review(mo, np, prior_counts_all, prior_log_paces):
    mo.md(f"""
    The prior simulation puts its 3rd-percentile finishing pace at **{np.exp(np.quantile(prior_log_paces,.03)):.2f} min/km**, which is implausibly fast for ultras. The user explicitly authorized retaining these illustrative priors **for synthetic computation/recovery checks only**, with physical calibration required before real-data use. No prior was changed after inspecting this output.

    Median prior field size is {prior_counts_all['starts'].median():.0f}; its 97th percentile is {prior_counts_all['starts'].quantile(.97):.0f}. Zero finishers occur in {100*(prior_counts_all['finishers']==0).mean():.2f}% of simulated editions. The unfiltered Normal-pace prototype is not deployment-ready.
    """)
    return


@app.cell(hide_code=True)
def likelihood_heading(mo):
    mo.md("""
    ## Likelihood verification

    Independent enumeration and integration tests precede inference. Compare quadrature orders at initial and perturbed points, including scaled gradients; do not start a fit when the numerical gate fails.
    """)
    return


@app.cell(hide_code=True)
def initial_quadrature_gates(
    PRIORS,
    datetime,
    jf,
    mo,
    pd,
    simulations,
    smoke_simulation,
):
    def run_quadrature_gate(data,priors,label):
        mo.output.replace(mo.callout(f'Compiling likelihood/gradient checks for {label}, started {datetime.now():%H:%M}; this may take several minutes.',kind='info'))
        try:
            order,audits=jf.select_field_quadrature(data,priors=priors)
        except ValueError as error:
            if not hasattr(error,'audits'):
                raise
            order,audits=None,error.audits
        rows=[]
        for audit in audits:
            for point in audit.point.values:
                rows.append({'case':label,'point':str(point),'lower_order':audit.attrs['lower_order'],'upper_order':audit.attrs['upper_order'],'max_absolute_loglik_difference':float(audit['absolute_loglik_difference'].sel(point=point).max()),'max_scaled_gradient_difference':float(audit['max_scaled_gradient_difference'].sel(point=point)),'passed':audit.attrs['passed']})
        return order,pd.DataFrame(rows)

    smoke_order,smoke_gate=run_quadrature_gate(smoke_simulation.data,PRIORS,'smoke')
    benign_order,benign_gate=run_quadrature_gate(simulations['benign'].data,PRIORS,'benign')
    initial_gate_table=pd.concat([smoke_gate,benign_gate],ignore_index=True)
    initial_gate_table
    return (smoke_order,)


@app.cell(hide_code=True)
def fit_heading(mo):
    mo.md("""
    ## Fit

    Fits are serial and cached separately. The initial smoke is an execution check only. After the first main fit, stop if its measured runtime projects more than four hours for the planned synthetic sampling.
    """)
    return


@app.cell(hide_code=True)
def fit_helpers(
    CACHE_ROOT,
    MODEL_VERSION,
    asdict,
    cache,
    check_memory,
    datetime,
    jf,
    mo,
    np,
    perf_counter,
    sample_lean,
    stage_seed,
):
    import json

    def fit_field_case(data,priors,order,label,settings):
        if order not in (15,25):
            raise RuntimeError('Quadrature gate failed; no sampling is authorized.')
        seed=stage_seed(label,'fit')
        fingerprint=jf.field_fingerprint(data,priors,quadrature_order=order,sampler=settings,seed=seed,version=MODEL_VERSION)
        output=CACHE_ROOT/fingerprint
        model=jf.build_joint_field_model(data,priors=priors,quadrature_order=order)
        scalar_names=[variable.name for variable in model.free_RVs if variable.ndim==0]
        metadata={'fingerprint':fingerprint,'label':label,'chains':settings['chains'],'draws':settings['draws'],'settings':settings,'quadrature_order':order,'scalar_names':scalar_names,'model_version':MODEL_VERSION}
        trace=cache.load_cached_trace(output/'trace.nc',metadata,chains=settings['chains'],draws=settings['draws'])
        runtime_path=output/'runtime.json'
        elapsed=None
        if trace is None:
            for table in ['editions','registry','candidates','anonymous','demographics']:
                cache.atomic_write_frame(output/f'visible_{table}.parquet',getattr(data,table))
            cache.atomic_json(output/'provenance.json',{'metadata':metadata,'priors':asdict(priors),'seed':seed,'scope':'synthetic only; externally supplied registry'})
            keep=[variable.name for variable in model.unobserved_RVs]
            memory=check_memory(model,keep,chains=settings['chains'],draws=settings['draws'],tune=settings['tune'],on_disk=True)
            mo.output.replace(mo.callout(f'{label}: started {datetime.now():%H:%M}; projected storage peak {memory:.2f} GiB. First synthetic runtime is unmeasured; sampling is serial.',kind='info'))
            store=output/'trace.zarr'
            trace=cache.recover_trace_store(store,metadata=metadata,chains=settings['chains'],draws=settings['draws'],scalar_names=scalar_names)
            if trace is None:
                cache.prepare_trace_store(store,metadata=metadata)
                started=perf_counter()
                trace=sample_lean(model,keep,store_dir=store,random_seed=seed,**settings)
                elapsed=perf_counter()-started
                cache.atomic_json(runtime_path,{'sampling_seconds':elapsed})
            cache.save_trace_atomic(trace,output/'trace.nc',metadata)
        if runtime_path.exists():
            elapsed=json.loads(runtime_path.read_text())['sampling_seconds']
        # nutpie's zarr path has positional chain/draw dimensions without labels.
        # Materialize those existing positions; never replace supplied labels.
        for group in trace.groups():
            dataset=trace[group]
            missing={dim:np.arange(dataset.sizes[dim]) for dim in ('chain','draw') if dim in dataset.dims and dim not in dataset.coords}
            if missing:
                trace[group]=dataset.assign_coords(missing)
        cache.save_trace_atomic(trace,output/'trace.nc',metadata)
        audit=jf.audit_posterior_quadrature(trace.posterior,data,quadrature_order=order,draws_per_chain=min(20,settings['draws']))
        cache.atomic_write_frame(output/'posterior_quadrature.parquet',audit.to_dataframe().reset_index())
        cache.atomic_json(output/'posterior_quadrature_status.json',dict(audit.attrs))
        return {'trace':trace,'model':model,'output':output,'metadata':metadata,'sampling_seconds':elapsed,'quadrature':audit}


    return (fit_field_case,)


@app.cell(hide_code=True)
def smoke_fit(
    PRIORS,
    SMOKE_SAMPLER,
    fit_field_case,
    pd,
    smoke_order,
    smoke_simulation,
):
    smoke_result=fit_field_case(smoke_simulation.data,PRIORS,smoke_order,'smoke',SMOKE_SAMPLER)
    smoke_status=pd.DataFrame([{'case':'smoke','sampling_seconds':smoke_result['sampling_seconds'],'chains':smoke_result['trace'].posterior.sizes['chain'],'draws':smoke_result['trace'].posterior.sizes['draw'],'posterior_quadrature_passed':smoke_result['quadrature'].attrs['passed'],'max_posterior_loglik_difference':float(smoke_result['quadrature']['absolute_loglik_difference'].max())}])
    smoke_status
    return (smoke_result,)


@app.cell(hide_code=True)
def quadrature_failure_diagnostic(
    jf,
    mo,
    np,
    pd,
    smoke_result,
    smoke_simulation,
):
    smoke_audit_errors=smoke_result['quadrature']['absolute_loglik_difference']
    worst_audit_position=np.unravel_index(int(smoke_audit_errors.values.argmax()),smoke_audit_errors.shape)
    worst_audit_chain=smoke_audit_errors.chain.values[worst_audit_position[0]]
    worst_audit_draw=smoke_audit_errors.draw.values[worst_audit_position[1]]
    worst_audit_did=str(smoke_audit_errors.did.values[worst_audit_position[2]])
    worst_audit_posterior=smoke_result['trace'].posterior.sel(chain=worst_audit_chain,draw=worst_audit_draw)
    worst_audit_parameters={name:np.asarray(worst_audit_posterior[name].values) for name in jf.PARAMETER_DIMS}
    worst_order_values={order:jf.field_loglik_numpy(worst_audit_parameters,smoke_simulation.data,quadrature_order=order).values for order in [15,25,35]}
    worst_order_comparison=pd.DataFrame({'did':smoke_audit_errors.did.values,'difference_15_25':np.abs(worst_order_values[15]-worst_order_values[25]),'difference_25_35':np.abs(worst_order_values[25]-worst_order_values[35])})
    worst_parameter_values=pd.DataFrame([{'parameter':name,'value':float(worst_audit_posterior[name])} for name in ['omega','omega_dnf','rho_runner','quadratic','tau_edition_dnf']])
    mo.vstack([worst_order_comparison,worst_parameter_values])
    return (
        smoke_audit_errors,
        worst_audit_chain,
        worst_audit_did,
        worst_audit_draw,
    )


@app.cell(hide_code=True)
def numerical_stop(
    ROOT,
    mo,
    smoke_audit_errors,
    smoke_result,
    worst_audit_chain,
    worst_audit_did,
    worst_audit_draw,
):
    mo.md(f"""
    ### Numerical stop

    The two-chain 50-draw smoke completed in **{smoke_result['sampling_seconds']:.1f} seconds** and is saved under `{smoke_result['output'].relative_to(ROOT)}`. This is an execution check, not a usable recovery fit.

    The pre-fit 15/25-node gates passed at the initial and two fixed perturbed parameter points. At 20 retained smoke draws per chain, the maximum per-edition log-likelihood difference was **{float(smoke_audit_errors.max()):.5f}**, exceeding the **0.001** tolerance. The most discrepant audited point is chain {worst_audit_chain}, draw {worst_audit_draw}, edition `{worst_audit_did}`. The additional 25/35 comparison above is diagnostic only; it does not authorize a higher-order refit or establish a converged integral.

    All six planned recovery/sensitivity fits are stopped before launch. No thresholds, priors or seeds were changed, and no valid cache was deleted. The prototype has mathematical and execution tests, but **numerical readiness is unresolved**. Recovery, identification and prior-sensitivity results are not available.

    A nutpie/zarr integration issue was repaired without resampling: its chain/draw dimensions lacked explicit coordinate labels, so the notebook materialized the existing positional indices before validation. The completed smoke trace was reused.
    """)
    return


@app.cell(hide_code=True)
def recovery_heading(mo):
    mo.md("""
    ## Recovery and missing counts

    Four fixed-truth scenarios and two predeclared field-size prior sensitivity fits. Report weakly identified quantities and failed diagnostics; do not tune until every truth falls in an interval.
    """)
    return


@app.cell(hide_code=True)
def recovery_checkpoint(PRIORS, SCENARIOS, pd):
    recovery_plan=pd.DataFrame([{'run':name,'scenario':name,'anonymous_mean_prior_sd':PRIORS.anonymous_mean_sd} for name in SCENARIOS]+[{'run':'participation_selection_prior_narrow','scenario':'participation_selection','anonymous_mean_prior_sd':.5},{'run':'participation_selection_prior_wide','scenario':'participation_selection','anonymous_mean_prior_sd':1.5}])
    recovery_plan['status']='Not run: smoke posterior quadrature failed'
    recovery_plan
    return


@app.cell(hide_code=True)
def identification_heading(mo):
    mo.md("""
    ## Identification

    Count recovery does not guarantee separation of field size, entrant strength, course difficulty and race severity. Inspect posterior dependence and sensitivity explicitly.
    """)
    return


@app.cell(hide_code=True)
def handoff_heading(ROOT, mo, smoke_result):
    mo.md(f"""
    ## Handoff

    **Status: stopped at the posterior quadrature gate.** Core implementation and execution tests pass, but the numerical integration is not accurate enough across the sampled smoke posterior. The six recovery/sensitivity runs are listed above as not run. The smoke trace and numerical evidence are saved at `{smoke_result['output'].relative_to(ROOT)}`.

    Resolve numerical integration before starting a main fit. Do not repair this by shrinking the scientific prior, dropping the problematic draws, or relaxing the tolerance. The retained illustrative pace priors are authorized only for synthetic testing and still need physical calibration before deployment.

    Real candidate exposure, anonymous repeat participation, outcome-dependent cleaning, reporting uncertainty and genuinely new evaluation data remain scientific design gates. No production model, 2.4 score or historical trace was modified.
    """)
    return


if __name__ == "__main__":
    app.run()
