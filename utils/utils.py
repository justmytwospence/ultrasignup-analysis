"""
Plotting utilities for Bayesian model diagnostics and visualization.

Provides reusable plotting functions for PyMC models across different notebooks.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_posterior_diagnostics(trace, hyperparam_vars, model_dir, tune, draws, target_accept):
    """
    Generate comprehensive posterior diagnostic plots for a PyMC trace.
    
    Creates a single diagnostic figure with three panels:
    1. ESS evolution during sampling (top left)
    2. Autocorrelation by lag (top right)
    3. Rank plot for visual convergence check (bottom, full width)
    
    Parameters
    ----------
    trace : arviz.InferenceData
        The PyMC trace object containing posterior samples
    hyperparam_vars : list of str
        List of hyperparameter variable names to plot
    model_dir : str
        Directory path where diagnostics plot will be saved
    tune : int
        Number of tuning steps (for filename)
    draws : int
        Number of posterior draws (for filename)
    target_accept : float
        Target acceptance rate (for filename)
    
    Returns
    -------
    None
        Displays plots and saves diagnostics plot to file
    """
    # Calculate divergence statistics
    divergences = trace.sample_stats.diverging.sum().values
    total_draws = trace.posterior.dims['chain'] * trace.posterior.dims['draw']
    div_pct = 100 * divergences / total_draws
    print(f"\nDivergences: {divergences} ({div_pct:.2f}%)")
    
    # Create a 3-panel diagnostic plot: 2 on top row, 1 on bottom row (full width)
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    # Panel 1: ESS plot (top left)
    ax_ess = fig.add_subplot(gs[0, 0])
    ax_ess.set_facecolor('white')
    az.plot_ess(
        trace,
        var_names=hyperparam_vars,
        kind='evolution',
        ax=ax_ess
    )
    ax_ess.set_title('ESS Evolution During Sampling', fontsize=12, fontweight='bold')
    ax_ess.axhline(y=400, color='red', linestyle='--', alpha=0.5, label='Min threshold (400)')
    ax_ess.grid(True, color='lightgray', alpha=0.7, linewidth=0.5)
    ax_ess.legend()
    
    # Panel 2: Autocorrelation plot (top right)
    ax_autocorr = fig.add_subplot(gs[0, 1])
    ax_autocorr.set_facecolor('white')
    az.plot_autocorr(
        trace,
        var_names=hyperparam_vars,
        max_lag=100,
        ax=ax_autocorr
    )
    ax_autocorr.set_title('Autocorrelation by Lag', fontsize=12, fontweight='bold')
    ax_autocorr.grid(True, color='lightgray', alpha=0.7, linewidth=0.5)
    
    # Panel 3: Rank plot (bottom row, full width)
    ax_rank = fig.add_subplot(gs[1, :])
    ax_rank.set_facecolor('white')
    az.plot_rank(
        trace,
        var_names=hyperparam_vars,
        kind='bars',
        ax=ax_rank
    )
    ax_rank.set_title('Rank Plot: Visual Convergence Check', fontsize=12, fontweight='bold')
    ax_rank.grid(True, color='lightgray', alpha=0.7, linewidth=0.5)
    
    # Save diagnostic plot
    diagnostics_file = f'{model_dir}/diagnostics_tune{tune}_draws{draws}_accept{target_accept}.png'
    fig.savefig(diagnostics_file, dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plot saved to {diagnostics_file}")
    plt.show()
