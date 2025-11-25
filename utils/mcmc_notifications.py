"""
MCMC Notification Utility

Provides notification functionality for long-running MCMC sampling jobs.
Sends push notifications via ntfy.sh to keep track of sampling progress remotely.

Usage:
    from mcmc_notifications import notify, notify_mcmc_start, notify_mcmc_complete, notify_mcmc_error
    
    # Simple notification
    notify("Title", "Message", priority="default")
    
    # MCMC start notification
    notify_mcmc_start(model_name="Model 1", n_results=50000, draws=2000)
    
    # MCMC completion notification
    notify_mcmc_complete(
        model_name="Model 1",
        elapsed_time=1234.5,
        n_results=50000,
        effective_draws=1000,
        divergences=0,
    )
    
    # MCMC error notification
    notify_mcmc_error(model_name="Model 1", error_msg="Out of memory", elapsed_time=500)
"""

import requests
from datetime import datetime

# Fixed notification topic (add manually in ntfy app)
# Note: Each model can override this if needed
NTFY_TOPIC = "ultrasignup-mcmc"


def notify(title, msg, priority="default", topic=None):
    """
    Send phone notification via ntfy.sh - emoji-safe version
    
    Args:
        title: Notification title (plain text, no emojis)
        msg: Notification message body (UTF-8 safe, can include emojis)
        priority: Priority level ("low", "default", "high", "urgent")
        topic: Optional override for NTFY_TOPIC
        
    Returns:
        True if notification sent successfully, False otherwise
    """
    if topic is None:
        topic = NTFY_TOPIC
        
    try:
        response = requests.post(
            f"https://ntfy.sh/{topic}", 
            data=msg.encode('utf-8'),
            headers={
                "Title": title,  # Plain ASCII/Latin-1 safe title
                "Priority": priority
            },
            timeout=10
        )
        response.raise_for_status()
        print(f"Notification sent: {title}")
        return True
    except Exception as e:
        print(f"Warning: Notification failed: {e}")
        return False


def build_mcmc_summary_string(trace, hyperparam_vars):
    """
    Build formatted summary string from MCMC trace for display and notifications.
    
    Args:
        trace: arviz.InferenceData object from pm.sample()
        hyperparam_vars: List of hyperparameter variable names to summarize
        
    Returns:
        str: Multi-line formatted summary string with diagnostics
    """
    import arviz as az
    
    # Extract basic statistics
    divergences = int(trace.sample_stats.diverging.sum().values)
    n_chains = trace.posterior.dims['chain']
    draws_per_chain = trace.posterior.dims['draw']
    total_draws = n_chains * draws_per_chain
    div_pct = 100 * divergences / total_draws
    
    # Get convergence diagnostics
    summary = az.summary(trace, var_names=hyperparam_vars, kind='diagnostics')
    rhat_max = summary['r_hat'].max() if 'r_hat' in summary.columns else None
    ess_bulk_min = summary['ess_bulk'].min() if 'ess_bulk' in summary.columns else None
    
    # Build summary lines
    lines = [
        f"📊 Sampling: {draws_per_chain:,} draws × {n_chains} chains = {total_draws:,} total",
        f"🔬 Divergences: {divergences} ({div_pct:.2f}%)"
    ]
    
    if rhat_max is not None:
        if rhat_max < 1.01:
            lines.append(f"✅ Converged: Rhat max = {rhat_max:.4f}")
        else:
            lines.append(f"⚠️ Check convergence: Rhat max = {rhat_max:.4f}")
    
    if ess_bulk_min is not None:
        lines.append(f"📈 ESS bulk min: {ess_bulk_min:.0f}")
    
    return "\n".join(lines)


def notify_mcmc_start(model_name, n_results=None, draws=None, thin=None, target_accept=None, message=None, topic=None):
    """
    Send notification when MCMC sampling starts
    
    Args:
        model_name: Name of the model (e.g., "Model 1", "Model 5b")
        n_results: Number of observations in dataset (optional if message provided)
        draws: Number of MCMC draws (optional if message provided)
        thin: Thinning factor (optional)
        target_accept: Target acceptance rate (optional)
        message: Pre-formatted message string (optional, overrides individual params)
        topic: Optional override for NTFY_TOPIC
    """
    if message is not None:
        # Use pre-formatted message
        msg = f"🚀 {model_name}\n{message}"
    else:
        # Fall back to building message from individual parameters
        msg_parts = [f"🚀 {model_name}: {n_results:,} results, {draws} draws"]
        
        if thin is not None:
            msg_parts.append(f"thin={thin}")
        if target_accept is not None:
            msg_parts.append(f"accept={target_accept}")
            
        msg = ", ".join(msg_parts)
    
    return notify("MCMC Started", msg, priority="low", topic=topic)


def notify_mcmc_complete(
    model_name,
    elapsed_time,
    n_results=None,
    effective_draws=None,
    divergences=None,
    n_chains=4,
    summary_text=None,
    topic=None,
    hyperparams=None,
    trace=None,
    **kwargs
):
    """
    Send notification when MCMC sampling completes successfully
    
    Args:
        model_name: Name of the model (e.g., "Model 1", "Model 5b")
        elapsed_time: Total elapsed time in seconds
        n_results: Number of observations in dataset (optional)
        effective_draws: Number of effective draws per chain (after thinning) (optional)
        divergences: Number of divergences (optional)
        n_chains: Number of chains (default: 4)
        summary_text: Pre-computed summary string from build_mcmc_summary_string (optional)
        topic: Optional override for NTFY_TOPIC
        hyperparams: List of hyperparameter names (ignored, for compatibility)
        trace: Arviz InferenceData object (ignored, for compatibility)
        **kwargs: Additional arguments (ignored, for compatibility)
    """
    completion_time = datetime.now().strftime("%I:%M %p")
    elapsed_min = elapsed_time / 60
    
    msg_lines = [
        f"✅ {model_name}",
        f"Finished at {completion_time} ({elapsed_min:.1f} min)"
    ]
    
    # If summary_text provided, use it; otherwise fall back to individual params
    if summary_text:
        msg_lines.append(summary_text)
    else:
        if n_results is not None and effective_draws is not None:
            msg_lines.append(f"📊 {n_results:,} results, {effective_draws:,} draws × {n_chains} chains")
        
        if divergences is not None:
            msg_lines.append(f"🔬 {divergences} divergences")
    
    msg_lines.append("Ready for analysis!")
    
    return notify(
        "MCMC Complete",
        "\n".join(msg_lines),
        priority="high",
        topic=topic
    )


def notify_mcmc_error(model_name, error_msg, elapsed_time=None, topic=None):
    """
    Send notification when MCMC sampling fails
    
    Args:
        model_name: Name of the model (e.g., "Model 1", "Model 5b")
        error_msg: Error message (will be truncated to 100 chars)
        elapsed_time: Time elapsed before failure in seconds (optional)
        topic: Optional override for NTFY_TOPIC
    """
    completion_time = datetime.now().strftime("%I:%M %p")
    error_short = str(error_msg)[:100]
    
    msg_lines = [
        f"❌ {model_name}",
        f"Failed at {completion_time}"
    ]
    
    if elapsed_time is not None:
        elapsed_min = elapsed_time / 60
        msg_lines.append(f"({elapsed_min:.1f} min)")
    
    msg_lines.append(f"{error_short}")
    
    return notify(
        "MCMC Failed",
        "\n".join(msg_lines),
        priority="urgent",
        topic=topic
    )
