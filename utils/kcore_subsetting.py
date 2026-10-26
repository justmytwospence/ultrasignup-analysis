#!/usr/bin/env python3
"""
K-Core Subset Selection for Two-Stage Bayesian Modeling

This module provides k-core based data flagging with course-completion closure.

Usage:
    from kcore_subsetting import subset_kcore_data, evaluate_all_kcore_candidates
    
    # Step 1: Explore parameter space to find optimal (α, β)
    candidates = evaluate_all_kcore_candidates(results, max_entities=10000, verbose=True)
    # ... analyze candidates, pick (α, β) ...
    
    # Step 2: Apply chosen (α, β) to flag data
    results = subset_kcore_data(results, alpha=5, beta=25)
    model_data = results[results['in_kcore'] | results['in_closure']]
"""

import pandas as pd
import networkx as nx
from typing import Set, Tuple, List, Dict, Optional
from collections import deque
import time
import pickle
from pathlib import Path
import shutil


def _format_ranges(values: Set[int], step: int = 1) -> str:
    """
    Convert set of integers to minimal dash-separated ranges.
    
    Examples:
        {4, 5, 6, 10, 11, 20}, step=1 → "[4-6, 10-11, 20]"
        {5}, step=1 → "[5]"
        {1, 2, 3, 4}, step=1 → "[1-4]"
        {30, 32, 34, 36, 38}, step=2 → "[30-38]" (just show overall range)
    
    Parameters
    ----------
    values : Set[int]
        Set of integer values to format
    step : int, default=1
        Step size between consecutive values. If > 1, shows only min-max range.
        
    Returns
    -------
    str
        Formatted string with minimal contiguous ranges
    """
    if not values:
        return "[]"
    
    sorted_vals = sorted(values)
    
    # If step > 1, just show min-max range
    if step > 1:
        if len(sorted_vals) == 1:
            return f"[{sorted_vals[0]}]"
        return f"[{sorted_vals[0]}-{sorted_vals[-1]}]"
    
    # Original logic for step=1
    ranges = []
    start = sorted_vals[0]
    end = sorted_vals[0]
    
    for val in sorted_vals[1:]:
        if val == end + 1:
            end = val  # Extend current range
        else:
            # Save current range and start new one
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = val
    
    # Don't forget the last range
    ranges.append(f"{start}-{end}" if start != end else str(start))
    
    return "[" + ", ".join(ranges) + "]"


def subset_kcore_data(
    results: pd.DataFrame,
    alpha: int,
    beta: int
) -> pd.DataFrame:
    """
    Flag race results using (α, β)-core decomposition with course-completion closure.
    
    This function identifies a densely-connected (α, β)-core of runners and courses,
    then flags ALL participant results at k-core courses to enable complete
    course-level modeling (course-completion closure).
    
    The method:
    1. Build bipartite graph (runners ↔ courses) from finisher data
    2. Find (α, β)-core where:
       - Every runner participated in ≥ α k-core courses
       - Every course had ≥ β k-core runners
    3. Flag results:
       - in_kcore: Both participant AND course are in the (α, β)-core
       - in_closure: Course is in k-core, but participant is not (course completion)
    
    Course-completion closure strategy:
    - K-core courses have 100% of their participants flagged (in_kcore OR in_closure)
    - Provides complete participant lists for estimating course-specific DNF rates
    - K-core runners may have incomplete race histories (some races outside k-core)
    - No additional courses added beyond the (α, β)-core
    
    Parameters
    ----------
    results : pd.DataFrame
        Processed results DataFrame with columns:
        - participant_id: Unique runner identifier
        - name: Course name
        - finished: Boolean indicating finish status
        - Additional columns preserved in output
    alpha : int
        Minimum number of k-core courses each k-core runner must have participated in.
        Higher α = more selective runners = smaller, denser core.
    beta : int
        Minimum number of k-core runners each k-core course must have had.
        Higher β = more selective courses = smaller, denser core.
        
    Returns
    -------
    pd.DataFrame
        Full results dataset with two additional boolean columns:
        - in_kcore: True if BOTH participant and course are in the (α, β)-core
        - in_closure: True if course is in k-core but participant is not
        - Results where neither flag is True are outside the k-core and closure
        
        Filter to modeling subset with:
        >>> model_data = results[results['in_kcore'] | results['in_closure']]
        
        Or filter to just k-core entities:
        >>> kcore_only = results[results['in_kcore']]
        
    Notes
    -----
    (α, β)-core properties:
    - Asymmetric thresholds allow independent control of runner/course selectivity
    - Smaller α or β = larger core, less dense
    - Larger α or β = smaller core, more dense
    - Use evaluate_all_kcore_candidates() to explore the parameter space
    
    Course-completion closure properties:
    - Every k-core course has ALL its participants in the flagged data
    - Enables unbiased estimation of course-specific DNF rates
    - K-core runners may be missing some of their race results
    - Total flagged data = k-core results + course-completion results
    
    Typical workflow:
    1. Use evaluate_all_kcore_candidates() to find optimal (α, β) via Pareto analysis
    2. Apply chosen (α, β) with this function to flag the data
    3. Stage 1 MCMC: Model k-core entities (in_kcore=True) with full convergence
    4. Stage 2 MAP: Model all flagged data, fixing k-core params from Stage 1
    
    Examples
    --------
    After exploring parameter space in kcore_optimization notebook:
    
    >>> # Apply optimal (α, β) chosen from Pareto frontier
    >>> results = subset_kcore_data(results, alpha=5, beta=25)
    >>> 
    >>> # Filter to k-core + closure for modeling
    >>> model_data = results[results['in_kcore'] | results['in_closure']]
    >>> 
    >>> # Check composition
    >>> print(f"K-core results: {results['in_kcore'].sum():,}")
    >>> print(f"Closure results: {results['in_closure'].sum():,}")
    >>> print(f"K-core courses: {results[results['in_kcore']]['name'].nunique():,}")
    >>> print(f"K-core runners: {results[results['in_kcore']]['participant_id'].nunique():,}")
    """
    # Build bipartite graph from finisher data
    # CRITICAL: Create course_id AFTER filtering to finishers (same as evaluate_all_kcore_candidates)
    finishers = results[results['finished']].copy()
    runners = set(finishers['participant_id'].unique())
    
    finishers['course_id'] = finishers.groupby(['name', 'distance_miles']).ngroup()
    courses = set(finishers['course_id'].unique())
    
    # Build graph
    G = nx.Graph()
    G.add_nodes_from([(f'r_{r}', {'bipartite': 0}) for r in runners])
    G.add_nodes_from([(f'c_{c}', {'bipartite': 1}) for c in courses])
    
    # Vectorized edge creation (3-5x faster than iterrows)
    edges = [(f'r_{pid}', f'c_{cid}') 
             for pid, cid in zip(finishers['participant_id'], finishers['course_id'])]
    G.add_edges_from(edges)
    
    # Find (α, β)-core
    kcore_runners, kcore_courses = _find_alpha_beta_core(G, runners, courses, alpha, beta)
    
    # Map course_id from finishers to full results DataFrame
    # CRITICAL: Use merge to preserve exact same course_id values from graph
    course_id_map = finishers[['name', 'distance_miles', 'course_id']].drop_duplicates()
    # Caller may already have a 'course_id' column with a different convention
    # (e.g., notebooks/4_unobserved_dnfs uses a string id "name||distance"). Drop
    # it before merging so we get a clean integer course_id rather than the
    # auto-suffixed course_id_x/course_id_y pandas would otherwise produce.
    if 'course_id' in results.columns:
        results = results.drop(columns=['course_id'])
    results = results.merge(course_id_map, on=['name', 'distance_miles'], how='left')
    
    # Course-completion closure: ALL participants at k-core courses
    kcore_course_ids = [int(c.split('_')[1]) for c in kcore_courses]
    
    # Flag results - keep as strings to match DataFrame dtype
    kcore_runner_ids = [r.split('_')[1] for r in kcore_runners]
    
    results['in_kcore'] = (
        results['participant_id'].isin(kcore_runner_ids) &
        results['course_id'].isin(kcore_course_ids)
    )
    
    results['in_closure'] = (
        results['course_id'].isin(kcore_course_ids) &
        ~results['participant_id'].isin(kcore_runner_ids)
    )
    
    # Drop the temporary course_id column
    results = results.drop(columns=['course_id'])
    
    return results


def evaluate_all_kcore_candidates(
    results: pd.DataFrame,
    max_entities: int,
    min_alpha: int = 2,
    min_beta: int = 10,
    beta_step: int = 1,
    verbose: bool = True,
    use_cache: bool = True,
    clear_cache: bool = False
) -> pd.DataFrame:
    """
    Evaluate all (α, β) combinations using BFS to find Pareto frontier.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results DataFrame with participant_id, name, finished, distance_miles
    max_entities : int
        Maximum total entities (runners + courses) budget
    min_alpha : int, default=2
        Minimum α threshold to explore
    min_beta : int, default=10
        Minimum β threshold to explore
    beta_step : int, default=1
        Step size for β increments during exploration.
        - beta_step=1: Explore every β value (default, finest granularity)
        - beta_step=2: Explore every other β value (2x faster, coarser)
        - beta_step=5: Explore every 5th β value (5x faster, coarsest)
        Use larger values for quick initial exploration, then refine with beta_step=1.
    verbose : bool, default=True
        Print progress during exploration
    use_cache : bool, default=True
        If True, load results from cache if available and save results to cache.
        Cache key is based on max_entities, min_alpha, min_beta, and beta_step.
    clear_cache : bool, default=False
        If True, delete all cached k-core candidate files before running.
        Useful for forcing fresh computation.
        
    Returns
    -------
    pd.DataFrame
        DataFrame of candidate configurations with metrics
        
    Examples
    --------
    # Quick coarse exploration with large beta steps
    >>> candidates_coarse = evaluate_all_kcore_candidates(
    ...     results, max_entities=5000, min_beta=30, beta_step=10
    ... )
    
    # Fine-grained search around interesting region
    >>> candidates_fine = evaluate_all_kcore_candidates(
    ...     results, max_entities=5000, min_beta=50, beta_step=1
    ... )
    """
    # Setup cache directory and key
    cache_dir = Path(__file__).parent.parent / "data" / "cache" / "kcore"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle cache clearing
    if clear_cache:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            if verbose:
                print("🗑️  Cleared k-core cache directory")
    
    # Generate cache key from parameters
    cache_key = f"candidates_maxent{max_entities}_alpha{min_alpha}_beta{min_beta}_step{beta_step}"
    cache_file = cache_dir / f"{cache_key}.pkl"
    
    # Check cache
    if use_cache and cache_file.exists():
        if verbose:
            print(f"✅ Loading cached k-core candidates from {cache_file.name}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Build graph once
    finishers = results[results['finished']].copy()
    runners = set(finishers['participant_id'].unique())
    
    finishers['course_id'] = finishers.groupby(['name', 'distance_miles']).ngroup()
    courses = set(finishers['course_id'].unique())
    
    G = nx.Graph()
    G.add_nodes_from([(f'r_{r}', {'bipartite': 0}) for r in runners])
    G.add_nodes_from([(f'c_{c}', {'bipartite': 1}) for c in courses])
    
    # Vectorized edge creation (3-5x faster than iterrows)
    edges = [(f'r_{pid}', f'c_{cid}') 
             for pid, cid in zip(finishers['participant_id'], finishers['course_id'])]
    G.add_edges_from(edges)
    
    if verbose:
        print(f"Built graph: {len(runners):,} runners, {len(courses):,} courses")
        print(f"Starting BFS exploration from ({min_alpha}, {min_beta})...")
    
    # BFS exploration with core caching for incremental computation
    candidates = []
    visited = set()
    core_cache = {}  # Cache computed cores for reuse
    queue = deque([(min_alpha, min_beta)])  # Use deque for O(1) popleft
    
    while queue:
        alpha, beta = queue.popleft()  # O(1) deque operation
        
        if (alpha, beta) in visited:
            continue
        visited.add((alpha, beta))
        
        # Try to use cached core from parent (α-1, β) or (α, β-1) for incremental computation
        initial_runners = None
        initial_courses = None
        
        # Prefer (α-1, β) as starting point (guaranteed subset)
        if (alpha - 1, beta) in core_cache:
            initial_runners, initial_courses = core_cache[(alpha - 1, beta)]
        # Fall back to (α, β-1) if available
        elif (alpha, beta - 1) in core_cache:
            initial_runners, initial_courses = core_cache[(alpha, beta - 1)]
        
        # Find core with timing (incremental if parent available)
        core_start = time.time()
        kcore_runners, kcore_courses = _find_alpha_beta_core(
            G, runners, courses, alpha, beta,
            initial_runners=initial_runners,
            initial_courses=initial_courses
        )
        core_time = time.time() - core_start
        
        # Cache this core for future incremental computations
        if kcore_runners and kcore_courses:
            core_cache[(alpha, beta)] = (kcore_runners.copy(), kcore_courses.copy())
        
        if not kcore_runners or not kcore_courses:
            continue
        
        # Course-completion closure
        kcore_course_ids = [int(c.split('_')[1]) for c in kcore_courses]
        
        closure_df = finishers[finishers['course_id'].isin(kcore_course_ids)]
        kcore_runner_ids = [str(int(r.split('_')[1])) for r in kcore_runners]  # Convert to string to match DataFrame
        
        closure_runners = set(closure_df['participant_id'].unique()) - set(kcore_runner_ids)
        
        total_entities = len(kcore_runners) + len(kcore_courses) + len(closure_runners)
        
        if max_entities is not None and total_entities > max_entities:
            continue
        
        # Calculate metrics
        kcore_results = len(closure_df[closure_df['participant_id'].isin(kcore_runner_ids)])
        closure_results = len(closure_df[closure_df['participant_id'].isin(closure_runners)])
        
        # Runner completeness: averaged across k-core runners, what % of their total courses are k-core
        runner_total_courses = finishers[finishers['participant_id'].isin(kcore_runner_ids)].groupby('participant_id')['course_id'].nunique()
        runner_kcore_courses = closure_df[closure_df['participant_id'].isin(kcore_runner_ids)].groupby('participant_id')['course_id'].nunique()
        avg_runner_completeness = (runner_kcore_courses / runner_total_courses * 100).mean()
        
        # Course completeness: averaged across k-core courses, what % of their total runners are k-core
        course_all_runners = closure_df.groupby('course_id')['participant_id'].nunique()
        course_kcore_runners = closure_df[closure_df['participant_id'].isin(kcore_runner_ids)].groupby('course_id')['participant_id'].nunique()
        avg_course_completeness = (course_kcore_runners / course_all_runners * 100).mean()
        
        candidate = {
            'alpha': alpha,
            'beta': beta,
            'kcore_runners': len(kcore_runners),
            'kcore_courses': len(kcore_courses),
            'closure_runners': len(closure_runners),
            'total_entities': total_entities,
            'kcore_results': kcore_results,
            'closure_results': closure_results,
            'total_results': kcore_results + closure_results,
            'density': (kcore_results + closure_results) / total_entities if total_entities > 0 else 0,
            'num_distances': len(closure_df['distance_miles'].unique()),
            'runner_course_ratio': (len(kcore_runners) + len(closure_runners)) / len(kcore_courses) if len(kcore_courses) > 0 else 0,
            'avg_runner_completeness': avg_runner_completeness,
            'avg_course_completeness': avg_course_completeness
        }
        
        candidate['core_time_ms'] = core_time * 1000  # Store timing
        candidates.append(candidate)
        
        # Expand search
        if (alpha + 1, beta) not in visited:
            queue.append((alpha + 1, beta))
        if (alpha, beta + beta_step) not in visited:
            queue.append((alpha, beta + beta_step))
        
        # In-place status update - show current frontier (queue contents)
        if verbose:
            # Extract alpha and beta values from current queue
            frontier_alphas = {a for a, b in queue}
            frontier_betas = {b for a, b in queue}
            alpha_str = _format_ranges(frontier_alphas, step=1) if frontier_alphas else f"[{alpha}]"
            beta_str = _format_ranges(frontier_betas, step=beta_step) if frontier_betas else f"[{beta}]"
            expansion_rate = len(candidates) / len(visited) if len(visited) > 0 else 0
            print(f"\rExploring α={alpha_str}, β={beta_str} | Visited: {len(visited)} | Valid: {len(candidates)} | Expansion: {expansion_rate:.1%} | Queue: {len(queue)} | entities={total_entities:,} courses={len(kcore_courses):,} runners={len(kcore_runners) + len(closure_runners):,} | {core_time*1000:.0f}ms", end='', flush=True)
    
    if verbose:
        print()  # New line after in-place updates
        print(f"Explored {len(visited)} (α, β) combinations")
        print(f"Found {len(candidates)} valid configurations")
    
    # Convert to DataFrame and add computed columns for analysis
    df = pd.DataFrame(candidates)
    
    if not df.empty:
        # Add columns expected by downstream analysis
        df['n_courses'] = df['kcore_courses']
        df['n_participants'] = df['kcore_runners'] + df['closure_runners']
    
    # Save to cache
    if use_cache and not df.empty:
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        if verbose:
            print(f"💾 Cached k-core candidates to {cache_file.name}")
    
    return df


def _find_alpha_beta_core(
    G: nx.Graph,
    runners: Set,
    courses: Set,
    alpha: int,
    beta: int,
    initial_runners: Optional[Set[str]] = None,
    initial_courses: Optional[Set[str]] = None
) -> Tuple[Set[str], Set[str]]:
    """
    Find (α, β)-core where runners have ≥α courses, courses have ≥β runners.
    
    Optimized with queue-based pruning and cached degree tracking:
    - Pre-computes degrees once instead of querying neighbors repeatedly
    - Only processes nodes affected by removals (queue-based)
    - Supports incremental computation from a previous core (5-10x faster)
    - Expected 10-20x speedup vs iterative full-graph scans
    
    Parameters
    ----------
    initial_runners : Set[str], optional
        Start from this subset of runners (e.g., previous core with lower α)
    initial_courses : Set[str], optional
        Start from this subset of courses (e.g., previous core with lower β)
    """
    # Use initial cores if provided (incremental mode), otherwise start with full graph
    if initial_runners is not None:
        kcore_runners = initial_runners.copy()
    else:
        kcore_runners = set([f'r_{r}' for r in runners])
    
    if initial_courses is not None:
        kcore_courses = initial_courses.copy()
    else:
        kcore_courses = set([f'c_{c}' for c in courses])
    
    # Pre-compute degrees (cached - avoids repeated neighbor queries)
    runner_degrees = {}
    for r in kcore_runners:
        neighbors = [c for c in G.neighbors(r) if c in kcore_courses]
        runner_degrees[r] = len(neighbors)
    
    course_degrees = {}
    for c in kcore_courses:
        neighbors = [r for r in G.neighbors(c) if r in kcore_runners]
        course_degrees[c] = len(neighbors)
    
    # Initialize removal queue with nodes that violate thresholds
    removal_queue = deque()
    for r in kcore_runners:
        if runner_degrees[r] < alpha:
            removal_queue.append(('runner', r))
    for c in kcore_courses:
        if course_degrees[c] < beta:
            removal_queue.append(('course', c))
    
    # Queue-based pruning: only process affected nodes
    while removal_queue:
        node_type, node = removal_queue.popleft()
        
        if node_type == 'runner':
            if node not in kcore_runners:
                continue
            kcore_runners.remove(node)
            del runner_degrees[node]
            
            # Update degrees of connected courses and queue if needed
            for c in G.neighbors(node):
                if c in course_degrees:
                    course_degrees[c] -= 1
                    if course_degrees[c] < beta:
                        removal_queue.append(('course', c))
        else:  # course
            if node not in kcore_courses:
                continue
            kcore_courses.remove(node)
            del course_degrees[node]
            
            # Update degrees of connected runners and queue if needed
            for r in G.neighbors(node):
                if r in runner_degrees:
                    runner_degrees[r] -= 1
                    if runner_degrees[r] < alpha:
                        removal_queue.append(('runner', r))
    
    return kcore_runners, kcore_courses
