"""
(α, β)-core subset selection on the runner-course graph.

Runners and courses form a bipartite graph with an edge wherever a runner started a course
(finished or DNF'd it, in any year). The (α, β)-core is the largest subgraph in which every
runner has started at least α of its courses and every course has at least β of its runners.
Edges count starts, not finishes: a finishes-only graph keeps runners who rarely DNF and
drops those who often do, which biases every DNF model fit on the core.

Courses are keyed by (name, distance_miles); with the mirror, name is the course_id, so that
is one key per course across all its editions.

Usage:
    candidates = evaluate_all_kcore_candidates(results)       # sweep (α, β)
    results = subset_kcore_data(results, alpha=3, beta=840)   # flag one choice
    model_data = results[results['in_kcore'] | results['in_closure']]

The core is found by peeling on a sparse incidence matrix. Cores are nested (raising α or β
only removes nodes), so the sweep starts each (α, β) from the core at the previous β, and
every candidate's metrics are sparse reductions: the full sweep takes seconds, not the
half hour the old networkx version needed.
"""

from __future__ import annotations

import hashlib
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "kcore"
# Bump when the graph definition or the metrics change, so old caches are not reused.
CACHE_VERSION = 2


@dataclass
class _Graph:
    runner_ids: np.ndarray      # participant_id per row of B
    course_keys: pd.DataFrame   # name, distance_miles, course_id per column of B
    B: sp.csr_matrix            # runners x courses, 1 where the runner started the course
    W: sp.csr_matrix            # runners x courses, number of result rows
    course_distance: np.ndarray


def _build_graph(results: pd.DataFrame) -> _Graph:
    """Incidence matrices over every start in `results` (process_results keeps only
    finishes and DNFs, so every row is a start)."""
    course_codes, course_index = pd.MultiIndex.from_frame(results[['name', 'distance_miles']]).factorize()
    runner_codes, runner_ids = pd.factorize(results['participant_id'])
    n_r, n_c = len(runner_ids), len(course_index)
    W = sp.csr_matrix(
        (np.ones(len(results), dtype=np.int32), (runner_codes, course_codes)), shape=(n_r, n_c)
    )
    W.sum_duplicates()
    B = W.copy()
    B.data[:] = 1
    course_keys = course_index.to_frame(index=False, name=['name', 'distance_miles'])
    course_keys['course_id'] = np.arange(n_c)
    return _Graph(np.asarray(runner_ids), course_keys, B, W, course_keys['distance_miles'].to_numpy())


def _peel(B: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray, alpha: int, beta: int) -> tuple[np.ndarray, np.ndarray]:
    """The (α, β)-core of B restricted to `rows` x `cols`, as index arrays into B."""
    sub = B[rows][:, cols]
    while rows.size and cols.size:
        keep_r = np.asarray(sub.sum(axis=1)).ravel() >= alpha
        rows, sub = rows[keep_r], sub[keep_r]
        keep_c = np.asarray(sub.sum(axis=0)).ravel() >= beta
        cols, sub = cols[keep_c], sub[:, keep_c]
        if keep_r.all() and keep_c.all():
            break
    if not (rows.size and cols.size):
        return rows[:0], cols[:0]
    return rows, cols


def _metrics(g: _Graph, rows: np.ndarray, cols: np.ndarray) -> dict:
    B_c = g.B[:, cols]  # every runner at the core courses
    W_c = g.W[:, cols]
    runners_at_core_courses = np.flatnonzero(B_c.getnnz(axis=1))
    in_core = np.zeros(g.B.shape[0], dtype=bool)
    in_core[rows] = True
    closure_runners = int((~in_core[runners_at_core_courses]).sum())

    total_results = int(W_c.sum())
    kcore_results = int(W_c[rows].sum())

    # Runner completeness: share of each core runner's courses that are core courses.
    runner_completeness = B_c[rows].getnnz(axis=1) / g.B[rows].getnnz(axis=1)
    # Course completeness: share of each core course's runners who are core runners.
    course_completeness = B_c[rows].getnnz(axis=0) / B_c.getnnz(axis=0)

    n_r, n_c = len(rows), len(cols)
    total_entities = n_r + n_c + closure_runners
    return {
        'kcore_runners': n_r,
        'kcore_courses': n_c,
        'closure_runners': closure_runners,
        'total_entities': total_entities,
        'kcore_results': kcore_results,
        'closure_results': total_results - kcore_results,
        'total_results': total_results,
        'density': total_results / total_entities,
        'num_distances': int(np.unique(g.course_distance[cols]).size),
        'runner_course_ratio': (n_r + closure_runners) / n_c,
        'avg_runner_completeness': float(runner_completeness.mean() * 100),
        'avg_course_completeness': float(course_completeness.mean() * 100),
    }


def subset_kcore_data(results: pd.DataFrame, alpha: int, beta: int) -> pd.DataFrame:
    """
    Flag results by the (α, β)-core, with course-completion closure.

    Returns `results` with two boolean columns:
    - in_kcore: the runner and the course are both in the core
    - in_closure: the course is in the core but the runner is not. Every start at a core
      course is flagged one way or the other, so course-level DNF rates see the full field.

    Model on `results[results['in_kcore'] | results['in_closure']]`.
    """
    g = _build_graph(results)
    rows, cols = _peel(g.B, np.arange(g.B.shape[0]), np.arange(g.B.shape[1]), alpha, beta)
    core_runners = set(g.runner_ids[rows])
    core_courses = g.course_keys.iloc[cols][['name', 'distance_miles']]
    at_core_course = (
        pd.MultiIndex.from_frame(results[['name', 'distance_miles']])
        .isin(pd.MultiIndex.from_frame(core_courses))
    )
    core_runner = results['participant_id'].isin(core_runners).to_numpy()
    results = results.copy()
    results['in_kcore'] = at_core_course & core_runner
    results['in_closure'] = at_core_course & ~core_runner
    return results


def _fingerprint(results: pd.DataFrame) -> str:
    """Hash of the graph's inputs, so a cached sweep is never reused on different data."""
    h = pd.util.hash_pandas_object(results[['participant_id', 'name', 'distance_miles']], index=False)
    return hashlib.sha1(h.to_numpy().tobytes()).hexdigest()[:12]


def evaluate_all_kcore_candidates(
    results: pd.DataFrame,
    max_entities: int | None = None,
    min_alpha: int = 2,
    min_beta: int = 10,
    beta_step: int = 1,
    verbose: bool = True,
    use_cache: bool = True,
    clear_cache: bool = False,
) -> pd.DataFrame:
    """
    Every (α, β) from (min_alpha, min_beta) upward whose core is non-empty, with size and
    completeness metrics, one row per configuration.

    max_entities drops configurations whose core plus closure exceeds that many runners and
    courses. The cache key includes a fingerprint of the data, so a new pull or a changed
    filter recomputes rather than loading a stale sweep.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if clear_cache:
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True)
    key = (
        f"candidates_v{CACHE_VERSION}_{_fingerprint(results)}"
        f"_maxent{max_entities}_alpha{min_alpha}_beta{min_beta}_step{beta_step}"
    )
    cache_file = CACHE_DIR / f"{key}.pkl"
    if use_cache and cache_file.exists():
        if verbose:
            print(f"Loading cached k-core candidates from {cache_file.name}")
        return pd.read_pickle(cache_file)

    start = time.time()
    g = _build_graph(results)
    if verbose:
        print(f"Graph: {g.B.shape[0]:,} runners, {g.B.shape[1]:,} courses, {g.B.nnz:,} runner-course edges")

    candidates = []
    rows_a, cols_a = np.arange(g.B.shape[0]), np.arange(g.B.shape[1])
    alpha = min_alpha
    while True:
        # Core at (α, min_beta), from the core at (α - 1, min_beta).
        rows_a, cols_a = _peel(g.B, rows_a, cols_a, alpha, min_beta)
        if not rows_a.size:
            break
        rows, cols, beta = rows_a, cols_a, min_beta
        while rows.size:
            t = time.time()
            rows, cols = _peel(g.B, rows, cols, alpha, beta)
            if not rows.size:
                break
            m = _metrics(g, rows, cols)
            if max_entities is None or m['total_entities'] <= max_entities:
                candidates.append({'alpha': alpha, 'beta': beta, **m, 'core_time_ms': (time.time() - t) * 1000})
            beta += beta_step
        if verbose:
            print(f"\rα={alpha}: β up to {beta - beta_step} | {len(candidates):,} configurations | {time.time() - start:.0f}s", end='', flush=True)
        alpha += 1
    if verbose:
        print(f"\nFound {len(candidates):,} configurations in {time.time() - start:.1f}s")

    df = pd.DataFrame(candidates)
    if not df.empty:
        df['n_courses'] = df['kcore_courses']
        df['n_participants'] = df['kcore_runners'] + df['closure_runners']
        if use_cache:
            df.to_pickle(cache_file)
    return df
