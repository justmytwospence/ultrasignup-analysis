# Overnight notebook convergence run

## Context

This project has 5 marimo notebooks under `analysis/notebooks/` that form a graduated modeling stack — k-core selection → reporting model → finish-time model → observed-DNF joint model → unobserved-DNF joint model. They were converted from `.ipynb` to marimo (Phase 4 of the reactivation plan) but several are believed to have model-fit / convergence issues (see beads `UltraSignup-618`, `-6rg`, `-9sx`, `-nfw`). Several open beads issues hint that the notebooks compile but don't necessarily yield good posteriors.

The goal of this overnight run is to walk through the notebooks in dependency order, getting each one to **execute end-to-end** with **R̂ < 1.01 on all parameters of interest** and **a sensible posterior** (no divergences worth worrying about, plausible PPC). I am allowed to shrink the k-core to whatever size makes each fit tractable (the user explicitly said models would never converge on the full data overnight). Each fix gets committed atomically inside `analysis/`.

## Run order (dependency-driven)

The notebook numbers don't match the dependency graph. Notebook 2 needs `p_reports` from notebook 3, and notebooks 2/4 read hyperparameters cached by notebook 1. The right order is:

```
0_kcore  →  3_dnf_reporting  →  1_finish_times  →  2_observed_dnfs  →  4_unobserved_dnfs
```

Rationale: 0 has no MCMC and produces the (α, β) reference point. 3 is a fast race-level beta-binomial with no upstream deps and outputs `p_zero_dnfs_if_reporting` consumed by 2 and 4. 1 fits course/runner hyperparameters used by 2 and 4. 2 layers the LKJ-correlated DNF effects on top of 1's structure. 4 is the full joint model and depends on everything.

`data_validation.py` is a pure QA notebook with no models; I'll run it once at the start as a sanity check but won't iterate on it.

## Per-notebook strategy

### Convergence + speed protocol (applied to every notebook)

1. **Open the notebook in marimo** (`uv run marimo edit notebooks/<file>.py --headless --no-token` as a background task), then drive it via `marimo-pair` (HTTP execute + cell mutations).
2. **Run cells top-down** until I hit the first MCMC `pm.sample(...)`.
3. **Speed check on first sampler progress output** — within ~30 s of sampling start, read the iter/s rate from nutpie's progress. Kill criteria:
   - < 1 iter/s on a dev k-core → too slow, abort and shrink k-core
   - First chain not past warmup in 90 s → abort, shrink
   - Any "sampling stalled" symptom → abort, reparametrize
4. **Convergence checks** after sampling:
   - `az.summary(idata).r_hat.max()` < 1.01 across all variables of interest
   - `idata.sample_stats.diverging.sum()` == 0 (or ≤ a handful of divergences with small effect, with explicit justification)
   - `ess_bulk` ≥ 400 for primary parameters
5. **Fit checks**:
   - Posterior predictive sanity (means/quantiles in the right ballpark vs. observed)
   - Trace plots show good mixing (no funnel artifacts on key params)
6. **If converged**: commit (`feat: <notebook> converges at (α,β)=(...)`), move on.
7. **If not converged**: iterate in this order — (a) tighten k-core for identifiability, (b) bump `target_accept` to 0.95 or 0.99, (c) bump `tune` from 1000 → 2000, (d) reparametrize (non-centered hierarchies, LKJ → simpler priors), (e) tighten priors using `utils/empirical_priors.py`. Stop after **2 hours of wall-clock effort per notebook** — at that point file a beads issue with the failure mode, R̂ summary, and what was tried, then move on to the next notebook (per user instruction).

### 0_kcore.py — exploratory, no MCMC

Goal: just run end to end and capture the candidate Pareto frontier. No convergence concerns. Likely fast unless `evaluate_all_kcore_candidates` has an unbounded sweep — I'll cap `max_entities` if needed.

Output to keep: a few candidate `(α, β)` tuples I want to use for downstream notebooks, written as `bd remember "kcore-recommendation ..."`.

### 3_dnf_reporting.py — race-level beta-binomial

No k-core (race-level aggregation). Single MCMC fit, `tune=2000, draws=2000, target_accept=0.95` already set. Should converge in under 10 min if the data are clean. Common failure modes: pre-1990 race dates skewing distance bins (see `data_validation.py` which flags this), overdispersion `κ` with poor identification at high distance.

Reads: `analysis/data/cache/model_m5_dnf/reporting_trace.nc` if cached. If cached and looks good, just verify and move on. If not cached or R̂ > 1.01, refit.

Writes: `p_zero_dnfs_if_reporting` joined onto `race_stats` in cache. Critical for notebooks 2 and 4 — verify the output column exists before moving on.

### 1_finish_times.py — gender-specific power law + course effects

Currently hardcoded to `(α=5, β=25)` "dev". I'll start with **`(α=4, β=80)`** (moderate — bigger course panel for partial pooling than 25, smaller than the 840 reference). Adjust based on fit time and identifiability:

- If iter/s < target → drop to `(α=5, β=25)` or `(α=5, β=15)`
- If R̂ converges easily and there's headroom → bump to `(α=3, β=200)` for a richer fit before committing

Known issues to watch (from beads):
- `UltraSignup-618`: likelihood mismatch on finish times. σ is fixed at 0.15 — if PPC under/over-disperses, learn σ as a parameter with a tight prior (`HalfNormal(0.2)`).
- `UltraSignup-nfw`: empirical-prior data leakage. If priors are computed from the same k-core that's being fit, refactor to compute priors on a held-out slice. Don't refactor if the notebook converges as-is — file as known issue.

Cache target: `analysis/data/cache/model_1/alpha{α}_beta{β}/{mcmc_trace.nc, map_trace_data.nc}`.

### 2_observed_dnfs.py — joint finish-time + DNF with LKJ correlation

Heaviest notebook (91 cells, 4 PyMC models, grid search for runner shrinkage). Real risk of OOM on the 423-course dev config. Start with **`(α=3, β=300)`** (roughly halve the runner panel from the documented 423) and shrink aggressively if fit time blows up. The grid search for `γ_runner_std` is a major time sink — I'll **either skip it** (use the documented best value) or **reduce the grid to 3 points** for the overnight run.

Known issues (beads):
- `UltraSignup-618`: shared finish-time likelihood issue applies here too.
- `UltraSignup-9sx`: censoring penalty approximation. If divergences cluster around the LKJ correlation parameter, this is the likely culprit.

Reparametrization moves I'll try if it doesn't converge:
- LKJ with `eta=2` instead of `eta=1` for tighter prior on the correlation
- Non-centered runner effects (`runner_raw * σ + μ`)
- Drop the runner-effects extension (Stage 2b) — keep only Stages 1, 1b, 2

Cache: `analysis/data/cache/model_2/alpha{α}_beta{β}/`.

### 4_unobserved_dnfs.py — full joint reporting-aware model

Most complex; depends on caches from 1, 2, 3. Documented at `(α=3, β=629)`. Start with **`(α=3, β=200)`** to keep fit under ~15 min. Watch for:
- `UltraSignup-6rg`: hierarchy identifiability — three-level hierarchy (population → course → race → participant) may not be identifiable on a small subset. If divergences won't go away, flatten to two levels (course → race) and file a beads issue noting the deferral.
- The reporting correction multiplies through the likelihood — if log-prob is `-inf`, check for races with `p_reports=0` (degenerate).

If this one doesn't converge in 2 hours, I'll commit the current state, file a beads issue with diagnostics, and end the run there.

## Loop pseudocode (run for each notebook)

```
start = now()
choose initial (alpha, beta) from this plan
while now() - start < 2h:
    open marimo, run all cells top-down
    if pre-MCMC error: fix it (data load / column / import), continue
    on first pm.sample: monitor first 30s
        if iter/s too slow: kill, shrink (alpha, beta), continue
    on sample complete:
        check r_hat, divergences, ess
        if converged + good PPC:
            commit, bd close any related issue, break
        else:
            try next reparametrization step (target_accept → tune → reparam → priors)
on stuck: bd create issue with notes/r_hat/last (alpha, beta)/what was tried, move on
```

## Critical files

| File | Why |
|---|---|
| `analysis/notebooks/0_kcore.py` | Step 1; produces (α,β) reference |
| `analysis/notebooks/3_dnf_reporting.py` | Step 2; produces p_reports trace |
| `analysis/notebooks/1_finish_times.py` | Step 3; hyperparameters for 2 & 4 |
| `analysis/notebooks/2_observed_dnfs.py` | Step 4; heaviest, expect to shrink k-core |
| `analysis/notebooks/4_unobserved_dnfs.py` | Step 5; full joint model |
| `analysis/utils/kcore_subsetting.py` | `subset_kcore_data(results, alpha, beta)` reused |
| `analysis/utils/empirical_priors.py` | Prior elicitation — used by all model notebooks |
| `analysis/utils/data_processing.py` | `load_results / process_results / filter_races_with_dnfs` |
| `analysis/data/cache/` | Trace caches; per-(α,β) subdirs |
| `analysis/.claude/skills/marimo-pair/` | The skill that drives running notebooks |

## Tooling

- **Drive notebooks**: `marimo-pair` skill (HTTP execute against a running marimo kernel).
- **Sampler**: `nutpie` (already the macOS default — `pm.sample(..., nuts_sampler="nutpie")`).
- **Diagnostics**: `arviz.summary`, `arviz.plot_trace`, `arviz.plot_ppc`.
- **Data questions**: DuckDB MCP (`mcp__duckdb__query`) — already configured in `.mcp.json`.
- **Task tracking**: `bd` only (per project rules) — `bd update --claim`, `bd close`, `bd create` for stuck-on issues.
- **Per-notebook commit**: `cd analysis && git add notebooks/<file>.py utils/... && git commit -m "feat: <notebook> converges at (α=N, β=M)"`. No push (CLAUDE.md: user pushes).

## Verification (end-to-end)

For each completed notebook, before moving on:

1. `uv run marimo run notebooks/<file>.py --headless` exits 0 (full top-down run with no errors).
2. `az.summary(idata).r_hat.max() < 1.01` printed in a cell and visible in the notebook output.
3. Trace cache file exists at the expected path under `analysis/data/cache/<model>/`.
4. Posterior predictive check cell renders a plot that's visually plausible vs. observed data.
5. `bd close <relevant-issue>` for any beads issue resolved by the run.

Final wake-up handoff (morning):

- Status table with notebook → (α, β) used → R̂ max → fit time → committed-y/n → notes.
- Any new beads issues filed for stuck notebooks (with `bd list --label=overnight-stuck`).
- Reminder: user must `git push` inside each subrepo per project CLAUDE.md.
