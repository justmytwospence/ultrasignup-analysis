# Plan: Reactivate the UltraSignup project

## Context

This project went dormant in Nov 2024 after being factored into two sibling repos (`scraper/` and `analysis/`). The codebase is in better shape than "abandoned" suggests — the scraper is mature and well-modularized (~8K LOC), the analysis layer has solid utilities, and the canonical DuckDB at `analysis/data/ultrasignup.duckdb` holds **77,333 races, 3,234,909 results, 1,741,596 participants** with sensible DNF-rate progression by distance (6% sub-50K → 26% over 100mi). All five UltraSignup API endpoints are still live; the only schema drift is `closestevents` now returning an additional `OpenRegistration` field. The goal of this plan is to (a) modernize harness setup so future Claude sessions are productive, (b) clean up known scraper rough edges, (c) migrate notebooks to Marimo so we can iterate with the `marimo-pair` skill, and (d) address conceptual concerns in the PyMC modeling.

## Repo structure

The project root is now a thin git repo (created by `bd init`) holding workspace tooling. `scraper/` and `analysis/` remain their own independent git repos. Don't `git add` the subrepos from the root — commit inside each subrepo independently. This is documented in the new root-level `CLAUDE.md`.

---

## Phase 1: Foundation & harness tooling — STATUS: largely complete

**Done in this planning session:**
- `bd init` ran at `/Users/spencerboucher/Projects/UltraSignup/`. `.beads/`, `.git/`, `.gitignore`, hooks installed; `bd` ready to use across both subrepos.
- Auto-generated `CLAUDE.md` and `AGENTS.md` at project root, then customized `CLAUDE.md` with real architecture / build / data-flow content (replacing the placeholders).
- `.claude/settings.json` registers `bd prime` as a SessionStart and PreCompact hook.
- `.mcp.json` at project root configures `mcp-server-duckdb` (read-only, pointing at `analysis/data/ultrasignup.duckdb`). Verified `uvx --from mcp-server-duckdb mcp-server-duckdb --help` boots cleanly. **Will be available after the next Claude Code restart.**
- Marimo skills scoped to analysis: `analysis/.claude/skills/` contains symlinks to `marimo-pair`, `marimo-notebook` (from `Projects/.agents/skills/`) and `jupyter-to-marimo` (from `Projects/fieldgoal/.claude/skills/`). They activate when the working directory is under `analysis/`.

**Still to do:**
- Clean stale logs in `scraper/logs/` (104 files, ~177MB from Oct–Nov 2024). Move to `scraper/logs/archive/` or delete; ensure `.gitignore` excludes future logs.
- Seed beads with the Phase 2–5 workstreams as issues (`bd create` with dependencies between phases).
- Restart Claude Code so the DuckDB MCP loads.

---

## Phase 2: Scraper cleanup (`scraper/`)

**Goal:** stabilize the scraper before re-running it for fresh data.

Critical files:
- `scraper/main.py` (1,759 lines — needs split)
- `scraper/ultrasignup_scraper/elevation_extractor.py` (266 lines — switch LLM)
- `scraper/ultrasignup_scraper/elevation_batch.py` (391 lines — same)
- `scraper/pyproject.toml`

Actions:

1. **Split `main.py` into `scraper/ultrasignup_scraper/cli/`.** Each `run_*_mode` function (≈100–200 lines each, lines 423–1700) becomes its own module: `cli/scrape_participants.py`, `cli/scrape_races.py`, `cli/scrape_results.py`, `cli/extract_elevations.py`, `cli/enrich_events.py`, etc. `main.py` shrinks to a thin click dispatcher (<300 lines).
2. **Dedupe `scrape_races_incremental`** — defined twice in `main.py` (lines 356 and 406). Remove the duplicate.
3. **Recreate `scraper/tests/`** matching the structure already declared in `scraper/pyproject.toml` (`testpaths = ["tests"]`):
   - `tests/conftest.py` — pytest fixtures
   - `tests/fixtures/` — JSON fixtures captured from live API now while endpoints are confirmed working (see endpoint list in `scraper/README.md` lines 192–407)
   - `tests/unit/` — parser tests (`races.py`, `results.py`, `participants.py`), distance/time extractors, database operations
   - `tests/integration/` — end-to-end CLI smoke tests against fixture-mocked APIs
   - Target ~70%+ coverage on the `ultrasignup_scraper/` package
4. **Switch elevation extractor to Anthropic Claude Haiku 4.5.** Rewrite `elevation_extractor.py` and `elevation_batch.py` using the `claude-api` skill. Use prompt caching on the static system prompt; use Anthropic's web_fetch tool to grab race websites. Update `pyproject.toml`: drop `openai`, add `anthropic`. Drop `beautifulsoup4` (no longer needed per `ELEVATION_EXTRACTION_SIMPLIFIED.md`). Update README cost/throughput estimates.
5. **Add `enrich-events` CLI command.** The `races` table has Event Search API fields (EventId, latitude, longitude, event_website, GroupId, group_name) declared in `scraper/ultrasignup_scraper/database/schema.py:86-96` but no CLI command populates them. Wire one up using the existing `api/event_search.py` client.
6. **Document API drift.** Add `OpenRegistration` to the closestevents schema doc in `scraper/README.md` (around line 349) and to `parsing/races.py` if the parser stores it.

---

## Phase 3: Data validation & top-up scrape

**Goal:** confirm the existing DuckDB is sound, then add fresh data from the past 6+ months.

Known data oddities discovered during planning (must be addressed):
- Earliest `race_date` is 1931-03-31 — almost certainly a parsing bug. Hunt down and fix `core/distance_extractor.py` or `parsing/races.py` (or filter at query time in the validation script).
- Latest `race_date` is 2026-12-29 — these are upcoming events. Decide whether to treat upcoming races as a separate concept or include them.

Actions:

1. **Add `scraper/scripts/validate_data.py`** running sanity queries against `analysis/data/ultrasignup.duckdb` via the new DuckDB MCP (or directly via duckdb-python). Queries:
   - Race count by year, flag pre-1990 and post-today rows
   - DNF rate by distance bucket (already verified during planning: 6% / 10% / 14% / 24% / 26% across 50K/50mi/100K/100mi/100mi+ — sanity-check matches)
   - Null/missing-field counts on key columns (`time_ms`, `event_distance_id`, `gender`, `distance_km`)
   - `time_ms` outliers (< 1 hour or > 100 hours)
   - Participant uniqueness — the scraper's unique constraint on `(first_name, last_name, age, gender, city, state)` will collide on common names; quantify how often
   - `races.last_scraped_at` distribution to find races whose results were never fetched
2. **Snapshot the DB** (`cp ultrasignup.duckdb ultrasignup_pre_topup.duckdb`) before running the top-up scrape.
3. **Run incremental top-up scrape** from `scraper/`:
   ```
   uv run python main.py scrape-races-incremental --since-date 2024-11-01 --max-workers 6
   uv run python main.py scrape-results --max-workers 8
   ```
4. **Re-run validation script** and diff counts before/after.

---

## Phase 4: Marimo migration

**Goal:** convert all 5 notebooks to `.py` marimo notebooks so we can use the `marimo-pair` skill for reactive iteration.

Source notebooks at `analysis/notebooks/`:
- `0_kcore.ipynb` (19MB — mostly visualization output)
- `1_finish_times.ipynb` (95KB)
- `2_observed_dnfs.ipynb` (7.5MB)
- `3_dnf_reporting.ipynb` (879KB)
- `4_unobserved_dnfs.ipynb` (990KB)

**Use the `jupyter-to-marimo` skill** (symlinked into `analysis/.claude/skills/` during planning) to drive the conversion of each notebook. Per that skill's procedure:

1. For each `.ipynb`, run `uvx marimo convert notebooks/<n>.ipynb -o notebooks/<n>.py` **before** reading the file. The skill emphasizes this saves tokens — the converted .py is much smaller than the .ipynb (which carries embedded plots).
2. Run `uvx marimo check notebooks/<n>.py` and fix anything it reports.
3. Read the converted .py and clean up: ensure script-metadata block lists all required packages, drop leftover `display()` calls or `%magic` lines, ensure each cell's final expression is the value to render, and port any LaTeX or ipywidgets per the skill's `references/latex.md` and `references/widgets.md`. The PyMC notebooks shouldn't have ipywidgets, but `mcmc_notifications.py` interactions and any tqdm progress bars need a once-over.
4. Run `uvx marimo check` again and confirm clean.
5. **Verify each converted notebook runs end-to-end** before discarding the .ipynb. Cached MCMC traces (`analysis/notebooks/models/model_2/...nc`) and the pickled k-core caches (`analysis/data/cache/`) survive — they're loaded by file path, not embedded in the notebook.
6. **Quarantine .ipynb files** in `analysis/notebooks/legacy_ipynb/` for one cycle in case the conversion misses something subtle. Delete after Phase 5 verifies the marimo versions still produce correct results.
7. **Confirm `marimo` is in `analysis/pyproject.toml`** dependencies. If not, add it (script metadata may already pull it in, but having it in the project's lockfile keeps `uv sync` honest).
8. **Verify `marimo-pair` activates** when working under `analysis/` — this requires the symlinks added in Phase 1 plus a Claude restart.

---

## Phase 5: Modeling fixes (in marimo)

**Goal:** address conceptual concerns identified in the notebooks. Tackle in priority order; each item gets its own beads task.

**Critical files to modify:**
- `analysis/notebooks/1_finish_times.py`, `2_observed_dnfs.py`, `4_unobserved_dnfs.py` (PyMC models)
- `analysis/utils/empirical_priors.py` (29KB — prior elicitation)
- `analysis/utils/data_processing.py` (data split for prior/likelihood separation)

Reuse existing utilities: `calculate_pace_priors`, `calculate_dnf_priors`, `calculate_hierarchical_variance_priors`, `calculate_difficulty_dnf_coupling`, `calculate_reporting_priors` (all in `utils/empirical_priors.py`).

**Priority 1 — Empirical-prior data leakage (HIGH):**
`utils/empirical_priors.py` computes priors from finisher data that is also fed to the likelihood. Add an explicit train/prior split: hold out 20% of races for prior elicitation, fit the model on the remaining 80%. Document the split in `data_processing.py`. Alternative (cheaper): swap to weakly-informative priors derived from ultramarathon literature instead of from this dataset.

**Priority 2 — Likelihood mismatch on finish times (notebooks 1, 2, 4):**
Notebook 1 puts Normal noise on log-pace then exponentiates; notebook 2 uses LogNormal with σ fixed at 0.15. Change to a proper LogNormal likelihood with σ as a learned parameter (`HalfNormal(0.3)` prior is a reasonable default). For notebook 4, allow σ to vary by gender or by course size.

**Priority 3 — Hierarchy identifiability in notebook 4:**
Three-level hierarchy (population → course → race) risks σ_race collapsing to zero. Fit a simpler two-level version (no race level) and compare via LOO-IC using `arviz.compare`. Inspect posterior widths on σ_race; if data doesn't support it, drop the race level.

**Priority 4 — Censoring penalty approximation (notebook 4):**
`-n_finishers[c] * log(E[P(finish)])` uses an expectation; bias depends on individual predictions. Replace with the exact marginal likelihood (integrate P(finish | x_i) over observed finishers) or use a posterior-predictive simulation to estimate bias.

**Priority 5 — Shared course effects across genders (notebooks 1, 2):**
Currently one `course_effect` vector is reused for M/F. Fit a sensitivity model with gender-specific course effects and compare. If correlation is high (>0.9), keep shared; otherwise split.

**Priority 6 — Reporting-model covariates (notebooks 3, 4):**
Distance-only reporting prior is simplistic. Add `EventId` (race-organization clustering) and field-size as covariates if available in the enriched data from Phase 2 step 5.

**Priority 7 — Selection-bias confounding (notebook 2):**
The LKJ correlation between course pace and DNF effects could reflect runner self-selection rather than course difficulty. Add a runner random effect (subset of repeat runners in the k-core) and refit; report whether the correlation persists.

For each fix: keep the prior model spec in a comment, report posterior parameter shifts, and check `divergences == 0` and `R-hat < 1.01`. Use `mcmc_notifications.py` (`utils/mcmc_notifications.py`) for long runs.

---

## Verification

End-to-end checks per phase:

- **Phase 1:** `bd ready` returns the seeded issues; root `CLAUDE.md` renders sensibly; after Claude restart, the DuckDB MCP appears as a tool and a query like `SELECT COUNT(*) FROM races` succeeds; the `marimo-pair` skill is listed as available when CWD is under `analysis/`.
- **Phase 2:** `cd scraper && uv run pytest` passes >70% coverage; `uv run python main.py estimate` works; `uv run python main.py extract-elevations --limit 5` succeeds against the new Claude-based extractor.
- **Phase 3:** `validate_data.py` reports zero unexpected nulls and explains the 1931 / future-date rows; race count post-topup is greater than pre-topup; spot-check 5 new races and confirm they appear in the DB with results.
- **Phase 4:** All 5 marimo notebooks open cleanly via `uv run marimo edit` and execute top-to-bottom without errors. Cached MCMC traces still load (notebook 2's `model_2/alpha3_beta233/...nc`).
- **Phase 5:** Each fixed model has documented posterior diagnostics (no divergences, R-hat<1.01, ESS>400) saved via `utils/utils.py:plot_posterior_diagnostics`. Compare posterior predictive checks to held-out data; document inference shifts (e.g., did selection-bias correction change estimated DNF rates).

---

## Suggested execution order

1. **Phase 1 leftovers** (~10 min) — clean logs, seed beads, restart Claude.
2. **Phase 2** (~few sessions) — most code changes; parallelizable into beads tasks.
3. **Phase 3** (~1 session, plus wallclock for the topup scrape) — depends on Phase 2 only for the new `enrich-events` command.
4. **Phase 4** (~1 session, mostly mechanical with `marimo convert`).
5. **Phase 5** (multi-session — substantive modeling work; one priority per session).
