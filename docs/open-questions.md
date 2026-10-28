# Open questions

Known problems and planned work for the models, grouped by notebook. Ids in brackets are the
old beads issues (history in git before commit bc7e863) for traceability.

## Moving to mirror data (all notebooks)

The data layer reads the mirror now (`utils/mirror.py`, `utils/data_processing.py`), but no
notebook has been run on mirror data yet, and every cached trace under `data/legacy/` was fit
on the 2025 scrape.

- **Migrate and refit, one notebook at a time, in order 0 to 4** (through marimo-pair). For each:
  run end to end, compare k-core sizes and headline posteriors with the legacy fit, then
  adopt the mirror's vocabulary in the notebook (course_id, did, runner_id) where it helps.
- **Statuses changed meaning.** Starts are now finish (`finished`, `finished_small_field`,
  `unofficial`) or `dnf`; `dns` and `dq` are dropped. The old code counted any non-2 status
  with a time as a finish, so DQs with times and DNS rows behaved differently. Expect small
  shifts in DNF rates.
- **Unknown ages are NULL now** (they were 0), so `process_results` drops them instead of
  modeling them as age 0.
- **Course identity** is the mirror's `course_id` (series + distance, stable across years and
  renames). `process_results` sets `name = course_id`, so nb 1 and nb 3, which key courses by
  `name`, are already id-based. Once refit, simplify nb 3/4's own `name||distance` keys to
  `course_id` [UltraSignup-7yu]. `course_id` is NULL only until the backfill fetches a race's
  series; `process_results` drops such rows as a safety net.
- **Distances and formats are inferred by the mirror** from labels (`races.distance_km`,
  `race_format`, `fixed_time_unit`); `distance_label` and the page's `distance_type` sit
  alongside. 7,265 distance-format races have NULL `distance_km` (dropped); spot-check races
  whose format disagrees with `distance_type` (backyard ultras posted as distance races)
  before trusting the k-core.
- **Plausibility bounds** in `process_results` (age 10 to 95, pace 5 to 60 min/mile) are
  first guesses against the mirror's reported junk (ages under 5, finishes over 14 days).
  Check how many rows each bound drops on the first pull.
- **Races retracted since 2025 have no results.** The mirror imports nothing from the 2025
  scrape, so `data/legacy/ultrasignup.duckdb` is the only copy of those rows. Count what the
  refits lose (the mirror's `scripts/validate.py` lists them) before deleting `data/legacy/`.
- **Delete `data/legacy/`** (old DBs, caches, traces, the old checkouts) once the refits are done
  and the retracted-race loss above is judged acceptable.

## New data the mirror makes possible

- **DNF reporting is explicit.** `races.reports_dnfs` is false when a race posted no DNF or DNS
  rows (then `n_dnf`, `n_dns` and `dnf_rate` are NULL, not 0). Use it in the reporting model
  (nb 2) and in place of `filter_races_with_dnfs`'s course-level heuristic where it fits.
- **Reporting-model covariates** [UltraSignup-ieg]: series (`series_id`), organizer and field size
  are now available directly from `races`/`courses`/`events`.
- **Experience.** `results.runner_nth_race` (1 = debut) is a point-in-time covariate that does
  not leak the future, unlike the old `runner_rank` (the runner's *current* rank).

Limited by the mirror's scope (it models only races that have happened; entrant rosters are
dropped and runner histories deferred with no plan):

- **DNS-aware modeling** (nb 2, nb 4): there is no roster-based registered field. What exists
  is `status = 'dns'` where a results page lists non-starters (4.1% of rows), `races.n_dns`
  and `races.reports_dnfs`. Races that post no DNS rows stay unknown.
- **Point-in-time runner strength:** no per-race score (`race_score` came from histories).
  Compute it from `results` itself: each runner's prior finishes before `race_date`, with
  `runner_nth_race`. `runners.current_rank` is a snapshot as of now and leaks the future.
- **Hidden results**: about a third of editions show results only on runner pages and are
  absent from `results`. If hiding correlates with race type or organizer, course coverage is
  biased; `events.results_hidden` and `races.results_state = 'hidden'` let us measure that.

## nb 1: finish times

- **Year trend** (seen in nb 1's posterior predictive check): mean residual drifts from about
  -5% in the 1990s to +3% in 2026, likely broadening fields. Not in the model; a smooth year
  term still integrates out. Decide after the runner-effect model, which absorbs part of it.
- **Per-course noise**: within-course SD varies more than one sigma per gender allows
  (championship fields tight, open races wide); a per-course sigma still collapses.

- **Terrain as a course covariate** (mirror schema 1.2): `courses.route_gain_m` / `route_loss_m` are
  measured on a DEM from a matched GPS track. Gain per km would explain part of each course
  effect, letting the remainder mean "harder than its terrain". Filter on `route_confidence`
  (provisional values, being validated mirror-side) and skip `route_distance_disagrees`.

- **No runner effect** [UltraSignup-e4o]: course effects absorb field strength (destination
  races attract atypical fields), and within-runner correlation understates uncertainty. Add a
  non-centered runner intercept (nb 3 model 2 has one).
- **Shared course effects across genders** [UltraSignup-wq3]: one course-effect vector serves M
  and F. Fit gender-specific effects as a sensitivity check; keep shared if their correlation is
  above about 0.9.

## nb 2: DNF reporting

- **Distance enters the logit linearly in km** [UltraSignup-rl7], while nb 3/4 use a log
  distance ratio (plus a quadratic in nb 4). Align the functional forms.
- **Selection bias vs course difficulty** [UltraSignup-i6e]: the LKJ correlation between course
  pace and DNF effects may reflect runner self-selection. Add a runner effect on repeat runners
  and check whether the correlation survives.

## nb 4: unobserved DNFs

- **Unidentifiable at (3, 840)** [UltraSignup-ubz]: the cached trace has r_hat 2.9 to 4.6 on all
  hyperparameters, a huge sigma_obs and mu_logit_dnf = 157. Check unit consistency of observed
  times (as was wrong in nb 1), try a larger k-core, and simplify the hierarchy.
- **Hierarchy identifiability** [UltraSignup-6rg]: the population/course/race hierarchy risks
  sigma_race collapsing to zero. Fit a two-level version and compare with LOO.
- **Censoring penalty approximation** [UltraSignup-9sx]: `-n_finishers * log E[P(finish)]` uses
  an expectation. Also, the zero-DNF mixture term uses a distance-only p while the Bernoulli
  uses the difficulty-adjusted p, and the selection-correction Potential omits the
  pace-dependent numerator. Replace with the exact marginal or a shared latent.
- **Informative quadratic prior** [UltraSignup-do5]: `beta_dist_dnf_quad ~ N(0.5, 0.3)` looks
  tuned to the data. Justify it or widen it toward N(0, 0.5) and check sensitivity.

## Empirical priors (`utils/empirical_priors.py`)

- **Data leakage** [UltraSignup-nfw, high priority]: priors are computed from the same finisher
  data the likelihood sees. Hold out a share of races for prior elicitation, or use weakly
  informative priors from the literature.
- **Hygiene** [UltraSignup-0dy]: `calculate_course_variation` computes course residuals on
  finishers only (biased for high-DNF courses) and silently drops courses with fewer than 5
  observations; pace prior functions assume miles without saying so.
