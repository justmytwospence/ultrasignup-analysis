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
  `course_id` [UltraSignup-7yu].
- **`is_running_race` keyword filter** matches substrings (`ski` in "Whiskey", `ride` in
  "Pride"). Check how many races it wrongly drops on mirror data and switch to word
  boundaries if it matters.
- **Delete `data/legacy/`** (old DBs, caches, traces, the old checkouts) once the mirror's
  `import-legacy` is verified and the refits are done.

## New data the mirror makes possible

- **DNS-aware modeling.** `races.n_entrants_final` (race-eve roster) and
  `races.n_registered_not_listed` give the registered field even where race directors do not
  post DNS/DNF rows. This bears directly on the reporting model (nb 2) and unobserved DNFs (nb 4).
- **Point-in-time runner strength.** `results.race_score` is UltraSignup's per-race score at the
  time; the old `runner_rank` was the runner's *current* rank (leaks the future). Use it, or a
  runner effect, instead.
- **Hidden results** (`source = 'participant_history'`) add races whose results pages are not
  public. Their fields are incomplete; decide per model whether to include them.
- **Reporting-model covariates** [UltraSignup-ieg]: series (`eid`), organizer and field size
  are now available directly from `races`/`courses`.

## nb 1: finish times

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
