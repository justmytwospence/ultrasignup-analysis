# Open questions

Known modeling problems and planned work, grouped by notebook.

## All notebooks

- **Refit on current data.** Run each notebook end to end through marimo-pair, in order, and
  compare headline posteriors with the last fit. Statuses are finish or DNF (DNS and DQ are
  dropped) and unknown ages are dropped rather than modeled as 0, so expect small shifts in
  DNF rates.
- **Plausibility bounds** in `process_results` (age 10 to 95, pace 5 to 60 min/mile) are first
  guesses. Check how many rows each bound drops.
- **Spot-check inferred distances and formats** (backyard ultras posted as distance races)
  before trusting the k-core.
- **Hidden results**: about a third of editions show results only on runner pages and are
  absent. If hiding correlates with race type or organizer, course coverage is biased;
  `events.results_hidden` and `races.results_state = 'hidden'` let us measure that.
- **Point-in-time runner strength** has to come from `results` itself (each runner's prior
  finishes before `race_date`, with `runner_nth_race`). `runners.current_rank` is a snapshot
  and leaks the future.

## 1.1 finish_courses and 1.2 finish_runners

- **Year trend**: mean residual drifts from about -5% in the 1990s to +3% in 2026, likely
  broadening fields. A smooth year term would absorb it; decide after the runner-effect model.
- **Per-course noise**: within-course SD varies more than one sigma per gender allows
  (championship fields tight, open races wide); a per-course sigma still collapses.
- **Terrain as a course covariate**: `courses.route_gain_m` / `route_loss_m` are measured from a
  matched GPS track. Gain per km would explain part of each course effect, letting the
  remainder mean "harder than its terrain". Filter on `route_confidence` and skip
  `route_distance_disagrees`.
- **Shared course effects across genders**: fit gender-specific effects as a sensitivity check;
  keep shared if their correlation is above about 0.9.

## 2.1 dnf_reporting

- **Distance enters the logit linearly in km**, while 2.2 and 2.3 use a log distance ratio.
  Align the functional forms.
- **Covariates**: series, organizer and field size are available from `races`, `courses` and
  `events`.
- **Runner pace-DNF shape** (2.2): DNFs are over-predicted by 1 to 3 points for runners just
  faster than average. A spline in the pace effect would close it.

## 2.3 dnf_unobserved

- **Hierarchy identifiability**: the population/course/race hierarchy risks sigma_race
  collapsing to zero. Fit a two-level version and compare with LOO.
- **Censoring penalty approximation**: `-n_finishers * log E[P(finish)]` uses an expectation.
  The zero-DNF mixture term uses a distance-only p while the Bernoulli uses the
  difficulty-adjusted p, and the selection-correction Potential omits the pace-dependent
  numerator. Replace with the exact marginal or a shared latent.
- **Informative quadratic prior**: `beta_dist_dnf_quad ~ N(0.5, 0.3)` looks tuned to the data.
  Justify it or widen it toward N(0, 0.5) and check sensitivity.

## Empirical priors (`utils/empirical_priors.py`)

- **Data leakage** (high priority): priors are computed from the same finisher data the
  likelihood sees. Hold out a share of races for prior elicitation, or use weakly informative
  priors from the literature.
- **Hygiene**: `calculate_course_variation` computes course residuals on finishers only (biased
  for high-DNF courses) and silently drops courses with fewer than 5 observations; pace prior
  functions assume miles without saying so.
