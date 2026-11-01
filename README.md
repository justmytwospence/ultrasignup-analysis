# UltraSignup analysis

Why do some ultramarathons chew people up while others are gentle? A 100-miler with 20,000 feet
of climbing and a flat 50k on a bike path both show up as "an ultra" on a results page, and a
fast time in one tells you little about the other. Raw finish times and DNF rates mix together
four things: how hard the course is, how strong the field was, what the weather did that year,
and who the runner is. This project pulls them apart.

It fits Bayesian generative models, in PyMC and marimo notebooks, to every UltraSignup result
(marathon to 100 miles). Each model says how a finish time, or a drop, could have come about,
and the posterior then answers questions that no leaderboard can: which course is hardest for
its terrain, who is unusually good on climbs, and how many runners quietly dropped from races
that never posted their DNFs. Every estimate comes with its uncertainty, so a small field or a
single odd year can't pass for a trend.

## Questions the models should answer

Each question names the model term that answers it. Ones marked (planned) need a term the
notebooks do not have yet (see Open questions below).

### Courses
- Which courses are hardest at a given distance? (course effect, 1.1)
- How much of a course's difficulty is field strength rather than the course? (course effects with and without runner effects, 1.1 vs 1.2)
- Which course is hardest for its terrain, independent of its length? (planned: course effect net of gain per km)
- Do destination races attract such strong fields that they look easier than they are? (1.2 vs 1.1)

### Race years
- Which years of a race were unusually hard or easy (heat, snow, mud) relative to that course's norm? (planned: race-year effect)
- Is there a trend in pace over the years, as fields broaden? (planned: year term)
- Which races post their DNFs, and what predicts it? (2.1)

### Runners
- Who is fastest, net of the courses they happened to run? (runner effect, 1.2)
- Who is disproportionately good on steep courses or long distances? (planned: runner-by-gain and runner-by-distance slopes)
- Who is consistent, and who is erratic? (planned: per-runner variance)
- How does a first-time ultrarunner compare with a veteran? (planned: experience term)

### DNFs
- Which courses drop the most runners at a given pace and distance? (course DNF effect, 2.2)
- Are hard courses also high-DNF courses, once runner quality is accounted for? (course pace-DNF correlation, 2.2)
- Do slower runners drop more often, and do the very fastest also drop more (going out too hard)? (runner pace-DNF curve, 2.2)
- Which runners are DNF-prone beyond what their pace predicts, and which are unusually durable? (runner DNF effect, 2.2)
- How does DNF risk change with distance, age and gender? (2.2)
- How much does a first-timer's risk of dropping differ from a veteran's? (planned: experience term)
- Does DNF risk rise with the course's climbing, or on hot or muddy years? (planned: terrain and race-year effects)
- Do runners who DNF once drop again more often, or bounce back? (planned: runner history)
- How many DNFs are hidden in races that post only finishers, and does leaving them out bias what we learn about course difficulty? (2.3)
- What share of a course's drops come from being a hard course, and what share from who showed up? (course vs runner DNF effects, 2.2)

### Causal questions
The data is observational, so these are answered by comparing a runner with themselves or a
course with its own norm, and each rests on a stated assumption. All are planned.
- Does experience make runners faster and less likely to drop, or do only the durable ones keep racing? (compare a runner's 1st, 2nd and 3rd race using `runner_nth_race`; assumes who stops entering is modeled, since quitters vanish)
- What does stepping up in distance do to the risk of dropping, for example a first 100-miler after 50-milers? (a runner's own record as the baseline; assumes the runner effect captures who they are)
- How much does one DNF change a runner's next race, in finish chance and pace? (before-and-after within a runner; unseen injury hits both races, so read it as an association)
- How much of the pace decline with age is aging, and how much is which cohort of runners shows up? (within-runner age changes against the pooled age curve)
- Is a course's effect the course, or the runners who chose it? (runners who ran several courses give same-runner comparisons; assumes runners do not pick courses by private suitability such as hill specialists choosing hilly races, which the runner-by-gain slope would test)
- How much do heat, snow or rain change pace and DNFs? (a race year against that course's own norm, a difference-in-differences; needs weather joined from an outside source)

Out of reach: training, shoes, nutrition, coaching and anything else not observed.

### Population
- How does pace scale with distance, and does that differ by gender? (1.1)
- How does pace change with age, and where is the peak? (1.1)

### Prediction
- Who will win a race? For every registered runner, what are the odds of winning, of a podium or top 10, and of finishing at all? (simulate the whole field from the joint finish and DNF posterior; planned: needs entrant rosters and a race-year effect for the day's conditions)
- What finish time, with an interval, should a given runner expect at a given course? (posterior predictive)
- What is the chance they finish at all, and at what pace is that chance best? (2.2, 2.3)

## Setup

```sh
uv sync
cp .env.example .env    # set ULTRASIGNUP_QUACK_URI and ULTRASIGNUP_QUACK_TOKEN
uv run pytest
```

## Data

Results come from a DuckDB database of UltraSignup races, courses, runners and results, read
over the network or from a local copy. Starts are finishes or DNFs (no-shows and
disqualifications are dropped). About a third of races hide their results, so they are
absent; courses are identified across years and renames, so a course effect pools every
year of a race.


```python
from utils import data_source
from utils.data_processing import load_results, process_results

con = data_source.connect()                    # live
results = process_results(load_results(con))

data_source.pull()                             # copy the tables to data/results.duckdb
con = data_source.connect(local=True)          # offline, or to pin a long fit to one snapshot
```

## Notebooks

[marimo](https://marimo.io) notebooks, run in order. Each adds one thing to the one before it. Model notebooks follow the same workflow: data, model with a prior predictive check, fit and convergence, posterior predictive checks, model comparison (PSIS-LOO), prior sensitivity, results. The validation notebook instead scores actual held-out predictions from refits.

| Notebook | What it does |
|---|---|
| `0.1_kcore.py` | Picks a dense (α, β)-core of runners and courses for models with runner effects |
| `1.1_finish_courses.py` | Pace by distance, gender and age, and course difficulty, on every finisher |
| `1.2_finish_runners.py` | Adds runner effects on the core, separating course difficulty from field strength |
| `2.1_dnf_reporting.py` | Which races post their DNFs |
| `2.2_dnf_observed.py` | Adds DNFs to 1.2: runner and course DNF tendencies, correlated with pace |
| `2.3_dnf_unobserved.py` | Adds races that post no DNFs, via 2.1's reporting model, and predicts their hidden DNFs |
| `2.4_dnf_validation.py` | Masks individual reporting races and refits to test hidden-DNF counts against a simple count baseline |

Names are `<track>.<step>_<topic>`: track 0 prepares data, 1 models finish times, 2 models
DNFs; a step builds on the earlier steps of its track, and a new model takes the next step
number.

```sh
uv run marimo edit notebooks/0.1_kcore.py
```

2.4 uses one fixed, distance/finisher-count-stratified 20% holdout of individual race
occurrences, keeping other years and sibling distances available. Masked races are known
nonreporters; it tests historical count reconstruction, not reporting classification or
forecasting. The target is the DNF count surviving the existing cleaning filters, conditional
on the fixed core. Validation artifacts are separate from production caches.

## Open questions

Known modeling problems and planned work, grouped by notebook.

### All notebooks

- **Refit on current data.** Run each notebook end to end through marimo-pair, in order, and
  compare headline posteriors with the last fit. Statuses are finish or DNF (DNS and DQ are
  dropped) and unknown ages are dropped rather than modeled as 0, so expect small shifts in
  DNF rates.
- **Plausibility bounds** in `process_results` (age 10 to 95, pace 5 to 60 min/mile) are first
  guesses. Check how many rows each bound drops.
- **Spot-check inferred distances and formats** (backyard ultras posted as distance races)
  before trusting the k-core.
- **Hidden results**: about a third of races show results only on runner pages and are
  absent. If hiding correlates with race type or organizer, course coverage is biased;
  `events.results_hidden` and `races.results_state = 'hidden'` let us measure that.
- **Point-in-time runner strength** has to come from `results` itself (each runner's prior
  finishes before `race_date`, with `runner_nth_race`). `runners.current_rank` is a snapshot
  and leaks the future.

### 1.1 finish_courses and 1.2 finish_runners

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

### 2.1 dnf_reporting

- **Distance enters the logit linearly in km**, while 2.2 and 2.3 use a log distance ratio.
  Align the functional forms.
- **Covariates**: series, organizer and field size are available from `races`, `courses` and
  `events`.
- **Runner pace-DNF shape** (2.2): DNFs are over-predicted by 1 to 3 points for runners just
  faster than average. A spline in the pace effect would close it.

### 2.3 dnf_unobserved

- **V2 large-core fit completed** on the pinned local snapshot: 4 chains, 2,000 warm-up and
  1,500 retained draws per chain, target acceptance 0.9. No divergences or tree-depth hits;
  population and runner effects pass the R-hat screen. Course/race-year separation still mixes
  slowly (10 course and 49 race-year coordinates have R-hat > 1.01; maximum 1.031), although
  their combined race-year logits mix well (maximum R-hat 1.005, minimum bulk ESS 2,249).
  Investigate this separation before reporting standalone course DNF rankings.
- **V2 predictive checks**: 93.75% coverage of nominal 94% count intervals across 1,344
  reporting race-years since 2010, but 59.6% coverage of nominal 50% intervals. Coverage varies
  by distance (94% intervals cover about 91% at 42–60 km versus 96% above 60 km). These are
  in-sample checks with fresh race-year effects, not held-out validation. PSIS remains unreliable
  for many observations; do not treat the single-model LOO table as a validated comparison.
- **Hierarchy identifiability**: the population/course/race hierarchy risks sigma_race
  collapsing to zero. Fit a two-level version and compare with LOO.
- **Censoring penalty approximation**: `-n_finishers * log E[P(finish)]` uses an expectation.
  The zero-DNF mixture term uses a distance-only p while the Bernoulli uses the
  difficulty-adjusted p, and the selection-correction Potential omits the pace-dependent
  numerator. Replace with the exact marginal or a shared latent.
- **Informative quadratic prior**: `beta_dist_dnf_quad ~ N(0.5, 0.3)` looks tuned to the data.
  Justify it or widen it toward N(0, 0.5) and check sensitivity.

### 2.4 dnf_validation

- **Masked-race refits completed:** 263 individual races across 114 evaluated courses, within
  the fixed 172-course core. The full pipeline took 4h14m on the pinned local snapshot.
  Reporting, baseline and joint fits had no retained-sample divergences or tree-depth hits.
- **The simpler count baseline wins this comparison:** mean WIS 9.03 versus 9.82 for the
  joint model (lower is better). Paired joint-minus-baseline WIS is +0.79, with a 94%
  course-clustered bootstrap evaluation interval of [0.30, 1.37]; equal-course weighting
  gives +0.72 [0.23, 1.24]. These are evaluation intervals, not posterior credible intervals.
- **Coverage does not imply sharp predictions:** nominal 94% intervals cover 94.7% of races
  for the baseline and 95.8% for the joint model, but their mean widths are 101.4 and 127.5
  DNFs. Of 9,680 actual held-out DNFs, predicted totals average 9,893 and 11,375 respectively.
  The predeclared second seed/800-donor check gives joint WIS 9.86 and coverage 95.4%, so the
  primary result is not reversed by that numerical check.
- **Interpretation remains conditional and provisional:** five course and 30 race-effect
  coordinates have R-hat above 1.01 (maximum 1.029), despite good population/runner screens.
  Fresh-effect-integrated predictive means also mix imperfectly (maximum R-hat 1.026,
  minimum bulk ESS 329); this is not merely a component-separation issue.
  This frozen-specification, out-of-fit test does not establish transportability to naturally
  nonreporting races, validate core selection, or count runners excluded by cleaning. It is
  not an untouched external test. Do not retune against this holdout and relabel it validation.

- **Saved-draw stability check:** all 16 cross-model chain comparisons favor the baseline
  (+0.69 to +0.95 WIS); early/late halves give +0.80/+0.79. The 13 targets failing integrated
  mean mixing screens account for about 15% of the summed difference; excluding them from
  both models leaves +0.70. Within-chain block bootstraps (25/50 saved draws, 200 replicates
  each) give Monte Carlo SD about 0.038 WIS. This supports the practical baseline preference
  within the explored chains, without certifying convergence or detecting unvisited modes.
  No new sampling, model revision or change to the primary score was made.

- **Prediction-stage bias diagnostic:** holding the fitted posterior and common donor sample
  fixed, integrating fresh race effects raises expected total DNFs from 8,823 (effect set to
  zero) to 11,291. Of the 2,468 increase, 871 reflects the probability shift and 1,597 the
  convexity of `p/(1-p)`. This is a diagnostic quadrature approximation, not a replacement
  prediction or a refit. Averaging conditional count means is mathematically appropriate;
  replacing it with a transform of average probability merely because that matches this
  holdout would be unjustified. Overprediction is concentrated above 90 km. About 98% of
  joint predictive moment variance comes from varying conditional means (race effects,
  posterior uncertainty and donor simulation together), rather than final count noise.
  Investigate race-effect/donor calibration; donor representativeness is not yet isolated.

### Empirical priors (`utils/empirical_priors.py`)

- **Data leakage** (high priority): priors are computed from the same finisher data the
  likelihood sees. Hold out a share of races for prior elicitation, or use weakly informative
  priors from the literature.
- **Hygiene**: `calculate_course_variation` computes course residuals on finishers only (biased
  for high-DNF courses) and silently drops courses with fewer than 5 observations; pace prior
  functions assume miles without saying so.

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md): free for personal use, research, education and noncommercial organizations. Commercial use, including building the methodology into a product or service, requires a separate license; contact Spencer Boucher.

Outside contributions require agreeing to a contributor license agreement; open an issue before sending a pull request.
