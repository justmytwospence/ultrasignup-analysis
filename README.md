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

### Editions and years
- Which editions were unusually hard or easy (heat, snow, mud) relative to their course's norm? (planned: edition effect)
- Is there a trend in pace over the years, as fields broaden? (planned: year term)
- Which editions report their DNFs, and what predicts it? (2.1)

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
- Does DNF risk rise with the course's climbing, or on hot or muddy years? (planned: terrain and edition effects)
- Do runners who DNF once drop again more often, or bounce back? (planned: runner history)
- How many DNFs are hidden in the editions that post only finishers, and does leaving them out bias what we learn about course difficulty? (2.3)
- What share of a course's drops come from being a hard course, and what share from who showed up? (course vs runner DNF effects, 2.2)

### Population
- How does pace scale with distance, and does that differ by gender? (1.1)
- How does pace change with age, and where is the peak? (1.1)

### Prediction
- Who will win a race? For every registered runner, what are the odds of winning, of a podium or top 10, and of finishing at all? (simulate the whole field from the joint finish and DNF posterior; planned: needs entrant rosters and an edition effect for the day's conditions)
- What finish time, with an interval, should a given runner expect at a given course? (posterior predictive)
- What is the chance they finish at all, and at what pace is that chance best? (2.2, 2.3)

## Setup

```sh
uv sync
cp .env.example .env    # set ULTRASIGNUP_QUACK_URI and ULTRASIGNUP_QUACK_TOKEN
uv run pytest
```

## Data access

```python
from utils import data_source
from utils.data_processing import load_results, process_results

con = data_source.connect()                    # live
results = process_results(load_results(con))

data_source.pull()                             # copy the tables to data/results.duckdb
con = data_source.connect(local=True)          # offline, or to pin a long fit to one snapshot
```

## Notebooks

[marimo](https://marimo.io) notebooks, run in order. Each adds one thing to the one before it, and each follows the same workflow: data, model with a prior predictive check, fit and convergence, posterior predictive checks, model comparison (PSIS-LOO), prior sensitivity, results.

| Notebook | What it does |
|---|---|
| `0.1_kcore.py` | Picks a dense (α, β)-core of runners and courses for models with runner effects |
| `1.1_finish_courses.py` | Pace by distance, gender and age, and course difficulty, on every finisher |
| `1.2_finish_runners.py` | Adds runner effects on the core, separating course difficulty from field strength |
| `2.1_dnf_reporting.py` | Which race editions report their DNFs |
| `2.2_dnf_observed.py` | Adds DNFs to 1.2: runner and course DNF tendencies, correlated with pace |
| `2.3_dnf_unobserved.py` | Adds editions that post no DNFs, via 2.1's reporting model, and predicts their hidden DNFs |

Names are `<track>.<step>_<topic>`: track 0 prepares data, 1 models finish times, 2 models
DNFs; a step builds on the earlier steps of its track.

```sh
uv run marimo edit notebooks/0.1_kcore.py
```

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
- **Hidden results**: about a third of editions show results only on runner pages and are
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

- **Hierarchy identifiability**: the population/course/race hierarchy risks sigma_race
  collapsing to zero. Fit a two-level version and compare with LOO.
- **Censoring penalty approximation**: `-n_finishers * log E[P(finish)]` uses an expectation.
  The zero-DNF mixture term uses a distance-only p while the Bernoulli uses the
  difficulty-adjusted p, and the selection-correction Potential omits the pace-dependent
  numerator. Replace with the exact marginal or a shared latent.
- **Informative quadratic prior**: `beta_dist_dnf_quad ~ N(0.5, 0.3)` looks tuned to the data.
  Justify it or widen it toward N(0, 0.5) and check sensitivity.

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
