# UltraSignup analysis

Bayesian models of ultramarathon finish times and DNF rates over every UltraSignup result.

## Questions the models should answer

Each question names the model term that answers it. Ones marked (planned) need a term the
notebooks do not have yet (see [docs/open-questions.md](docs/open-questions.md)).

**Courses**
- Which courses are hardest at a given distance? (course effect, 1.1)
- How much of a course's difficulty is field strength rather than the course? (course effects with and without runner effects, 1.1 vs 1.2)
- Which course is hardest for its terrain, independent of its length? (planned: course effect net of gain per km)
- Which courses drop the most runners at a given pace? (course DNF effect, 2.2)
- Are harder courses also the ones with more DNFs, once runner quality is accounted for? (course pace-DNF correlation, 2.2)

**Editions and years**
- Which editions were unusually hard or easy (heat, snow, mud) relative to their course's norm? (planned: edition effect)
- Is there a trend in pace over the years, as fields broaden? (planned: year term)
- Which editions report their DNFs, and what predicts it? (2.1)
- How many DNFs are hidden in editions that post only finishers? (2.3)

**Runners**
- Who is fastest, net of the courses they happened to run? (runner effect, 1.2)
- Who is disproportionately good on steep courses or long distances? (planned: runner-by-gain and runner-by-distance slopes)
- Who is consistent, and who is erratic? (planned: per-runner variance)
- Who is likely to drop, and are the fastest runners the least likely to? (runner DNF effect and its correlation with pace, 2.2)
- How does a first-time ultrarunner compare with a veteran? (planned: experience term)

**Population**
- How does pace scale with distance, and does that differ by gender? (1.1)
- How does pace change with age, and where is the peak? (1.1)
- How does DNF probability scale with distance, age and gender? (2.2)

**Prediction**
- What finish time, with an interval, should a given runner expect at a given course? (posterior predictive)
- What is the chance they finish at all? (2.2, 2.3)

## Setup

```sh
uv sync
cp .env.example .env    # set ULTRASIGNUP_QUACK_URI and ULTRASIGNUP_QUACK_TOKEN
uv run pytest
```

## Data access

```python
from utils import mirror
from utils.data_processing import load_results, process_results

con = mirror.connect()                    # live
results = process_results(load_results(con))

mirror.pull()                             # copy the tables to data/mirror.duckdb
con = mirror.connect(local=True)          # offline, or to pin a long fit to one snapshot
```

## Notebooks

marimo notebooks, run in order:

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

Known problems and planned work: [docs/open-questions.md](docs/open-questions.md).

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md): free for personal use, research, education and noncommercial organizations. Commercial use, including building the methodology into a product or service, requires a separate license; contact Spencer Boucher.

Outside contributions require agreeing to a contributor license agreement; open an issue before sending a pull request.
