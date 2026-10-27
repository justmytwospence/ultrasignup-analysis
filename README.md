# UltraSignup analysis

Bayesian models of ultramarathon finish times and DNF rates, built on a complete mirror of
UltraSignup results.

The data comes from [ultrasignup-mirror](https://github.com/justmytwospence/ultrasignup-mirror),
a self-hosted sync engine that keeps every UltraSignup race, result, entrant list and runner
history in DuckDB and serves its analyst schema over DuckDB's Quack protocol. This repo only
reads it.

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

con = mirror.connect()                    # live: queries run on the mirror
results = process_results(load_results(con))

mirror.pull()                             # copy the analyst tables to data/mirror.duckdb
con = mirror.connect(local=True)          # offline, or to pin a long fit to one snapshot
```

The tables (`races`, `results`, `runners`, `courses`, `events`, `relay_results`, `about`) are
documented with ERDs in the mirror's `docs/schema.md`.

## Notebooks

marimo notebooks, run in order:

| Notebook | What it does |
|---|---|
| `0_kcore.py` | Picks a dense (α, β)-core of runners and courses to model |
| `1_finish_times.py` | Course difficulty from finish times |
| `2_dnf_reporting.py` | Which races report their DNFs |
| `3_observed_dnfs.py` | Joint finish-time and DNF model |
| `4_unobserved_dnfs.py` | Adds DNFs that races never reported |

```sh
uv run marimo edit notebooks/0_kcore.py
```

Known problems and planned work: [docs/open-questions.md](docs/open-questions.md).

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md): free for personal use, research, education and noncommercial organizations. Commercial use, including building the methodology into a product or service, requires a separate license; contact Spencer Boucher.

Outside contributions require agreeing to a contributor license agreement; open an issue before sending a pull request.
