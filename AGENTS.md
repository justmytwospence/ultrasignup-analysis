# UltraSignup analysis: agent guide

Bayesian models (PyMC, marimo notebooks) of ultramarathon finish times and DNF rates over
UltraSignup results. This repo only analyzes: the data lives in a DuckDB database of every
UltraSignup result, served read-only over DuckDB's Quack protocol (on the NUC).

## Data

- `utils/data_source.py`: `connect()` attaches the database live (queries run on the NUC);
  `connect(local=True)` opens `data/results.duckdb`, the copy `pull()` makes for offline work,
  pinned long fits, RunPod and the duckdb MCP. Settings: `ULTRASIGNUP_QUACK_URI` and
  `ULTRASIGNUP_QUACK_TOKEN` in a gitignored `.env` (see `.env.example`).
  `ULTRASIGNUP_SOURCE=local` makes plain `connect()` (and so every notebook) read the pull;
  use it for refits, since the k-core and trace caches are keyed by parameters, not data.
- Read only the data source's analyst schema `main` (`races`, `results`, `runners`, `courses`,
  `events`, `relay_results`, `about`).
  Never its `sync` schema. `check_schema` refuses a database whose schema major differs from
  `EXPECTED_SCHEMA_MAJOR`; on a bump, update `utils/` and the tests, then the constant.
- `utils/data_processing.py`: `load_results()` reads `main.results`; `process_results()` builds
  the model frame every notebook uses (column contract in its docstring). Starts are finish or
  DNF; `dns` and `dq` are dropped, as are races with a NULL `course_id` or `distance_km`, and
  implausible ages and paces (the source never cleans posted values; bounds are constants
  there). `source` is always `results_api`: hidden results (about a third of editions) are not
  included, and nothing from the 2025 scrape is imported. Relays live in `relay_results`.
- Quack limits: bound parameters do not pass through (inline literals), and a query cannot
  stream two remote tables at once (no join of two remote tables: pull, or make a local
  TEMP TABLE from one first). Needs WireGuard off the LAN.
- DuckDB is pinned to 2.0 (the data source's version, required by Quack).
- `data/` is gitignored: `results.duckdb`, and `cache/` (k-core subsets and traces).

## Setup and tests

```sh
uv sync
cp .env.example .env    # then fill in the token
uv run pytest           # synthetic database + a real Quack round trip; offline-safe
uv run pytest -m live   # contract check against the real data source
```

## Notebooks

Notebooks are named `<track>.<step>_<topic>.py`: track 0 data preparation (`0.1_kcore`),
track 1 finish times (`1.1_finish_courses`, `1.2_finish_runners`), track 2 DNFs
(`2.1_dnf_reporting`, `2.2_dnf_observed`, `2.3_dnf_unobserved`).
Each step adds one thing to the earlier steps of its track; a new model gets the next step
number (`1.3_...`), and steps past 9 are fine (`1.10`). Each notebook keeps its traces and
writes its output for later ones in `data/cache/<notebook stem>/`: `0.1_kcore/core.parquet`
(the dev core), `1.1_finish_courses/course_effects.parquet`,
`1.2_finish_runners/{course,runner}_effects.parquet`,
`2.1_dnf_reporting/edition_reporting.parquet`, `2.2_dnf_observed/{course,runner}_effects.parquet`. Only models with runner effects need the core.

Workflow conventions (1.1 and 2.1 are the templates): sections Setup, Data, Model (fixed
priors, simulator or prior predictive check), Fit (summary, rank plots), Posterior predictive
checks, Model comparison (PSIS-LOO), Prior sensitivity (power-scaling), Results, Handoff.
Priors are fixed in advance, never computed from the data. Distance enters as
`log(distance_km / 50)`; the data are marathon (42 km) to 100 miles. Trace caches are keyed by
a model version and a data hash.

- Load the project's `marimo-pair` skill and the global `marimo-style` skill before touching a
  notebook. Every change goes through marimo-pair against a live session, never by editing the
  `.py` file.
- One persistent session per notebook, started from the repo root and left running:
  `uv run marimo edit --watch --no-token --session-ttl 86400 --no-skew-protection notebooks/<nb>.py --port <port>`
  (not headless: the user watches it; 2718 and up, check what is taken).
- Sampling runs inside the kernel (never in a separate script), and stale downstream cells are
  run afterwards so results appear without a manual "run all". Wait for long fits with a
  background job, never a sleep loop.
- Notebooks orchestrate; reusable logic lives in `utils/`. Cache paths are anchored at
  `Path(__file__)`, with caches under `data/cache/`.
- Sampler on macOS: `nuts_sampler='nutpie'` (it renders marimo's live progress widget). Never
  blackjax on Apple Silicon (jax-metal), and do not switch to numpyro to dodge a nutpie error:
  fix the model (for example, `LKJCholeskyCov` trips numba's `cumsum(axis=)`; build it from
  `HalfNormal` scales and `LKJCorr` instead). Give long cells a `mo.callout` beacon with the
  start time and expected duration. For final inference too slow locally, the `Dockerfile`
  builds a RunPod JAX-CUDA image (numpyro is fine there).
- Large fits (runner effects): sample through `utils.sampling.sample_lean` with `store_dir`, which
  streams draws to a zarr store on disk (nutpie otherwise holds every draw, warm-up included, in
  memory at ~2.75x their size), leaves per-runner deterministics out (derive them after sampling),
  refuses to start when `check_memory` says it will not fit, and rejects interrupted traces. The old way needed ~72 GB for 2.2 on the (3, 500)
  core and macOS killed the kernel. Before a heavy fit, restart idle kernels and marimo servers:
  a server with big plotly outputs (0.1) can hold 30+ GB of compressed memory by itself.
- Unattended runs: call marimo-pair with `--url` and `--file` spelled out in the tool call
  (pi-marimo then opens a headless session if no tab is open, also after a kernel restart), and
  wait with `bg_wait marimo:<file>`, which ends within seconds if the kernel disconnects.
- Dev k-core: (3, 840) on every start at 42 km or longer (28 courses, about 8,500 runners),
  chosen in 0.1; check `n_courses` after subsetting, and loosen for final fits. The full (α, β) sweep takes about a minute; its cache key includes a data
  fingerprint.

Open modeling problems and planned work: the README's Open questions section.

## Conventions

- Conventional commits, one logical change each; push to `main`.
- No emojis in code or docs.
