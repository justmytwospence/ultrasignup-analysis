# UltraSignup analysis: agent guide

Bayesian models (PyMC, marimo notebooks) of ultramarathon finish times and DNF rates over
UltraSignup results. This repo only analyzes. The data comes from
[ultrasignup-mirror](https://github.com/justmytwospence/ultrasignup-mirror), a sync engine on
the NUC (`~/Projects/ultrasignup-mirror` there, deployed as `~/homelab/ultrasignup/`) that keeps
a complete DuckDB mirror of UltraSignup and serves it read-only over DuckDB's Quack protocol.

## Data

- `utils/mirror.py`: `connect()` attaches the mirror live (queries run on the NUC);
  `connect(local=True)` opens `data/mirror.duckdb`, the copy `pull()` makes for offline work,
  pinned long fits, RunPod and the duckdb MCP. Settings: `ULTRASIGNUP_QUACK_URI` and
  `ULTRASIGNUP_QUACK_TOKEN` in a gitignored `.env` (see `.env.example`).
  `ULTRASIGNUP_SOURCE=local` makes plain `connect()` (and so every notebook) read the pull;
  use it for refits, since the k-core and trace caches are keyed by parameters, not data.
- Read only the mirror's analyst schema `main` (`races`, `results`, `runners`, `courses`,
  `entrants`, `relay_results`, `about`), documented with ERDs in the mirror's `docs/schema.md`.
  Never its `sync` schema. `check_schema` refuses a mirror whose schema major differs from
  `EXPECTED_SCHEMA_MAJOR`; on a bump, update `utils/` and the tests, then the constant.
- `utils/data_processing.py`: `load_results()` reads `main.results`; `process_results()` builds
  the model frame every notebook uses (column contract in its docstring). Starts are finish or
  DNF; `dns` and `dq` are dropped. Hidden (`participant_history`) results are opt-in.
- DuckDB is pinned to 2.0 (the mirror's version, required by Quack).
- `data/` is gitignored: `mirror.duckdb`, `cache/` (k-core subsets and traces) and `legacy/`
  (the 2025 scrape and old checkouts, kept until the refits on mirror data are done).

## Setup and tests

```sh
uv sync
cp .env.example .env    # then fill in the token
uv run pytest           # synthetic mirror + a real Quack round trip; offline-safe
uv run pytest -m live   # contract check against the real mirror
```

## Notebooks

`notebooks/0_kcore.py` to `4_unobserved_dnfs.py`, run in that order (each later one reads the
k-core chosen in 0). `notebooks/legacy_ipynb/` holds the pre-marimo originals, read only.

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
- Dev k-core: develop at a tight (α, β) such as (3, 840), check `n_courses` after subsetting
  (nb 0/1 filter to courses with DNFs, nb 3/4 do not), and loosen for final fits.

Open modeling problems and planned work: `docs/open-questions.md`. Past plans: `docs/history/`.

## Conventions

- Conventional commits, one logical change each; push to `main`.
- No emojis in code or docs.
