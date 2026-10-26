# Analysis repo — agent guide

PyMC modeling layer. Reads the canonical DuckDB at `data/ultrasignup.duckdb` (written by sibling `scraper/` repo) and runs Bayesian inference notebooks under `notebooks/`.

## Marimo-pair workflow (mandatory)

When working on a notebook, marimo-pair is the **only** acceptable execution path. The skill lives at `.claude/skills/marimo-pair/`.

### Hard rules (no exceptions)

1. **One persistent marimo session per notebook, always.** At the start of any notebook task, run `bash .claude/skills/marimo-pair/scripts/discover-servers.sh`. If a session for the target notebook is already running, **reuse it**. If not, start one (see below) and leave it running for the rest of the conversation. Never restart a healthy session "to be sure" — restarts wipe kernel state and force the user to re-trigger everything.

2. **Never sample / fit outside the marimo kernel.** No standalone Python scripts that build the model and call `pm.sample()` separately. They orphan the trace from the kernel — the trace lives on disk but no in-kernel cell ever consumed it, so every downstream visualization shows `None` until the user manually re-triggers the trace cell. Long-running work (sampling, posterior predictive, MAP optimization) goes through `ctx.run_cell` on the appropriate cell so the kernel owns the result and downstream cells cascade automatically.

3. **Always cascade downstream cells after any compute-heavy update.** When you finish a fit (or any structural change), immediately queue every downstream stale cell with `ctx.run_cell` so outputs land in the user's browser without them clicking anything. The user should never have to click "Run all" because Claude triggered a re-fit.

4. **Do NOT use `--headless`.** Headless mode prevents browser sessions from being created, which means `marimo._code_mode` calls fail with "No active sessions on the server." Always launch with the default browser-opening behavior so the user can watch live and the kernel session is created immediately.

5. **OOM recovery: restart marimo, then re-trigger via the kernel.** If the kernel dies (OOM, browser tab close, manual kill), the recovery sequence is: restart marimo → wait for it to come up → use `ctx.run_cell` to re-fire the relevant cells. The cached trace on disk loads automatically. Do **not** fall back to a standalone script because "marimo is fragile" — fix the underlying memory problem (smaller k-core, fewer chains/draws, drop runner effects from the dev variant) and refit through the kernel.

6. **Long-running cells: monitor without polluting the kernel queue.** Probes via `execute-code.sh` queue behind a running cell. To monitor a fit in progress, watch process state (`ps`, log files written by the cell, file system for cached trace) instead of repeatedly calling `execute-code.sh` — every probe just adds latency to the user's next interaction once the running cell finishes.

7. **Wait for compute via `run_in_background`, never `Bash(sleep N; ...)` loops.** Long fits (sampling, posterior predictive on large data) take 30 min – several hours. The wrong pattern is `Bash("sleep 540; check_status")` repeated — it blocks the conversation, hits Bash's 10-min timeout ceiling, and burns turns on no-op polls. The right pattern is a single watcher script launched with `run_in_background=true`:

   ```bash
   # /tmp/wait_for_<thing>.sh
   #!/bin/bash
   while kill -0 <SAMPLER_PID> 2>/dev/null; do sleep 30; done
   sleep 60  # let downstream cells cascade
   bash <execute-code.sh> --port <port> <<'PYEOF'
   # diagnostics that run in the kernel
   PYEOF
   ```

   Launch with `Bash("/tmp/wait_for_thing.sh", run_in_background=true)` — returns immediately with a task id. Claude Code fires a `<task-notification>` system-reminder the instant the script exits. Zero idle polling, zero blocked conversation.

### Starting a marimo session

Always launch with `--watch --no-token --session-ttl 86400 --no-skew-protection` and **fully detach from Claude Code's process group** so it survives Bash-tool cleanup. The macOS-friendly way is `subprocess.Popen(..., start_new_session=True)`:

```bash
python3 -c "
import subprocess
subprocess.Popen(
    ['uv', 'run', 'marimo', 'edit', '--watch', 'notebooks/<name>.py',
     '--port', '<port>', '--no-token', '--session-ttl', '86400',
     '--no-skew-protection'],
    stdout=open('/tmp/marimo_<port>.log', 'w'),
    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    start_new_session=True,
    cwd='/Users/spencerboucher/Projects/UltraSignup/analysis',
)
"
```

After launch, verify with `ps -o pid,ppid,pgid -p <pid>` — `ppid` should be `1` (init) and `pgid` should equal the marimo pid. Anything else means it's still in Claude Code's bash group and will be killed on the next Bash-tool tear-down.

**Why `nohup … &` alone fails:** on macOS inside Claude Code's Bash tool, the backgrounded subprocess inherits the bash subprocess's process group. When the parent shell exits between commands, the group gets `SIGTERM`, which marimo treats as a graceful shutdown ("ERROR: Cancel N running task(s), timeout graceful shutdown exceeded" in the log). `setsid` would fix it on Linux but isn't installed on macOS by default. `start_new_session=True` is the portable fix.

Flag rationale:
- `--watch` — file edits hot-reload in the running kernel.
- `--no-token` — puts the server in the local registry so `marimo-pair` discovers it automatically. With a token, set `MARIMO_TOKEN` before calling the scripts; passing `--token` in process args leaks the token.
- `--session-ttl 86400` — keeps the kernel alive for 24h after websocket disconnect. **Required.** Without it, marimo gracefully shuts down a few seconds after the browser tab closes (or the laptop sleeps briefly), wiping kernel state.
- `--no-skew-protection` — disables the version-mismatch token check on the websocket. **Required.** When marimo restarts, the browser tab still holds the *previous* server's skew token. Skew protection rejects the connection ("invalid server token"), the browser silently fails to reconnect, marimo sees zero clients, and graceful-shutdown fires. Disable so a long-lived browser tab survives any marimo restart without a hard refresh. (Skew protection is for production multi-version deploys, not local notebooks.)
- Pick a port that isn't already taken (`lsof -i :2718`). Default 2718; use 2719+ for additional notebooks. Other projects (e.g. `fieldgoal`) often run on 2719.
- **Never `--headless`** (Hard Rule 4).

### Driving the running notebook

```bash
# Discover all running marimo servers (in any project / session)
bash .claude/skills/marimo-pair/scripts/discover-servers.sh

# Execute Python in a kernel — heredoc preferred for multiline
bash .claude/skills/marimo-pair/scripts/execute-code.sh --port <port> <<'PYEOF'
import marimo._code_mode as cm
async with cm.get_context() as ctx:
    for c in ctx.cells.values():
        if c.status == 'stale':
            ctx.run_cell(c.id)
PYEOF
```

`execute-code.sh` POSTs to the running kernel — every variable defined by the notebook is in scope. Use `marimo._code_mode` to mutate the cell graph (create / edit / delete cells, install packages, run cells); use plain inspection for debugging.

### Editing rules with --watch

- **`--watch` lets you edit the `.py` file directly** and the kernel reloads — this is the one case where direct file edits are safe. Use it for big multi-cell rewrites where authoring through `ctx.edit_cell` would be tedious.
- **For single-cell edits, prefer `ctx.edit_cell(cell_id, new_code)`** via marimo-pair so the kernel applies the change immediately and you don't race the file watcher.
- **NEVER write to the `.py` file when running marimo `edit` *without* `--watch`** — the kernel owns the file in that mode and will silently desync from your edits.
- After cell-graph changes (create / edit / delete), call `ctx.run_cell(id)` on the affected cell or its consumers. Marimo will not re-execute by itself unless `--watch` triggers a full reload.

### Cell-private vs cross-cell variables

Marimo treats single-underscore-prefixed names (`_var`) as cell-private: they don't leak across cells. Use this to avoid `MultipleDefinitionError` when loop variables (`for _i, _x in ...`) would otherwise leak. Names without the underscore prefix become cross-cell exports — only use those when other cells need to read them. The `_fig` vs `fig` collision in `3_observed_dnfs.py:plot_ppc` we hit during conversion was exactly this issue: the function parameter was `fig` but the body used `_fig`, which marimo namespaced as cell-private and left undefined.

### Pre-flight checks before structural edits

```bash
bash .claude/skills/marimo-pair/scripts/execute-code.sh --port <port> <<'PYEOF'
import marimo._code_mode as cm
async with cm.get_context() as ctx:
    err = [c for c in ctx.cells.values() if c.status in ('marimo-error', 'exception')]
    print(f"Errors: {len(err)}")
    for c in err:
        print(f"  {c.id} ({c.status}): {c.code[:80]!r}")
PYEOF
```

Surfaces existing errors so you don't pile new edits on a broken graph.

## Sampling on macOS

### Hard rule: the user must be able to follow the sampler live

Live progress in the browser tab is non-negotiable. The user should never have to refresh, tail logs, run `ps`, or guess which cell is running. To guarantee that:

1. **Default to `nuts_sampler='nutpie'`.** Nutpie's progress messages route through marimo's display callback and render as the **native marimo progress widget** in the cell output — visible in real time without refresh.
2. **Never silently fall back to `nuts_sampler='numpyro'` on macOS.** Numpyro's progress is raw JAX tqdm to stderr. Marimo renders it as plain text, often with carriage-return artifacts that don't update visibly. If you switch to numpyro, the user will say "I can't see what's running."
3. **Never use `nuts_sampler='blackjax'` on Apple Silicon.** It dies with `UNIMPLEMENTED: default_memory_space is not supported` from jax-metal. `JAX_PLATFORMS=cpu` works but nutpie is still faster.
4. **PyMC's default NUTS shows the marimo widget too**, but is ~10× slower than nutpie. Use only when nutpie is genuinely impossible.
5. **Every long-running cell must emit a beacon as its first output** so the user can locate it from any view (edit *or* app). Use a `mo.callout` with the operation name, start timestamp, and expected duration:
   ```python
   import datetime
   mo.output.replace(mo.callout(
       f"⏳ Sampling Model 3 (observed DNFs) — started {datetime.datetime.now():%H:%M:%S}, expected ~70 min",
       kind="info",
   ))
   # ... then pm.sample(...)
   ```
   This way the cell renders a visible block immediately, before sampling output starts streaming, and stays visible the whole time. Without it, an empty-output running cell is invisible in app view (where running-cell indicators are hidden by design — app view assumes end users aren't running anything).
6. **Tell the user to use edit view when actively watching MCMC.** App view hides running-cell indicators by design. Edit view shows the sticky status bar at the top with the running cell ID and a pulsing border around the running cell.

### When nutpie blows up — fix nutpie, don't switch samplers

The temptation is to flip `nuts_sampler='numpyro'` and move on. **Don't.** Diagnose and patch the model so nutpie works. Known issues:

| Symptom | Cause | Fix |
|---|---|---|
| `TypingError: cumsum(array(float64, 1d, C), axis=Literal[int](0))` | Numba (used by nutpie's compile path) doesn't accept the `axis=` kwarg on 1D `cumsum` calls. PyMC's `LKJCholeskyCov` produces such a call internally. | Replace `LKJCholeskyCov` with manual parametrization: separate `pm.HalfNormal` for each std + `pm.LKJCorr(n=k, eta=eta, return_matrix=True)` for the correlation matrix, then assemble Σ and take `pm.math.cholesky(Σ)`. Keeps the same prior, dodges the cumsum. |
| `UNIMPLEMENTED: default_memory_space is not supported` | jax-metal incompatibility. | Don't use blackjax/numpyro on Apple Silicon. |
| `divergences > 0`, very slow tuning | Bad geometry, not a sampler bug. | Reparametrize: non-centered hierarchies, looser priors on hyperparams, scale the data. |

For final inference on RunPod/GPU, `numpyro` is fine because no humans are watching the progress bar.

- The existing `Dockerfile` + `entrypoint.sh` set up a RunPod-compatible image with JAX-CUDA and JupyterLab — reach for that when local CPU sampling is intractable.

## K-core dev pattern

The `0_kcore.py` notebook explores the (α, β) Pareto frontier and selects a dev k-core. Modeling notebooks should default to a tight dev k-core for fast iteration, with looser configs reserved for final inference. Empirical scaling on this dataset:

| (α, β) | nb 1 obs | nb 4 courses | nb 4 runners | dev fit time |
|--------|----------|--------------|--------------|--------------|
| (3, 840) | 62K | 11 | 23K | <1 min, **R-hat ~4.5 (unidentifiable)** |
| (3, 629) | TBD | 61 | 50K | ~5–10 min target |
| (3, 423) | 499K | 266 | 222K | hours; intractable on macOS |
| (3, 233) | 591K | ~510 | ~190K | nb 2 ran in <30 min (production sized) |

Filter pipelines differ: `1_finish_times` and `0_kcore` use `filter_races_with_dnfs`; `3_observed_dnfs` and `4_unobserved_dnfs` don't, so the same (α, β) gives different course counts. Inspect the resulting `n_courses` after subsetting before judging whether the model is identifiable.

## Cache paths

All cache paths in notebook cells must be anchored to `Path(__file__).parent` rather than relative to CWD. Marimo's CWD is wherever you ran `uv run marimo` from (typically `analysis/`), not the notebook directory like Jupyter.

## Notebook authoring rules

- **Notebooks orchestrate; `utils/` does the work.** Notebooks read like a narrative — load data, transform, model, diagnose. Reusable logic (priors, k-core, MCMC notifications, plotting) belongs in `utils/`. Notebooks call into `utils/`, they don't reinvent it.
- **Each cell = one logical step.** A cell does one thing: load data, build model, fit, plot. Short cells are easier to reason about, re-run, and edit.
- **Reactive graph — define before use.** Every variable must be defined before its first use. Circular dependencies cause marimo to refuse execution. Split cells so the dependency order is explicit.
- **Cells should be re-runnable.** Avoid cells that only work once (e.g. appending to a list on every execution). When idempotency isn't possible (sampling, file IO), the cell should at least be safe to re-run without corrupting state.
- **Markdown cells carry the narrative.** Every numbered section opens with a `mo.md(...)` cell explaining *what* and *why*. Good markdown makes the notebook readable top-to-bottom without executing it.
- **Imports only in the environment-setup cell.** Never re-import elsewhere. The `os`-import-missing bug we hit in three notebooks was because the original `.ipynb` relied on Jupyter's implicit globals; in marimo every cell sees only what its parent cells return.
- **Configuration constants in a dedicated cell**, `ALL_CAPS` (e.g. `TUNE = 500`, `DRAWS = 500`, `TARGET_ACCEPT = 0.95`, `ALPHA = 3`, `BETA = 629`, `RANDOM_SEED = 42`).
- **Print DataFrame shape** as the last statement of any cell that creates or transforms a DataFrame (rows × cols, key range, n unique).
- **Matplotlib defaults** at top of notebook: `plt.rcParams['figure.figsize'] = (14, 5)`. Consistent colors across charts.
- **Split expensive ops into their own cell.** Then run faster downstream operations on the result in separate cells — feedback loop on the cheap parts is fast.
- **When editing or deleting cells, verify the whole notebook still runs end-to-end.** Don't break dependencies of downstream cells. Use the pre-flight error check (above) before and after structural changes.
- **Comments explain *why*, not *what*** and never reference previous versions of the code or notebook.

## Skills

The marimo skills (`marimo-pair`, `jupyter-to-marimo`, `marimo-notebook`) are symlinked into `.claude/skills/`. To install fresh in a new project, prefer the targeted form:

```
npx skills add marimo-team/skills --agent claude-code --yes
npx skills add marimo-team/marimo-pair --agent claude-code --yes
```

Do **NOT** use `npx skills experimental_install` — it ignores `--agent` and creates skill directories for all 45 known agent namespaces (`.agents/`, `.cortex/`, `.kiro/`, `.windsurf/`, etc.), all gitignored. The targeted `add --agent claude-code` installs only to `.claude/skills/`.
