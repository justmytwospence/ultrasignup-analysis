# Analysis repo — agent guide

This is the PyMC modeling layer. It reads the canonical DuckDB at `data/ultrasignup.duckdb` (written by the sibling `scraper/` repo) and runs Bayesian inference notebooks under `notebooks/`.

## Marimo: one live watched session per notebook

The `marimo-pair` skill assumes a long-lived marimo kernel that reflects file edits automatically. Hard rules:

1. **Always start with `--watch`** so the kernel re-reads the `.py` on save:
   ```bash
   uv run marimo edit --watch --no-token --port <PORT> --host 127.0.0.1 \
       --session-ttl 86400 notebooks/<name>.py
   ```
   Without `--watch`, edits to the file *won't* propagate to the running kernel — you'll get phantom errors where the kernel is on the old code while the file has the new code.

2. **One port per notebook.** `2718` is the marimo default; use `2720`, `2721`, etc. for additional notebooks. Run `bash ~/Projects/.agents/skills/marimo-pair/scripts/discover-servers.sh` first to see what's already up — there may be sessions from other projects (e.g., `fieldgoal` often runs `2719`) or earlier sessions still alive.

3. **Don't kill and restart** unless something is genuinely wedged. To switch which notebook you're editing, just open a new marimo on a different port. Kernels keep state; restarting throws away PyMC compile caches, JAX caches, and any in-memory traces.

4. **Don't `open <url>` after starting marimo** — `marimo edit` auto-opens a browser tab. Calling `open` again spawns duplicate tabs. If the user already has a tab pointing at the URL, it'll reconnect automatically when the kernel comes up.

5. **Edit through `code_mode`, not the file.** While a session is running, the kernel owns the `.py`. Use `ctx.edit_cell(cell_id, new_code)` via the marimo-pair `execute-code.sh` script. Direct file edits will desync the kernel from the file (especially without `--watch`).

6. **Be aware of cross-project / cross-session marimos.** Sessions persist across Claude Code sessions and across projects. Always discover first.

## Sampling on macOS

- `nuts_sampler='nutpie'` is the default for dev iteration. Pure Rust, no JAX, ~20× faster than the PyMC Python NUTS for our models.
- `nuts_sampler='blackjax'` is broken on Apple Silicon: `UNIMPLEMENTED: default_memory_space is not supported.` from `jax-metal`. Either set `JAX_PLATFORMS=cpu` to force CPU JAX, or just use nutpie.
- For final inference (Linux/GPU box, e.g. RunPod), `numpyro` is fine.

## K-core dev pattern

The `0_kcore.py` notebook explores the (α, β) Pareto frontier and selects a dev k-core. Modeling notebooks should default to that dev k-core for fast iteration, with looser configs reserved for final inference. Note that filter pipelines differ: `1_finish_times` and `0_kcore` use `filter_races_with_dnfs`; `2_observed_dnfs` and `4_unobserved_dnfs` don't, so the same (α, β) gives different course counts. Inspect the resulting `n_courses` after subsetting before judging whether the model is identifiable.

## Cache paths

All cache paths in notebook cells must be anchored to `Path(__file__).parent` rather than relative to CWD. Marimo's CWD is wherever you ran `uv run marimo` from (typically `analysis/`), not the notebook directory like Jupyter.
