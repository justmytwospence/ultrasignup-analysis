# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Architecture Overview

This is a **monorepo of two independent git repos** plus shared workspace tooling:

- `scraper/` — Python web scraper for UltraSignup.com. Writes to a DuckDB file. Has its own `.git`, `.venv`, `pyproject.toml`. CLI commands: `scrape-participants`, `scrape-races-from-participants`, `scrape-results`, `scrape-races-incremental`, `extract-elevations`, `enrich-races`, `estimate`. See `scraper/README.md` for the full API/schema reference.
- `analysis/` — PyMC Bayesian modeling layer. Reads the DuckDB written by `scraper/`. Has its own `.git`, `.venv`, `pyproject.toml`. Notebooks under `analysis/notebooks/` model finish times, DNFs, reporting bias.
- `data/cache/` — root-level shared caches (mostly stale; the canonical analytical DB is `analysis/data/ultrasignup.duckdb`).
- The project root itself is a thin git repo created by `bd init` to hold `.beads/`, `.claude/`, `CLAUDE.md`, `AGENTS.md`, and the DuckDB MCP config. Don't `git add scraper/` or `git add analysis/` from root — they're nested repos and would become broken submodule pointers. Commit inside each subrepo.

### Data flow

```
UltraSignup API → scraper (DuckDB writer) → analysis/data/ultrasignup.duckdb → analysis notebooks (PyMC)
```

The DuckDB file is the contract between the two halves. The schema is documented in `scraper/README.md` (mermaid ERD around line 600).

## Build & Test

```bash
# Scraper (from scraper/)
cd scraper && uv sync                                  # install deps
cd scraper && uv run python main.py estimate           # smoke test the API
cd scraper && uv run pytest                            # run tests (once tests/ is populated)

# Analysis (from analysis/)
cd analysis && uv sync
cd analysis && uv run marimo edit notebooks/           # open marimo notebooks
cd analysis && uv run jupyter lab                      # legacy ipynb workflow
```

Two separate venvs: `scraper/.venv` and `analysis/.venv`. Don't try to merge them — they have different scientific stacks (analysis pulls in PyMC + JAX; scraper is HTTP-focused).

## Harness tooling

- **Beads** at project root for cross-session task tracking. `bd ready` to find available work, `bd prime` for full workflow.
- **DuckDB MCP** configured at project root (`.mcp.json`) in read-only mode. Available from anywhere in the workspace; query the DuckDB without writing Python.
- **Marimo skills** scoped to `analysis/.claude/skills/` — `marimo-pair` and `marimo-notebook` for reactive editing/authoring; `jupyter-to-marimo` for converting `.ipynb` files to `.py` marimo notebooks.
- **claude-api skill** (built-in) for the elevation extractor rewrite (Phase 2 of the active plan).

## Conventions

- Conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`).
- Atomic commits — one logical change per commit.
- Never force-push to main.
- No emojis in code or docs.
- Plans live in `.claude/plans/<topic>.md`. Active plan: see most recent file there.
- Use `bd` for tasks, not TodoWrite or markdown TODO lists.
