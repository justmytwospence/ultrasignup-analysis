"""Access to the UltraSignup mirror (https://github.com/justmytwospence/ultrasignup-mirror).

The mirror runs on the NUC and serves its analyst schema (`main`: races, results, runners,
courses, entrants, relay_results, about) read-only over DuckDB's Quack protocol. Two ways in:

    con = connect()             # live: queries execute on the NUC, results stream back
    con = connect(local=True)   # the local copy in data/mirror.duckdb, made by pull()

`pull()` copies the analyst tables into data/mirror.duckdb, for offline work, for long MCMC
runs that should not see the data move underneath them, for RunPod (which cannot reach the
NUC), and for the duckdb MCP server in .mcp.json.

Configuration comes from the environment (or a gitignored .env at the repo root):
    ULTRASIGNUP_QUACK_URI    e.g. quack:ultrasignup.example.com:443
    ULTRASIGNUP_QUACK_TOKEN  the mirror's read-only Quack token
    ULTRASIGNUP_SOURCE       live (default) or local: what connect() opens when `local` is
                             not given. Set local to pin notebooks and their caches (keyed
                             by parameters, not data) to one pull.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PATH = REPO_ROOT / "data" / "mirror.duckdb"

# Major version of the mirror's `main` schema this code understands. The mirror bumps the
# major on breaking changes to `main` and records it in main.about.schema_version.
EXPECTED_SCHEMA_MAJOR = 1

TABLES = ("about", "courses", "races", "results", "relay_results", "runners", "entrants")

# Storage format of the local copy: readable by DuckDB 1.5+ (the duckdb MCP, older images).
LOCAL_STORAGE_VERSION = "v1.5.0"


class MirrorConfigError(RuntimeError):
    pass


class SchemaMismatch(RuntimeError):
    pass


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


def _settings() -> tuple[str, str]:
    _load_env()
    uri = os.environ.get("ULTRASIGNUP_QUACK_URI", "").strip()
    token = os.environ.get("ULTRASIGNUP_QUACK_TOKEN", "").strip()
    if not uri or not token:
        raise MirrorConfigError(
            "Set ULTRASIGNUP_QUACK_URI and ULTRASIGNUP_QUACK_TOKEN (see .env.example), "
            "or use connect(local=True) after a pull()."
        )
    return uri, token


def _attach_remote(con: duckdb.DuckDBPyConnection, alias: str = "mirror") -> None:
    uri, token = _settings()
    con.execute(f"CREATE OR REPLACE TEMPORARY SECRET ultrasignup_mirror (TYPE quack, TOKEN {_quote(token)}, SCOPE {_quote(uri)})")
    con.execute(f"ATTACH {_quote(uri)} AS {alias} (TYPE quack)")


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def check_schema(con: duckdb.DuckDBPyConnection, catalog: str | None = None) -> str:
    """Return the mirror's schema version; raise SchemaMismatch if its major is not the expected one."""
    table = f"{catalog}.main.about" if catalog else "main.about"
    try:
        version = con.execute(f"SELECT schema_version FROM {table} LIMIT 1").fetchone()
    except duckdb.Error as err:
        raise SchemaMismatch(f"Cannot read schema_version from {table}: {err}") from err
    if version is None or version[0] is None:
        raise SchemaMismatch(f"{table} has no schema_version")
    text = str(version[0])
    major = int(text.split(".")[0])
    if major != EXPECTED_SCHEMA_MAJOR:
        raise SchemaMismatch(f"Mirror schema {text} is not compatible with this code (expects major {EXPECTED_SCHEMA_MAJOR}). Update utils/ to the new schema.")
    return text


def _default_local() -> bool:
    _load_env()
    source = os.environ.get("ULTRASIGNUP_SOURCE", "live").strip().lower() or "live"
    if source not in ("live", "local"):
        raise MirrorConfigError(f"ULTRASIGNUP_SOURCE must be live or local, not {source!r}")
    return source == "local"


def connect(*, local: bool | None = None, path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Open the mirror's analyst schema.

    local=None (default) follows ULTRASIGNUP_SOURCE (live unless set to local).
    live: attach the mirror over Quack and CONNECT to it, so plain SQL such as
    `SELECT * FROM results` runs on the NUC.
    local=True: open the pulled copy (data/mirror.duckdb, or `path`) read-only.
    """
    if local is None:
        local = _default_local()
    if local:
        target = Path(path) if path else LOCAL_PATH
        if not target.exists():
            raise FileNotFoundError(f"{target} does not exist; run utils.mirror.pull() first.")
        con = duckdb.connect(str(target), read_only=True)
        check_schema(con)
        return con
    con = duckdb.connect()
    _attach_remote(con)
    check_schema(con, "mirror")
    con.execute("CONNECT mirror")
    return con


def pull(tables: tuple[str, ...] = TABLES, *, dest: Path | str | None = None, source: duckdb.DuckDBPyConnection | None = None, source_catalog: str = "mirror") -> Path:
    """Copy the analyst tables into a local DuckDB file and return its path.

    The copy is written to a temporary file and moved into place, so a reader never sees a
    half-written file. `source` is for tests: a connection that already has the mirror
    attached as `source_catalog`.
    """
    target = Path(dest) if dest else LOCAL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    for stale in (tmp, tmp.with_name(tmp.name + ".wal")):
        stale.unlink(missing_ok=True)

    con = source or duckdb.connect()
    try:
        if source is None:
            _attach_remote(con, source_catalog)
        check_schema(con, source_catalog)
        con.execute(f"ATTACH {_quote(str(tmp))} AS pulled (STORAGE_VERSION '{LOCAL_STORAGE_VERSION}')")
        try:
            for table in tables:
                con.execute(f"CREATE TABLE pulled.main.{table} AS FROM {source_catalog}.main.{table}")
        finally:
            con.execute("DETACH pulled")
    finally:
        if source is None:
            con.close()
    os.replace(tmp, target)
    return target
