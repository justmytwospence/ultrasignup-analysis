import os

import duckdb
import pytest
from conftest import build_mirror

from utils import mirror
from utils.data_processing import RESULT_COLUMNS, process_results


def test_connect_local(mirror_file):
    con = mirror.connect(local=True, path=mirror_file)
    assert con.execute("SELECT count(*) FROM results").fetchone()[0] > 0
    con.close()


def test_connect_local_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        mirror.connect(local=True, path=tmp_path / "nope.duckdb")


def test_schema_major_mismatch(tmp_path):
    path = build_mirror(tmp_path / "v2.duckdb", schema_version="2.0")
    with pytest.raises(mirror.SchemaMismatch):
        mirror.connect(local=True, path=path)


def test_pull_round_trip(mirror_file, tmp_path):
    source = duckdb.connect()
    source.execute(f"ATTACH '{mirror_file}' AS remote (READ_ONLY)")
    dest = mirror.pull(dest=tmp_path / "pulled.duckdb", source=source, source_catalog="remote")
    source.close()
    con = mirror.connect(local=True, path=dest)
    assert set(mirror.TABLES) <= {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert con.execute("SELECT count(*) FROM results").fetchone()[0] == duckdb.connect(str(mirror_file), read_only=True).execute("SELECT count(*) FROM results").fetchone()[0]
    con.close()
    assert not dest.with_name(dest.name + ".tmp").exists()


def test_missing_config(monkeypatch):
    monkeypatch.setattr(mirror, "_load_env", lambda: None)
    monkeypatch.delenv("ULTRASIGNUP_QUACK_URI", raising=False)
    monkeypatch.delenv("ULTRASIGNUP_QUACK_TOKEN", raising=False)
    monkeypatch.delenv("ULTRASIGNUP_SOURCE", raising=False)
    with pytest.raises(mirror.MirrorConfigError):
        mirror.connect()


def test_source_env_selects_local(monkeypatch, mirror_file):
    monkeypatch.setattr(mirror, "_load_env", lambda: None)
    monkeypatch.setenv("ULTRASIGNUP_SOURCE", "local")
    con = mirror.connect(path=mirror_file)
    assert con.execute("SELECT count(*) FROM results").fetchone()[0] > 0
    con.close()
    monkeypatch.setenv("ULTRASIGNUP_SOURCE", "nuc")
    with pytest.raises(mirror.MirrorConfigError):
        mirror.connect(path=mirror_file)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ULTRASIGNUP_QUACK_TOKEN") and not (mirror.REPO_ROOT / ".env").exists(), reason="no mirror credentials")
def test_live_mirror_contract():
    con = mirror.connect()
    columns = {r[0] for r in con.execute("DESCRIBE results").fetchall()}
    assert set(RESULT_COLUMNS) <= columns
    sample = con.execute(f"SELECT {', '.join(RESULT_COLUMNS)} FROM results USING SAMPLE 2000").df()
    statuses = {r[0] for r in con.execute("SELECT DISTINCT status FROM results").fetchall()}
    formats = {r[0] for r in con.execute("SELECT DISTINCT race_format FROM races").fetchall()}
    con.close()
    assert statuses <= {"finished", "finished_small_field", "unofficial", "dnf", "dns", "dq", None}
    assert formats <= {"distance", "fixed_time", "relay", None}
    sample[["virtual", "cancelled"]] = False
    assert len(process_results(sample)) > 0
    races = {r[0] for r in mirror.connect().execute("DESCRIBE races").fetchall()}
    assert {"did", "virtual", "cancelled"} <= races
