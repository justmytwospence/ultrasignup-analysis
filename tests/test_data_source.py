import os

import duckdb
import pytest
from conftest import build_db

from utils import data_source
from utils.data_processing import RESULT_COLUMNS, process_results


def test_connect_local(db_file):
    con = data_source.connect(local=True, path=db_file)
    assert con.execute("SELECT count(*) FROM results").fetchone()[0] > 0
    con.close()


def test_connect_local_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        data_source.connect(local=True, path=tmp_path / "nope.duckdb")


def test_schema_major_mismatch(tmp_path):
    path = build_db(tmp_path / "v2.duckdb", schema_version="2.0")
    with pytest.raises(data_source.SchemaMismatch):
        data_source.connect(local=True, path=path)


def test_pull_round_trip(db_file, tmp_path):
    source = duckdb.connect()
    source.execute(f"ATTACH '{db_file}' AS remote (READ_ONLY)")
    dest = data_source.pull(dest=tmp_path / "pulled.duckdb", source=source, source_catalog="remote")
    source.close()
    con = data_source.connect(local=True, path=dest)
    assert set(data_source.TABLES) <= {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert con.execute("SELECT count(*) FROM results").fetchone()[0] == duckdb.connect(str(db_file), read_only=True).execute("SELECT count(*) FROM results").fetchone()[0]
    con.close()
    assert not dest.with_name(dest.name + ".tmp").exists()


def test_missing_config(monkeypatch):
    monkeypatch.setattr(data_source, "_load_env", lambda: None)
    monkeypatch.delenv("ULTRASIGNUP_QUACK_URI", raising=False)
    monkeypatch.delenv("ULTRASIGNUP_QUACK_TOKEN", raising=False)
    monkeypatch.delenv("ULTRASIGNUP_SOURCE", raising=False)
    with pytest.raises(data_source.DataSourceConfigError):
        data_source.connect()


def test_source_env_selects_local(monkeypatch, db_file):
    monkeypatch.setattr(data_source, "_load_env", lambda: None)
    monkeypatch.setenv("ULTRASIGNUP_SOURCE", "local")
    con = data_source.connect(path=db_file)
    assert con.execute("SELECT count(*) FROM results").fetchone()[0] > 0
    con.close()
    monkeypatch.setenv("ULTRASIGNUP_SOURCE", "nuc")
    with pytest.raises(data_source.DataSourceConfigError):
        data_source.connect(path=db_file)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ULTRASIGNUP_QUACK_TOKEN") and not (data_source.REPO_ROOT / ".env").exists(), reason="no data source credentials")
def test_live_contract():
    con = data_source.connect()
    columns = {r[0] for r in con.execute("DESCRIBE results").fetchall()}
    assert set(RESULT_COLUMNS) <= columns
    sample = con.execute(f"SELECT {', '.join(RESULT_COLUMNS)} FROM results USING SAMPLE 2000").df()
    statuses = {r[0] for r in con.execute("SELECT DISTINCT status FROM results").fetchall()}
    formats = {r[0] for r in con.execute("SELECT DISTINCT race_format FROM races").fetchall()}
    con.close()
    assert statuses <= {"finished", "finished_small_field", "unofficial", "dnf", "dns", "dq", None}
    assert formats <= {"distance", "fixed_time", "relay", None}
    sample[["virtual", "cancelled"]] = False
    sample["reports_dnfs"] = True
    assert len(process_results(sample)) > 0
    races = {r[0] for r in data_source.connect().execute("DESCRIBE races").fetchall()}
    assert {"did", "virtual", "cancelled", "reports_dnfs"} <= races
