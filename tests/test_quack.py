"""The live path over a real Quack server (a subprocess serving the synthetic mirror).

Catches what a local file cannot: e.g. bound parameters do not travel through CONNECT.
Skipped when the quack extension cannot be installed (offline).
"""

import socket
import subprocess
import sys
import textwrap
import time

import duckdb
import pytest

from utils import mirror
from utils.data_processing import load_results, process_results

TOKEN = "test_token_0123456789abcdef0123"


def _quack_available() -> bool:
    try:
        duckdb.connect().execute("INSTALL quack; LOAD quack")
        return True
    except duckdb.Error:
        return False


pytestmark = pytest.mark.skipif(not _quack_available(), reason="quack extension unavailable")


@pytest.fixture
def quack_server(mirror_file):
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    script = textwrap.dedent(f"""
        import duckdb, time
        con = duckdb.connect({str(mirror_file)!r})
        con.execute("CALL quack_serve('quack:localhost:{port}', token = '{TOKEN}')")
        print("SERVING", flush=True)
        time.sleep(120)
    """)
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if proc.stdout.readline().strip() == "SERVING":
                break
        else:
            pytest.fail("quack server did not start")
        yield f"quack:localhost:{port}"
    finally:
        proc.kill()
        proc.wait()


def test_live_connect_load_and_pull(quack_server, monkeypatch, tmp_path):
    monkeypatch.setattr(mirror, "_load_env", lambda: None)
    monkeypatch.setenv("ULTRASIGNUP_QUACK_URI", quack_server)
    monkeypatch.setenv("ULTRASIGNUP_QUACK_TOKEN", TOKEN)

    con = mirror.connect()
    frame = process_results(load_results(con))
    con.close()
    assert set(frame["event_distance_id"]) == {1, 2}

    pulled = mirror.pull(dest=tmp_path / "pulled.duckdb")
    local = mirror.connect(local=True, path=pulled)
    assert local.execute("SELECT count(*) FROM results").fetchone()[0] == 16
    local.close()
