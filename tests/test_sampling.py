import numpy as np
import pymc as pm
import pytest

from utils import sampling


def _model():
    with pm.Model(coords={"runner": range(50)}) as m:
        s = pm.HalfNormal("s", 1.0)
        raw = pm.Normal("raw", 0, 1, dims="runner")
        pm.Deterministic("effect", s * raw, dims="runner")
        pm.Normal("y", raw.sum(), 1, observed=0.0)
    return m


def test_stored_values_counts_free_params_and_listed_only():
    m = _model()
    assert sampling.stored_values(m, ["s"]) == 51 + 1          # s_log__ + raw, plus constrained s
    assert sampling.stored_values(m, ["s", "effect"]) == 51 + 1 + 50


def test_check_memory_refuses_when_too_big(monkeypatch):
    monkeypatch.setattr(sampling, "available_bytes", lambda: 10 * 2**30)
    with pytest.raises(MemoryError):
        sampling.check_memory(_model(), ["s"], chains=4, draws=10**8)
    assert sampling.check_memory(_model(), ["s"], chains=4, draws=100, reserve_gb=1) < 1
    # in memory, warm-up draws count too
    on_disk = sampling.check_memory(_model(), ["s"], chains=4, draws=1000, tune=1000, reserve_gb=0)
    in_mem = sampling.check_memory(_model(), ["s"], chains=4, draws=1000, tune=1000, on_disk=False, reserve_gb=0)
    assert in_mem > 3 * on_disk


def test_sample_lean_drops_warmup_and_unlisted():
    trace = sampling.sample_lean(_model(), ["s", "raw"], draws=50, tune=50, chains=2, random_seed=1, progressbar=False)
    assert "effect" not in trace.posterior
    assert trace.posterior.sizes["draw"] == 50
    assert "warmup_posterior" not in trace.groups()


def test_sample_lean_on_disk(tmp_path):
    store = tmp_path / "zarr"
    trace = sampling.sample_lean(_model(), ["s", "raw"], draws=40, tune=40, chains=2, store_dir=store, random_seed=2, progressbar=False)
    assert trace.posterior.sizes["draw"] == 40
    assert "effect" not in trace.posterior
    assert any(store.rglob("*"))
