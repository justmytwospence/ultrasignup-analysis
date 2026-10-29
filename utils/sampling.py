"""
Memory-safe sampling for the large models (runner effects, 1.2 / 2.2 / 2.3).

nutpie holds every stored draw in memory until sampling ends: all free (unconstrained)
parameters, plus whatever constrained or deterministic variables are listed in `var_names`.
For models with a few hundred thousand runner effects that is gigabytes per chain, so:

- draws stream to a zarr store on disk (`store_dir`) instead of nutpie's in-memory trace, which
  holds warm-up draws too until sampling ends,
- warm-up draws are not returned (`save_warmup=False`),
- per-runner deterministics are left out of `var_names` and derived after sampling,
- `check_memory` estimates the peak before sampling and refuses to start if it would not fit,
  instead of letting macOS kill the kernel halfway through.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pymc as pm

BYTES_PER_VALUE = 8
# nutpie's in-memory (Arrow) trace, measured: about 2.75x the raw size of the draws it holds.
IN_MEMORY_OVERHEAD = 2.75


def available_bytes() -> int:
    """Memory available to a new allocation without swapping (free + reclaimable)."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:  # pragma: no cover - psutil ships with the notebook dependencies
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
        page = 16384
        pages = sum(int(line.split(":")[1].strip(" .")) for line in out.splitlines()
                    if line.startswith(("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")))
        return pages * page


def stored_values(model: pm.Model, var_names: list[str]) -> int:
    """Numbers stored per draw: every free parameter (unconstrained) plus `var_names`."""
    ip = model.initial_point()
    n = sum(v.size for v in ip.values())
    extra = [name for name in var_names if name not in ip]
    if extra:
        shape_fn = model.compile_fn([model[name].shape for name in extra], inputs=model.value_vars, on_unused_input="ignore")
        n += sum(math.prod(s) for s in shape_fn(ip))
    return n


def check_memory(model: pm.Model, var_names: list[str], chains: int, draws: int, tune: int = 1000,
                 on_disk: bool = True, reserve_gb: float = 6.0) -> float:
    """Estimated peak memory in GB; raises MemoryError if it exceeds what is available minus
    `reserve_gb` (left for the OS, the server and other kernels).

    In memory, nutpie keeps every draw, warm-up included, until sampling ends (`save_warmup`
    only filters at conversion), at about 2.75x the raw size (measured). On disk (zarr), the
    peak is reading the posterior back once to derive effects and save the trace.
    """
    values = stored_values(model, var_names) * BYTES_PER_VALUE * chains
    need = values * draws * 1.5 if on_disk else values * (draws + tune) * IN_MEMORY_OVERHEAD
    have = available_bytes() - reserve_gb * 2**30
    if need > have:
        raise MemoryError(
            f"Sampling needs about {need / 2**30:.1f} GB but only {max(have, 0) / 2**30:.1f} GB is available "
            f"(after a {reserve_gb:g} GB reserve). Restart idle notebook kernels and marimo servers, or reduce draws."
        )
    return need / 2**30


def sample_lean(model: pm.Model, var_names: list[str], *, draws: int, tune: int = 1000, chains: int = 4,
                store_dir: str | Path | None = None, **kwargs):
    """pm.sample with nutpie, keeping only post-warm-up draws of the free parameters and
    `var_names`. With `store_dir`, draws stream to a zarr store there instead of memory (the
    returned trace reads from it lazily; delete the directory once the trace is saved).
    Checks memory first, and refuses a trace with fewer draws than asked (interrupted)."""
    check_memory(model, var_names, chains, draws, tune, on_disk=store_dir is not None)
    nuts_kwargs = {"save_warmup": False}
    if store_dir is not None:
        import nutpie

        store_dir = Path(store_dir)
        if store_dir.exists():
            shutil.rmtree(store_dir)
        store_dir.mkdir(parents=True)
        nuts_kwargs["zarr_store"] = nutpie.zarr_store.LocalStore(str(store_dir))
    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains, nuts_sampler="nutpie", var_names=var_names,
                          nuts_sampler_kwargs=nuts_kwargs, **kwargs)
    got = trace.posterior.sizes["draw"]
    if got != draws:
        # nutpie returns what it has when sampling is interrupted; never let that be cached
        raise RuntimeError(f"Sampling returned {got} of {draws} draws per chain (interrupted?); not using this trace.")
    return trace



def load_store(store_dir: str | Path, draws: int):
    """The posterior and sample stats of a finished zarr store from `sample_lean`, or None if it
    is missing or incomplete. Lets a fit whose save failed be recovered without resampling."""
    import arviz as az
    import xarray as xr

    store_dir = Path(store_dir)
    if not store_dir.exists():
        return None
    try:
        tree = xr.open_datatree(str(store_dir), engine="zarr", consolidated=False)
        post, stats = tree["posterior"].to_dataset(), tree["sample_stats"].to_dataset()
    except (KeyError, OSError, ValueError):
        return None
    if post.sizes.get("draw") != draws or stats.sizes.get("draw") != draws:
        return None
    return az.InferenceData(posterior=post, sample_stats=stats)


def netcdf_safe(idata):
    """Load string coordinates and variables (zarr-backed traces carry numpy StringDType, which
    may only show once loaded) as plain object strings, which netCDF can store."""
    for group in idata.groups():
        ds = idata[group]
        fixed = {}
        for name, var in ds.variables.items():
            if var.dtype.kind in "OTU":
                fixed[name] = (var.dims, np.asarray([str(x) for x in np.asarray(var.values).ravel()], dtype=object).reshape(var.shape))
        if fixed:
            coords = {k: v for k, v in fixed.items() if k in ds.coords}
            data = {k: v for k, v in fixed.items() if k not in ds.coords}
            idata[group] = ds.assign_coords(coords).assign(data)
    return idata


__all__ = ["available_bytes", "stored_values", "check_memory", "sample_lean", "load_store", "netcdf_safe"]
