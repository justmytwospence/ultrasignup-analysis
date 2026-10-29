"""
Memory-safe sampling for the large models (runner effects, 1.2 / 2.2 / 2.3).

nutpie holds every stored draw in memory until sampling ends: all free (unconstrained)
parameters, plus whatever constrained or deterministic variables are listed in `var_names`.
For models with a few hundred thousand runner effects that is gigabytes per chain, so:

- warm-up draws are not kept (`save_warmup=False`),
- per-runner deterministics are left out of `var_names` and derived after sampling,
- `check_memory` estimates the peak before sampling and refuses to start if it would not fit,
  instead of letting macOS kill the kernel halfway through.
"""

from __future__ import annotations

import math
import subprocess

import pymc as pm

BYTES_PER_VALUE = 8
# nutpie's buffers, the conversion to InferenceData and our float32 copies of derived
# effects all exist at once at the end of sampling.
OVERHEAD = 2.5


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


def check_memory(model: pm.Model, var_names: list[str], chains: int, draws: int, reserve_gb: float = 6.0) -> float:
    """Estimated peak memory of sampling in GB; raises MemoryError if it exceeds what is
    available minus `reserve_gb` (left for the OS, the server and other kernels)."""
    need = stored_values(model, var_names) * BYTES_PER_VALUE * chains * draws * OVERHEAD
    have = available_bytes() - reserve_gb * 2**30
    if need > have:
        raise MemoryError(
            f"Sampling needs about {need / 2**30:.1f} GB but only {max(have, 0) / 2**30:.1f} GB is available "
            f"(after a {reserve_gb:g} GB reserve). Restart idle notebook kernels, or reduce draws."
        )
    return need / 2**30


def sample_lean(model: pm.Model, var_names: list[str], *, draws: int, tune: int = 1000, chains: int = 4, **kwargs):
    """pm.sample with nutpie, keeping only post-warm-up draws of the free parameters and
    `var_names`. Checks memory first."""
    check_memory(model, var_names, chains, draws)
    with model:
        return pm.sample(draws=draws, tune=tune, chains=chains, nuts_sampler="nutpie", var_names=var_names,
                         nuts_sampler_kwargs={"save_warmup": False}, **kwargs)


__all__ = ["available_bytes", "stored_values", "check_memory", "sample_lean"]
