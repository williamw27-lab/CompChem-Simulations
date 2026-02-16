### Stores the old crank-nicolson stepper and the new stepper for rho (ChatGPT)

# hydrogen_sim/steppers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.sparse.linalg import expm_multiply

from hydrogen_sim.liouvillian import (
    vec,
    unvec,
    enforce_rho_physical,
    LiouvillianDecomposition,
)

Array = np.ndarray

from scipy.interpolate import interp1d

def make_E_callable(t_grid: np.ndarray, E_grid: np.ndarray):
    t_grid = np.asarray(t_grid, dtype=float)
    E_grid = np.asarray(E_grid, dtype=float)

    # interpolation within range; outside → 0 (or raise)
    return interp1d(
        t_grid,
        E_grid,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )

def step_rho_expm(
    rho: Array,
    t: float,
    dt: float,
    E_of_t: Callable[[float], float],
    decomp: LiouvillianDecomposition,
    cleanup: bool = True,
) -> Array:
    """
    One step:
      vec(rho_{n+1}) = exp(dt * L(t_mid)) vec(rho_n)

    Uses midpoint time t + dt/2 (matches your CN midpoint idea).
    Compatible with SciPy sparse matrices or sparse arrays.
    """
    N = decomp.N
    tm = t + 0.5 * dt
    Em = float(E_of_t(tm))

    Lm = decomp.L_of_t(Em)  # sparse matrix OR sparse array (both OK)

    v = vec(rho)
    v_next = expm_multiply(Lm * dt, v)  # scalar multiply works for both sparse types
    rho_next = unvec(v_next, N)

    if cleanup:
        rho_next = enforce_rho_physical(rho_next)

    return rho_next


@dataclass
class RhoTrajectory:
    t: Array
    energy_H0: Array
    pops: Array
    trace: Array
    purity: Array
    rho_final: Array


def run_rho(
    rho0: Array,
    t: Array,
    dt: float,
    E_of_t: Callable[[float], float],
    decomp: LiouvillianDecomposition,
    H0: Optional[Array] = None,
    store_every: int = 1,
    cleanup: bool = True,
) -> RhoTrajectory:
    """
    Propagate density matrix across a time grid, recording diagnostics every store_every steps.
    """
    rho = np.asarray(rho0, dtype=np.complex128)
    N = rho.shape[0]
    if rho.shape != (N, N):
        raise ValueError("rho0 must be square (N,N).")
    if N != decomp.N:
        raise ValueError("rho0 dimension does not match Liouvillian decomposition.")
    if H0 is not None:
        H0 = np.asarray(H0, dtype=np.complex128)

    t = np.asarray(t, dtype=np.float64)
    n_steps = len(t)

    store_idx = np.arange(0, n_steps, store_every, dtype=int)
    m = len(store_idx)

    energy = np.full(m, np.nan, dtype=np.float64)
    pops = np.zeros((m, N), dtype=np.float64)
    tr = np.zeros(m, dtype=np.float64)
    pur = np.zeros(m, dtype=np.float64)

    out_k = 0
    for k in range(n_steps):
        if k == store_idx[out_k]:
            tr[out_k] = float(np.real(np.trace(rho)))
            pur[out_k] = float(np.real(np.trace(rho @ rho)))
            pops[out_k] = np.real(np.diag(rho))
            if H0 is not None:
                energy[out_k] = float(np.real(np.trace(rho @ H0)))
            out_k += 1
            if out_k >= m:
                # All requested diagnostics have been stored; keep stepping silently.
                pass

        if k < n_steps - 1:
            rho = step_rho_expm(rho, t[k], dt, E_of_t, decomp, cleanup=cleanup)

    return RhoTrajectory(
        t=t[store_idx],
        energy_H0=energy,
        pops=pops,
        trace=tr,
        purity=pur,
        rho_final=rho,
    )


# Optional: CN coefficient step (kept for regression)
from scipy.linalg import solve

def crank_nicolson_step(c, t, dt, H_of_t): # replace H with H(t)
    t_mid = t + 0.5*dt

    Hmid = H_of_t(t_mid)  # NxN complex/real Hermitian matrix

    I = np.eye(Hmid.shape[0], dtype=np.complex128)
    A = I + 0.5j*dt*Hmid
    B = I - 0.5j*dt*Hmid

    rhs = B @ c
    c_next = solve(A, rhs, assume_a='gen')  # small N -> fine

    return c_next