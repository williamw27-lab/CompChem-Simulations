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
    hermiticity_error,
    min_eig_hermitized,
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
    herm_error: Array
    rho_final: Array
    t_snaps: Array
    rho_snaps: Array
    min_eig: Array

def run_rho(
    rho0: Array,
    t: Array,
    dt: float,
    E_of_t: Callable[[float], float],
    decomp: LiouvillianDecomposition,
    H0: Optional[Array] = None,
    store_every: int = 1,
    rho_store_every: int = 20,
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

    # diagnostics storage
    store_idx = np.arange(0, n_steps, store_every, dtype=int)
    m = len(store_idx)

    # rho and diagnostics storage
    store_rho_idx = np.arange(0,n_steps,rho_store_every,dtype=int)
    n = len(store_rho_idx)

    # all diagnostics
    energy = np.full(m, np.nan, dtype=np.float64)
    pops = np.zeros((m, N), dtype=np.float64)
    tr = np.zeros(m, dtype=np.float64)
    pur = np.zeros(m, dtype=np.float64)
    herm_error = np.zeros(m,dtype=np.float64)
    min_eig = np.zeros(n,dtype=np.float64)

    # snapshots
    rho_snaps = []
    t_snaps = []

    out_k = 0
    out_l = 0

    # stepper
    for k in range(n_steps):
        # storing diagnostics (hermiticity error, trace, purity, populations, energy) every store every
        if k == store_idx[out_k]:
            herm_error[out_k] = hermiticity_error(rho)
            tr[out_k] = float(np.real(np.trace(rho)))
            pur[out_k] = float(np.real(np.trace(rho @ rho)))
            pops[out_k] = np.real(np.diag(rho))
            if H0 is not None:
                energy[out_k] = float(np.real(np.trace(rho @ H0)))
            out_k += 1
            if out_k >= m:
                # All requested diagnostics have been stored; keep stepping silently.
                pass
        # storing rho snapshots and positivity
        if k == store_rho_idx[out_l]:
            min_eig[out_l] = min_eig_hermitized(rho)
            rho_snaps.append(rho.copy())
            t_snaps.append(t[k])
            out_l += 1
            if out_l >= n:
                # All requested diagnostics have been stored; keep stepping silently.
                pass

        # evolve rho
        if k < n_steps - 1:
            rho = step_rho_expm(rho, t[k], dt, E_of_t, decomp, cleanup=cleanup)

    return RhoTrajectory(
        t=t[store_idx], # not all t is stored (only store when broad diagnostics are stored)
        energy_H0=energy,
        pops=pops,
        trace=tr,
        purity=pur,
        herm_error=herm_error,
        rho_final=rho,
        t_snaps=np.asarray(t_snaps),
        rho_snaps=np.asarray(rho_snaps),
        min_eig=min_eig
    )


# Optional: CN coefficient step (kept for regression)
from scipy.linalg import solve

def crank_nicolson_step(c, t, dt, H_of_t, ops, E_of_t):
    t_mid = t + 0.5*dt

    Hmid = H_of_t(t_mid, ops, E_of_t)  # NxN complex/real Hermitian matrix

    I = np.eye(Hmid.shape[0], dtype=np.complex128)
    A = I + 0.5j*dt*Hmid
    B = I - 0.5j*dt*Hmid

    rhs = B @ c
    c_next = solve(A, rhs, assume_a='gen')  # small N -> fine

    return c_next

def run_c(
    c0: Array,
    ts: Array,
    dt: float,
    H_of_t: Callable,
    ops: object,
    E_of_t: Callable
) -> np.ndarray:
    """
    evolve coefficients method, returning a (T,N) array of all coefficients
    """
    c_array = np.empty(shape=(len(ts),len(c0)),dtype=np.complex128)
    c_array[0] = c0

    for step in range(1,len(ts)):
        c_array[step] = crank_nicolson_step(c_array[step-1],ts[step],dt,H_of_t,ops,E_of_t)


    return c_array