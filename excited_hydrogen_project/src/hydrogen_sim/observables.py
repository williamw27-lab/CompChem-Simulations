### calculate observable quantities as checks
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


Array = np.ndarray


def populations_from_c(c: Array) -> Array:
    """
    Populations in the basis for a pure state coefficient vector c.
    Returns real array of shape (N,).
    """
    c = np.asarray(c, dtype=np.complex128)
    return np.abs(c) ** 2


def populations_from_rho(rho: Array) -> Array:
    """
    Populations in the basis for a density matrix rho.
    Returns real array of shape (N,).
    """
    rho = np.asarray(rho, dtype=np.complex128)
    return np.real(np.diag(rho))


def norm_from_c(c: Array) -> float:
    """
    Norm of coefficient vector (should be 1 for unitary evolution).
    """
    c = np.asarray(c, dtype=np.complex128)
    return float(np.vdot(c, c).real)


def trace_from_rho(rho: Array) -> float:
    """
    Trace of density matrix (should be 1 for CPTP evolution).
    """
    rho = np.asarray(rho, dtype=np.complex128)
    return float(np.trace(rho).real)


def purity(rho: Array) -> float:
    """
    Purity Tr(rho^2). Equals 1 for pure states, <1 for mixed.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    return float(np.trace(rho @ rho).real)


def expectation_from_c(c: Array, A: Array) -> complex:
    """
    <A> = c† A c
    """
    c = np.asarray(c, dtype=np.complex128)
    A = np.asarray(A, dtype=np.complex128)
    return complex(np.vdot(c, A @ c))


def expectation_from_rho(rho: Array, A: Array) -> complex:
    """
    <A> = Tr(rho A)
    """
    rho = np.asarray(rho, dtype=np.complex128)
    A = np.asarray(A, dtype=np.complex128)
    return np.trace(rho @ A)


def energy_from_c(c: Array, H: Array) -> float:
    """
    Energy expectation value for pure state.
    Returns real scalar.
    """
    return float(np.real(expectation_from_c(c, H)))


def energy_from_rho(rho: Array, H: Array) -> float:
    """
    Energy expectation value for density matrix.
    Returns real scalar.
    """
    return float(np.real(expectation_from_rho(rho, H)))


def dipole_from_c(c: Array, D: Array) -> float:
    """
    <D> for pure state (often real, but return real part).
    """
    return float(np.real(expectation_from_c(c, D)))


def dipole_from_rho(rho: Array, D: Array) -> float:
    """
    <D> = Tr(rho D) (often real, but return real part).
    """
    return float(np.real(expectation_from_rho(rho, D)))


def hermiticity_error(rho: Array) -> float:
    """
    ||rho - rho†||_F as a diagnostic.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    diff = rho - rho.conj().T
    return float(np.linalg.norm(diff))


def positivity_min_eig(rho: Array) -> float:
    """
    Minimum eigenvalue of rho (should be >= 0).
    Warning: O(N^3) eigendecomposition; use sparingly.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    # force Hermitian for stable eigs
    rhoH = 0.5 * (rho + rho.conj().T)
    evals = np.linalg.eigvalsh(rhoH)
    return float(np.min(evals).real)


@dataclass
class RhoDiagnostics:
    trace: float
    purity: float
    herm_err: float
    min_eig: Optional[float] = None


def rho_diagnostics(rho: Array, check_positivity: bool = False) -> RhoDiagnostics:
    """
    Bundle common diagnostics for density matrix runs.
    Set check_positivity=True only occasionally (expensive).
    """
    tr = trace_from_rho(rho)
    pur = purity(rho)
    herr = hermiticity_error(rho)
    mine = positivity_min_eig(rho) if check_positivity else None
    return RhoDiagnostics(trace=tr, purity=pur, herm_err=herr, min_eig=mine)
