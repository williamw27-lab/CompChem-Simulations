### calculations of charge density to model dissipation (ChatGPT)

# hydrogen_sim/liouvillian.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.sparse import identity, kron
from scipy.sparse import issparse

# Prefer the new sparray constructors when available; fall back otherwise.
try:
    from scipy.sparse import csc_array as _csc_type  # SciPy newer
except Exception:  # pragma: no cover
    from scipy.sparse import csc_matrix as _csc_type  # SciPy older

# Type alias for "some SciPy sparse type" (matrix OR array).
# We intentionally do NOT import spmatrix/sparray here to avoid version/type-check pain.
SparseLike = object

Array = np.ndarray


# -------------------------
# Vectorization helpers
# -------------------------

def vec(rho: Array) -> Array:
    """Column-stacking vec(rho) with Fortran order. Shape (N*N,)."""
    return np.asarray(rho, dtype=np.complex128).reshape(-1, order="F")


def unvec(v: Array, N: int) -> Array:
    """Inverse of vec(). Returns (N,N)."""
    return np.asarray(v, dtype=np.complex128).reshape((N, N), order="F")


# -------------------------
# Sparse construction helper
# -------------------------

def _to_csc_sparse(A: Array) -> SparseLike:
    """
    Convert dense ndarray to CSC sparse (sparray if available, else spmatrix).
    """
    A = np.asarray(A, dtype=np.complex128)
    S = _csc_type(A)
    return S


def _zero_super(N: int) -> SparseLike:
    """Return a zero (N^2 x N^2) sparse operator."""
    return _csc_type((N * N, N * N), dtype=np.complex128)


# -------------------------
# Superoperators
# -------------------------

def commutator_super(H: Array) -> SparseLike:
    """
    Superoperator for -i [H, rho] acting on vec(rho) (column-stacking):
      vec(-i[H,rho]) = -i (I⊗H - H^T⊗I) vec(rho)
    """
    Hs = _to_csc_sparse(H)
    N = Hs.shape[0]
    I = identity(N, format="csc", dtype=np.complex128)
    # kron works with sparse matrices/arrays and returns a matching sparse type
    return (-1j) * (kron(I, Hs) - kron(Hs.T, I))


def dissipator_super(L: Array) -> SparseLike:
    """
    Superoperator for D_L(rho) = L rho L† - 1/2 {L†L, rho}.

    Using column-stacking:
      vec(L rho L†) = (L* ⊗ L) vec(rho)
      vec((L†L) rho) = (I ⊗ (L†L)) vec(rho)
      vec(rho (L†L)) = ((L†L)^T ⊗ I) vec(rho)
    """
    Ls = _to_csc_sparse(L)
    N = Ls.shape[0]
    I = identity(N, format="csc", dtype=np.complex128)

    LdL = (Ls.conj().T @ Ls)                      # L†L
    jump = kron(Ls.conj(), Ls)                    # L* ⊗ L
    loss = 0.5 * (kron(I, LdL) + kron(LdL.T, I))  # 1/2(I⊗LdL + (LdL)^T⊗I)
    return jump - loss


def sum_dissipators_super(L_list: Sequence[Array], N: int) -> SparseLike:
    """Sum_k dissipator_super(L_k). If empty, return zero superoperator."""
    if len(L_list) == 0:
        return _zero_super(N)

    out: Optional[SparseLike] = None
    for L in L_list:
        term = dissipator_super(L)
        out = term if out is None else (out + term)
    assert out is not None
    return out


# -------------------------
# Collapse operator helpers
# -------------------------

def make_collapse(N: int, i_from: int, i_to: int, gamma: float) -> Array:
    """
    Create L = sqrt(gamma) |to><from| in the chosen basis.
    i_from: excited/source index
    i_to:   lower/target index
    """
    L = np.zeros((N, N), dtype=np.complex128)
    L[i_to, i_from] = np.sqrt(float(gamma))
    return L


# -------------------------
# Liouvillian decomposition
# -------------------------

def compile_E1_spontaneous(
        basis,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        gamma_scale: float = 1.0,
        min_gamma: float = 0.0):
    """
    Build L_list for all E1-allowed downward transitions within the basis.
    Rates are relative unless you calibrate gamma_scale.

    Returns: list of (L, meta) where meta helps debugging/inspection.
    """
    N = basis.N
    L_list = []
    meta = []

    # precompute dipole strength matrix element squared in a rotationally fair way
    dip2 = np.abs(X)**2 + np.abs(Y)**2 + np.abs(Z)**2  # (N,N)

    energies = np.array([orb.E for orb in basis.orbitals], dtype=float)

    for i, oi in enumerate(basis.orbitals):      # source (excited)
        for j, oj in enumerate(basis.orbitals):  # target (lower)
            if energies[i] <= energies[j]:
                continue  # not downward

            # E1 selection rules (hydrogen)
            if abs(oi.l - oj.l) != 1:
                continue
            if abs(oi.m - oj.m) > 1:
                continue

            omega = energies[i] - energies[j]  # >0 in a.u.
            strength = dip2[j, i]              # |<j|r|i>|^2

            gamma = gamma_scale * (omega**3) * strength
            if gamma <= min_gamma:
                continue

            L = make_collapse(N=N, i_from=i, i_to=j, gamma=gamma)
            L_list.append(L)
            meta.append({"from": oi.key, "to": oj.key, "gamma": float(gamma), "omega": float(omega)})

    return L_list, meta

@dataclass(frozen=True)
class LiouvillianDecomposition:
    """
    Represents L(t) = L0 + E(t)*LE for H(t)=H0 - E(t) D, with fixed dissipators.

    L0 and LE may be SciPy sparse matrices (spmatrix) or sparse arrays (sparray).
    """
    L0: SparseLike
    LE: SparseLike
    N: int

    def L_of_t(self, E: float) -> SparseLike:
        return self.L0 + (float(E) * self.LE)


def build_L0_LE(H0: Array, D: Array, L_list: Sequence[Array]) -> LiouvillianDecomposition:
    """
    For H(t)=H0 - E(t)*D:
      -i[H(t),·] = -i[H0,·] + E(t) * i[D,·]

    commutator_super(D) returns -i[D,·], so i[D,·] = - commutator_super(D).
    Therefore:
      L0 = commutator_super(H0) + sum dissipators
      LE = - commutator_super(D)
    """
    H0 = np.asarray(H0, dtype=np.complex128)
    D = np.asarray(D, dtype=np.complex128)

    if H0.shape != D.shape or H0.ndim != 2 or H0.shape[0] != H0.shape[1]:
        raise ValueError("H0 and D must be square matrices of the same shape.")

    N = H0.shape[0]

    L_H0 = commutator_super(H0)
    L_diss = sum_dissipators_super(L_list, N=N)
    L0 = L_H0 + L_diss

    LE = -1 * commutator_super(D)

    return LiouvillianDecomposition(L0=L0, LE=LE, N=N)


# -------------------------
# Optional: cleanup helper
# -------------------------

def enforce_rho_physical(rho: Array) -> Array:
    """
    Light cleanup: Hermitize and renormalize trace. (Does not guarantee positivity.)
    """
    rho = np.asarray(rho, dtype=np.complex128)
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    rho = rho / complex(tr)  # cast avoids numpy scalar typing issues
    return rho