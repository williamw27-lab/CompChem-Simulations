### calculate wavefunction for plotting (ChatGPT)

# hydrogen_sim/projection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

Array = np.ndarray


def cart_to_sph(x: Array, y: Array, z: Array) -> Tuple[Array, Array, Array]:
    """
    Convert Cartesian arrays to spherical coordinates (r, theta, phi).
    theta: polar angle in [0, pi]
    phi: azimuth in [-pi, pi]
    """
    r = np.sqrt(x*x + y*y + z*z)
    # Avoid division by zero at r=0
    theta = np.zeros_like(r)
    nonzero = r > 0
    theta[nonzero] = np.arccos(np.clip(z[nonzero] / r[nonzero], -1.0, 1.0))
    phi = np.arctan2(y, x)
    return r, theta, phi


@dataclass(frozen=True)
class SliceGrid:
    """
    A 2D Cartesian slice grid.

    plane:
      - "xz": y=0, axes are x and z
      - "yz": x=0, axes are y and z
      - "xy": z=0, axes are x and y
    """
    plane: Literal["xz", "yz", "xy"] = "xz"
    extent: float = 30.0      # half-width in a.u.
    n: int = 301              # points per axis (n x n grid)
    fixed_value: float = 0.0  # value of the fixed axis (e.g. y=0 for xz)

    def mesh(self) -> Tuple[Array, Array, Array, Array, Array]:
        """
        Returns:
          X, Y, Z : (n,n) mesh in Cartesian
          A, B    : (n,n) mesh for plotting coordinates (axis1, axis2)
        """
        a = np.linspace(-self.extent, self.extent, self.n)
        A, B = np.meshgrid(a, a, indexing="xy")

        if self.plane == "xz":
            X, Z = A, B
            Y = np.full_like(X, self.fixed_value)
            return X, Y, Z, A, B
        if self.plane == "yz":
            Y, Z = A, B
            X = np.full_like(Y, self.fixed_value)
            return X, Y, Z, A, B
        if self.plane == "xy":
            X, Y = A, B
            Z = np.full_like(X, self.fixed_value)
            return X, Y, Z, A, B
        raise ValueError(f"Unknown plane: {self.plane}")


class GridProjector:
    """
    Precomputes psi[p, i] = psi_i(r_p) for a chosen grid, so density evaluations are fast.

    Expects:
      - basis.orbitals : list of orbital objects
      - orbital.psi(r, theta, phi) : returns complex scalar/array (broadcastable)
    """

    def __init__(self, basis, grid: SliceGrid):
        self.basis = basis
        self.grid = grid
        self.N = basis.N

        X, Y, Z, A, B = grid.mesh()
        self.X = X
        self.Y = Y
        self.Z = Z
        self.A = A
        self.B = B

        r, theta, phi = cart_to_sph(X, Y, Z)
        self.r = r
        self.theta = theta
        self.phi = phi

        # Flatten points for storage: p = n*n
        self.p_shape = X.shape
        self.P = X.size

        self.psi = self._precompute_psi()  # shape (P, N)

    def _precompute_psi(self) -> Array:
        psi = np.zeros((self.P, self.N), dtype=np.complex128)

        r = self.r.reshape(-1)
        th = self.theta.reshape(-1)
        ph = self.phi.reshape(-1)

        for i, orb in enumerate(self.basis.orbitals):
            # orb.psi should broadcast over arrays
            vals = orb.psi_nlm(r, th, ph)
            psi[:, i] = np.asarray(vals, dtype=np.complex128)

        return psi

    def density_from_rho(self, rho: Array) -> Array:
        """
        Compute P(p) = <r_p|rho|r_p> on the grid.
        Returns a real (n,n) array.
        """
        rho = np.asarray(rho, dtype=np.complex128)
        # einsum over i,j: psi[p,i] rho[i,j] psi*[p,j]
        Pp = np.einsum("pi,ij,pj->p", self.psi, rho, self.psi.conj(), optimize=True)
        out = np.real(Pp).reshape(self.p_shape)
        # tiny negative numerical noise can happen:
        out[out < 0] = 0.0
        return out

    def density_from_c(self, c: Array) -> Array:
        """
        For a pure state c, compute |Psi(r)|^2 on the grid.
        Psi(p) = sum_i c_i psi_i(p)
        """
        c = np.asarray(c, dtype=np.complex128)
        Psi_p = self.psi @ c  # (P,)
        out = np.real(Psi_p * Psi_p.conj()).reshape(self.p_shape)
        return out

    def density_difference(self, rho_a: Array, rho_b: Array) -> Array:
        """
        Convenience: difference of densities (useful for debugging relaxation effects).
        """
        return self.density_from_rho(rho_a) - self.density_from_rho(rho_b)
