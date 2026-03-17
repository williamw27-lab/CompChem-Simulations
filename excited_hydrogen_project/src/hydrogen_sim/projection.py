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

# 2D projection
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

# 3D projection
@dataclass(frozen=True)
class VolumeGrid:
    """
    Uniform 3D Cartesian grid.

    extent:
        half-width of the cube in atomic units
        x,y,z in [-extent, extent]
    n:
        number of points per axis
    """
    extent: float = 40.0
    n: int = 64

    def mesh(self) -> Tuple[Array, Array, Array]:
        a = np.linspace(-self.extent, self.extent, self.n)
        X, Y, Z = np.meshgrid(a, a, a, indexing="ij")
        return X, Y, Z

class GridProjector3D:
    """
    Precompute psi[p, i] = psi_i(r_p) on a 3D grid.

    Parameters
    ----------
    basis
        Basis object with:
          - basis.orbitals
          - basis.N
    grid
        VolumeGrid instance
    psi_dtype
        dtype used to store cached orbital values.
        complex64 is often enough for visualization and saves memory.
    """

    def __init__(
        self,
        basis,
        grid: VolumeGrid,
        psi_dtype=np.complex64,
    ):
        self.basis = basis
        self.grid = grid
        self.N = basis.N
        self.psi_dtype = psi_dtype

        X, Y, Z = grid.mesh()
        self.X = X
        self.Y = Y
        self.Z = Z

        r, theta, phi = cart_to_sph(X, Y, Z)
        self.r = r
        self.theta = theta
        self.phi = phi

        self.grid_shape = X.shape
        self.P = X.size

        self.psi = self._precompute_psi()

    def _precompute_psi(self) -> Array:
        """
        Build psi[p, i] with p the flattened grid index.
        """
        psi = np.zeros((self.P, self.N), dtype=self.psi_dtype)

        r = self.r.reshape(-1)
        th = self.theta.reshape(-1)
        ph = self.phi.reshape(-1)

        for i, orb in enumerate(self.basis.orbitals):
            vals = orb.psi_nlm(r, th, ph)
            psi[:, i] = np.asarray(vals, dtype=self.psi_dtype)

        return psi

    def density_from_rho(self, rho: Array) -> Array:
        """
        Compute real-space density from a density matrix.

        Returns
        -------
        P : ndarray, shape (nx, ny, nz)
        """
        rho = np.asarray(rho, dtype=np.complex128)

        P_flat = np.einsum(
            "pi,ij,pj->p",
            self.psi,
            rho,
            self.psi.conj(),
            optimize=True,
        )

        P = np.real(P_flat).reshape(self.grid_shape)

        # Clip tiny negative numerical noise
        P[P < 0.0] = 0.0
        return P

    def density_from_c(self, c: Array) -> Array:
        """
        Compute density from pure-state coefficients c.
        """
        c = np.asarray(c, dtype=np.complex128)

        psi_total = self.psi @ c
        P = np.real(psi_total * psi_total.conj()).reshape(self.grid_shape)
        P[P < 0.0] = 0.0
        return P

    def density_slice(self, rho: Array, axis: str = "z", index: int | None = None) -> Array:
        """
        Convenience helper: extract a central 2D slice from 3D density.

        axis : 'x', 'y', or 'z'
        index : slice index; if None, uses center
        """
        P = self.density_from_rho(rho)

        if index is None:
            index = self.grid.n // 2

        if axis == "x":
            return P[index, :, :]
        if axis == "y":
            return P[:, index, :]
        if axis == "z":
            return P[:, :, index]

        raise ValueError("axis must be 'x', 'y', or 'z'")