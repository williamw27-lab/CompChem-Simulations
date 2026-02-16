### Computes initial hamiltonian and dipole matrices

import numpy as np
from hydrogen_sim.integrals import _angular_overlap, _radial_r3_overlap

def compute_dipole_matrices_xyz(
    basis,
    r_max=400.0,
    quad_limit=300,
    epsabs=1e-10,
    epsrel=1e-10,
    hermitize=True,
    use_symmetry=True
):
    """
    Returns X, Y, Z dipole matrices for a basis of Orbital objects.

    basis: list[Orbital], length N
    r_max: radial cutoff (a.u.) for r-integrals; increase for higher n
    hermitize: enforce (M + M†)/2 to remove numerical asymmetry
    use_symmetry: compute only i<=j and fill j,i by conjugation (faster)
    """
    orbs = basis.orbitals
    N = len(orbs)

    # Angular factors for x,y,z
    fx = lambda th, ph: np.sin(th) * np.cos(ph)
    fy = lambda th, ph: np.sin(th) * np.sin(ph)
    fz = lambda th, ph: np.cos(th)

    X = np.zeros((N, N), dtype=np.complex128)
    Y = np.zeros((N, N), dtype=np.complex128)
    Z = np.zeros((N, N), dtype=np.complex128)

    # Optional cache for radial overlaps (saves time)
    radial_cache = {}

    for i in range(N):
        j_start = i if use_symmetry else 0
        for j in range(j_start, N):
            key = (i, j)
            if key not in radial_cache:
                Ir = _radial_r3_overlap(basis.orbitals[i], basis.orbitals[j], r_max=r_max, quad_limit=quad_limit)
                radial_cache[key] = Ir
            else:
                Ir = radial_cache[key]

            Iax = _angular_overlap(basis.orbitals[i], basis.orbitals[j], fx, dblquad_epsabs=epsabs, dblquad_epsrel=epsrel)
            Iay = _angular_overlap(basis.orbitals[i], basis.orbitals[j], fy, dblquad_epsabs=epsabs, dblquad_epsrel=epsrel)
            Iaz = _angular_overlap(basis.orbitals[i], basis.orbitals[j], fz, dblquad_epsabs=epsabs, dblquad_epsrel=epsrel)

            Xij = Ir * Iax
            Yij = Ir * Iay
            Zij = Ir * Iaz

            X[i, j] = Xij
            Y[i, j] = Yij
            Z[i, j] = Zij

            if use_symmetry and j != i:
                # Dipole operator is Hermitian, so M[j,i] = conj(M[i,j])
                X[j, i] = np.conj(Xij)
                Y[j, i] = np.conj(Yij)
                Z[j, i] = np.conj(Zij)

    if hermitize:
        X = 0.5 * (X + X.conj().T)
        Y = 0.5 * (Y + Y.conj().T)
        Z = 0.5 * (Z + Z.conj().T)

    return X, Y, Z

def dipole_matrix_for_polarization(xyz, polarization, normalize=True):
    """
    Given X,Y,Z and a polarization vector (ex,ey,ez), return D = ex X + ey Y + ez Z.
    """

    X, Y, Z = xyz

    e = np.asarray(polarization, dtype=np.complex128)
    if e.shape != (3,):
        raise ValueError("polarization must be a length-3 vector [ex, ey, ez].")

    if normalize:
        nrm = np.linalg.norm(e)
        if nrm == 0:
            raise ValueError("polarization vector must be nonzero.")
        e = e / nrm

    ex, ey, ez = e
    return ex * X + ey * Y + ez * Z

class Operators:
    def __init__(self, basis, config):
        self.H0 = np.diag([orb.E for orb in basis.orbitals])
        self.X, self.Y, self.Z = compute_dipole_matrices_xyz(basis=basis)
        self.D = dipole_matrix_for_polarization((self.X,self.Y,self.Z), config.pulse.polarization)