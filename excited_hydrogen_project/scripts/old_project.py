### Simulate the excitation and relaxation of a hydrogen atom using light

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y, genlaguerre
from scipy.constants import physical_constants
from scipy.linalg import solve
from scipy import integrate
from numpy.linalg import norm

## ! creating a set of basis functions - from "orbital_visualization.py"
a0 = 1

class Orbital:
    def __init__(self,n,l,m): # when I create an object of the orbital class, I need to provide n, l, m, but energy is created as itself
        self.n = n
        self.l = l
        self.m = m
        self.E = -1 / (2 * n**2)

    def orb_to_string(self):
        if self.l == 0:
            sub = 's'
        elif self.l == 1:
            sub = 'p'
        elif self.l == 2:
            sub = 'd'
        elif self.l == 3:
            sub = 'f'

        return f'{self.n}{sub}{self.m}'
    
    def key(self):
        return (self.n,self.l,self.m)
    
    # Radial hydrogen wavefunction
    def R_nlm(self, r):
        rho = 2 * r / (self.n)
        # Associated Laguerre polynomial
        L = genlaguerre(self.n-self.l-1, 2*self.l+1)(rho)

        # Normalization constant
        num = 2.0 / (self.n * a0) # changed n**2 to n
        pref = num**1.5 * np.sqrt(math.factorial(self.n-self.l-1) / (2*self.n*math.factorial(self.n+self.l)))

        return pref * np.exp(-rho/2) * rho**self.l * L


    # Angular part (Y_l^m)
    # represents the angular component of the wavefunction in terms of theta and phi
    def Y_lm(self, theta, phi):
        return sph_harm_y(self.m, self.l, phi, theta)


    # Full separated wavefunction ψ = R(r) Y(θ,φ)
    def psi_nlm(self, r, theta, phi):
        return self.R_nlm(r) * self.Y_lm(theta,phi)

def make_hydrogen_orbitals(nmax):

    orbs_list = []

    for n in range(1,nmax+1):
        for l in range(n):
            if 0 <= l <= 3:
                for m in range(-l,l+1):
                    orbs_list.append(Orbital(n,l,m))
            
            else:
                continue

    return orbs_list

class Basis:
    def __init__(self, orbitals):
        self.orbitals = list(orbitals)
        self.N = len(self.orbitals)

        self.key_to_index = {
            orb.key: i for i, orb in enumerate(self.orbitals)
        }

    # accessing
    def numbers_to_index(self, n = None, l = None, m = None, key = None):
        if key is not None:
            return self.key_to_index[key]
        return self.key_to_index[(n,l,m)]
    
    def index_to_numbers(self,i):
        return self.orbitals[i]
    
    # selecting orbitals
    def select(self, predicate):
        """
        Return indices i for which predicate(orbital) is True
        """
        return [i for i, orb in enumerate(self.orbitals) if predicate(orb)]

    def select_n(self, n):
        return self.select(lambda o: o.n == n)

    def select_l(self, l):
        return self.select(lambda o: o.l == l)

    def select_nl(self, n, l):
        return self.select(lambda o: o.n == n and o.l == l)

    # constructing operators
    def build_H0(self):
        energies = [orb.E for orb in self.orbitals]
        return np.diag(energies)

class Operators:
    pass

# checking the normalization of the orbital (ChatGPT)
# norm_check = Orbital(n=1,l=0,m=0)

# def radial_norm(orb, r_max=200.0):
#     integrand = lambda r: (np.abs(orb.R_nlm(r))**2) * (r**2)
#     val, err = integrate.quad(integrand, 0.0, r_max, limit=200)
#     return val, err

# val, err = radial_norm(norm_check, r_max=100.0)
# print(val, err) 

# for r_max in [50, 100, 200, 400]:
#     val, err = radial_norm(norm_check, r_max=r_max)
#     print(r_max, val)

# def angular_norm(orb):
#     integrand = lambda phi, theta: (np.abs(orb.Y_lm(theta, phi))**2) * np.sin(theta)
#     val, err = integrate.dblquad( # idk why this says an error
#         integrand,
#         0.0, np.pi,          # theta bounds
#         lambda theta: 0.0,
#         lambda theta: 2*np.pi
#     )
#     return val, err

# val, err = angular_norm(norm_check)
# print(val, err) 

## ! orbital orthogonality functions (ChatGPT)
# def radial_overlap(orb1, orb2, r_max=200.0):
#     integrand = lambda r: (
#         np.conj(orb1.R_nlm(r)) *
#         orb2.R_nlm(r) *
#         r**2
#     )
#     val, err = integrate.quad(integrand, 0.0, r_max, limit=200)
#     return val, err

# def angular_overlap(orb1, orb2):
#     integrand = lambda phi, theta: (
#         np.conj(orb1.Y_lm(theta, phi)) *
#         orb2.Y_lm(theta, phi) *
#         np.sin(theta)
#     )
#     val, err = integrate.dblquad(
#         integrand,
#         0.0, np.pi,
#         lambda theta: 0.0,
#         lambda theta: 2*np.pi
#     )
#     return val, err

# s1 = Orbital(1,0,0) # 1s
# s2 = Orbital(2,0,0) # 2s

# p2a = Orbital(2,1,0) # 2p y?
# p2b = Orbital(2,1,-1) # 2p x?

# print(radial_overlap(s1,s2))
# print(angular_overlap(p2a,p2b))

## ! creating the basis 

basis = Basis(make_hydrogen_orbitals(nmax=2))

## ! creating the hamiltonian

FieldFreeHam = basis.build_H0()

## ! Hamiltonian time evolution (ChatGPT)

# * r3 overlap functions

def _radial_r3_overlap(orb_i, orb_j, r_max=400.0, quad_limit=300):
    """
    I_r = ∫_0^∞ R_i(r) R_j(r) r^3 dr  (atomic units)
    Approximates ∞ with r_max; increase r_max for higher-n states.
    """
    def integrand(r):
        return np.conj(orb_i.R_nlm(r)) * orb_j.R_nlm(r) * (r**3)

    val, err = integrate.quad(integrand, 0.0, r_max, limit=quad_limit)
    return val

def _angular_overlap(orb_i, orb_j, f_theta_phi, dblquad_epsabs=1e-10, dblquad_epsrel=1e-10):
    """
    I_ang = ∫ Y_i*(θ,φ) f(θ,φ) Y_j(θ,φ) dΩ
          = ∫_0^{2π} ∫_0^π [ ... ] sinθ dθ dφ
    """
    def integrand(phi, theta):
        return (np.conj(orb_i.Y_lm(theta, phi)) *
                f_theta_phi(theta, phi) *
                orb_j.Y_lm(theta, phi) *
                np.sin(theta))

    # dblquad integrates in order: inner variable = phi, outer = theta
    val, err = integrate.dblquad(
        integrand,
        0.0, np.pi,                 # theta bounds
        lambda theta: 0.0,
        lambda theta: 2.0*np.pi,
        epsabs=dblquad_epsabs,
        epsrel=dblquad_epsrel
    )
    return val

# * dipole matrices

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
    basis = list(basis)
    N = len(basis)

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
                Ir = _radial_r3_overlap(basis[i], basis[j], r_max=r_max, quad_limit=quad_limit)
                radial_cache[key] = Ir
            else:
                Ir = radial_cache[key]

            Iax = _angular_overlap(basis[i], basis[j], fx, dblquad_epsabs=epsabs, dblquad_epsrel=epsrel)
            Iay = _angular_overlap(basis[i], basis[j], fy, dblquad_epsabs=epsabs, dblquad_epsrel=epsrel)
            Iaz = _angular_overlap(basis[i], basis[j], fz, dblquad_epsabs=epsabs, dblquad_epsrel=epsrel)

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

# * complete dipole matrix

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

# ? defining the polarization and computing the dipole polarization matrix
pol = np.array([0.,0.,1.0],dtype=np.complex128)
dipoles_xyz = (compute_dipole_matrices_xyz(basis=basis.orbitals))
dipole_matrix = dipole_matrix_for_polarization(xyz=dipoles_xyz,polarization=pol)

# * H(t)

E0 = 0.01
omega = 0.375
N_cycles = 15
T = N_cycles * 2 * np.pi / omega
t0 = 300.0
phi = 0.0

def compute_hamiltonian(t, H0=FieldFreeHam, D=dipole_matrix, E0=E0, omega=omega, T=T, t0=t0, phi=phi):
    if np.abs(t-t0) <= T/2:
        envelope = np.sin(np.pi*(t-(t0-T/2))/T)**2
    else:
        envelope = 0.

    elec_field = E0 * envelope * np.cos(omega*(t-t0)+phi)

    hamiltonian = H0 - elec_field * D

    return hamiltonian

## ! coefficient vector and time evolution
# initial state
initial_coeff = np.array([1.0,0.,0.,0.,0.],dtype=np.complex128) # example, where only the 1s orbital is pictured
# initial_coeff = np.array([1/np.sqrt(2),0.,1/np.sqrt(2),0.],dtype=np.complex128)
# x0 = np.real(initial_coeff)
# y0 = np.imag(initial_coeff)
# z0 = np.concatenate([x0,y0])

# # evolution
# def coeff_evolution(t, c0, H):
#     # c = x + iy
#     N = int(len(c0)/2)
#     x0 = c0[:N]
#     y0 = c0[N:]


#     dxdt = H @ y0
#     dydt = -H @ x0

#     return np.concatenate([dxdt,dydt])

# ## time scale 
# t_span = (0,100) # time in atomic units

# sol = integrate.solve_ivp(fun=coeff_evolution,t_span=t_span,y0=z0,t_eval=np.linspace(0,100,101),args=(H,),method='DOP853')
# print(sol.y)

# N = int(len(sol.y)/2)
# for i in range(len(sol.y[0])):
#     x = sol.y[:N, i]
#     y = sol.y[N:, i]
#     print(norm(x**2+y**2)) NOT USING SOLVE_IVP METHOD

def crank_nicolson_step(c, t, dt, H_of_t): # replace H with H(t)
    t_mid = t + 0.5*dt

    Hmid = H_of_t(t_mid)  # NxN complex/real Hermitian matrix

    I = np.eye(Hmid.shape[0], dtype=np.complex128)
    A = I + 0.5j*dt*Hmid
    B = I - 0.5j*dt*Hmid

    rhs = B @ c
    c_next = solve(A, rhs, assume_a='gen')  # small N -> fine

    return c_next # [c_0, c_1, c_2, c_3] (coefficient for each orbital in the basis)

dt = 0.15 # atomic units
t_min = 0
t_max = 600
ts = np.linspace(t_min,t_max,int((t_max-t_min)/dt)+1)

def FindCoeffs(c0, t_array, dt, H_of_t):

    c_array = np.empty(shape=(len(ts),len(initial_coeff)),dtype=np.complex128)
    c_array[0] = c0

    for step in range(1,len(t_array)):
        c_array[step] = crank_nicolson_step(c_array[step-1],t_array[step],dt,H_of_t)


    return c_array

FinalCoeff = FindCoeffs(initial_coeff,ts,dt,compute_hamiltonian)

# * coefficient checks
norms = np.array([norm(FinalCoeff[i]) for i in range(len(FinalCoeff))]) # checking normalization

pops = np.abs(FinalCoeff)**2 # Checking populations

Eexp = np.array([np.vdot(FinalCoeff[c_in], compute_hamiltonian(c_in*dt) @ FinalCoeff[c_in]).real for c_in in range(len(FinalCoeff))]) # checking energy expectation

def psi_eval(basis, coeffs, step, Rmax=15*a0, N=60):
    # 3D grid
    x = np.linspace(-Rmax, Rmax, N)
    y = np.linspace(-Rmax, Rmax, N)
    z = np.linspace(-Rmax, Rmax, N)
    xx, yy, zz = np.meshgrid(x, y, z)

    r = np.sqrt(xx**2 + yy**2 + zz**2)
    theta = np.arccos(np.divide(zz, r, out=np.zeros_like(r), where=r!=0))
    phi = np.arctan2(yy, xx)

    # psi = sum(coefficient * basisfunc)
    # evaluate the basisfunc across all of space (CONSTANT), then multiply the result by the respective coefficient, then sum 

    orbs = np.array([orb.psi_nlm(r, theta, phi) for orb in basis])

    vals = np.array([orbs[i]*coeffs[step][i] for i in range(len(basis))]) # 

    final_psi = sum(vals)

    pass

## ! relaxation implementation



## ! Saving results (ChatGPT)

# from pathlib import Path
# from datetime import datetime
# import shutil
# import json

# # * Create a run directory
# def create_run_dir(base_dir="runs", make_latest=True):
#     base = "excited_hydrogen_project" / Path(base_dir)
#     base.mkdir(exist_ok=True)

#     timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
#     run_dir = base / timestamp
#     run_dir.mkdir()

#     if make_latest:
#         latest = base / "latest"
#         if latest.exists():
#             shutil.rmtree(latest)
#         latest.mkdir()

#     return run_dir, (base / "latest" if make_latest else None)

# run_dir, latest_dir = create_run_dir()

# # * Save results (arrays)

# def save_results_npz(run_dir, **arrays):
#     np.savez(run_dir / "results.npz", **arrays)

# save_results_npz(
#     run_dir,
#     t=ts,
#     C=FinalCoeff,
#     populations=pops,
#     energy=Eexp,
#     norm=norms
# )

# # * json summary

# summary = {
#     "pulse": {
#         "omega": omega,
#         "T": T,
#         "t0": t0,
#         "E0": E0
#     },
#     "time_step": dt,
#     "basis": [orb.orb_to_string() for orb in basis.orbitals],
#     "checks": {
#         "norm_max_dev": float(np.max(np.abs(norms - 1))),
#         "energy_max_dev": float(np.max(Eexp) - np.min(Eexp)),
#         "final_populations": pops[-1].tolist()
#     }
# }

# def save_summary_json(run_dir, summary_dict):
#     with open(run_dir / "summary.json", "w") as f:
#         json.dump(summary_dict, f, indent=2)

# save_summary_json(run_dir,summary)

# # * update latest

# def update_latest(run_dir, latest_dir):
#     for fname in ["results.npz", "summary.json"]:
#         src = run_dir / fname
#         dst = latest_dir / fname
#         shutil.copy2(src, dst)

# update_latest(run_dir, latest_dir)