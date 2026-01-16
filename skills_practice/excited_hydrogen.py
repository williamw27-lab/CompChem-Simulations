### Simulate the excitation and relaxation of a hydrogen atom using light

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import sph_harm, genlaguerre
from scipy.constants import physical_constants
from scipy import integrate

## creating a set of basis functions - from "orbital_visualization.py"
a0 = 1

class Orbital:
    def __init__(self,n,l,m): # when I create an object of the orbital class, I need to provide n, l, m, but energy is created as itself
        self.n = n
        self.l = l
        self.m = m
        self.E = -1 / (2 * n**2)
    
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
        return sph_harm(self.m, self.l, phi, theta)


    # Full separated wavefunction ψ = R(r) Y(θ,φ)
    def psi_nlm(self, r, theta, phi):
        return self.R_nlm(r) * self.Y_lm(theta,phi)
    
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

# checking the orthogonality of orbitals (ChatGPT)
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

## creating the basis and 
Basis = [Orbital(1,0,0),Orbital(2,1,-1),Orbital(2,1,0),Orbital(2,1,1)]

## creating the hamiltonian
E_list = [orb.E for orb in Basis]
H = np.diag(E_list)

## creating the coefficent