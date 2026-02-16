### Orbital class, psi evaluation method

from scipy.special import sph_harm_y, genlaguerre
import numpy as np
import math

from hydrogen_sim.config import AU

a0 = AU.a0

class Orbital:
    def __init__(self,n,l,m): # when I create an object of the orbital class, I need to provide n, l, m, but energy is created as itself
        self.n = n
        self.l = l
        self.m = m
        self.E = -1 / (2 * n**2)
        self._radial_pref = (2.0/(n*a0))**1.5 * np.sqrt(math.factorial(n-l-1) / (2*n*math.factorial(n+l)))

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

        return self._radial_pref * np.exp(-rho/2) * rho**self.l * L


    # Angular part (Y_l^m)
    # represents the angular component of the wavefunction in terms of theta and phi
    def Y_lm(self, theta, phi):
        return sph_harm_y(self.m, self.l, phi, theta)


    # Full separated wavefunction ψ = R(r) Y(θ,φ)
    def psi_nlm(self, r, theta, phi):
        return self.R_nlm(r) * self.Y_lm(theta,phi)