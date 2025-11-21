import numpy as np
import math

import matplotlib.pyplot as plt
from scipy.special import sph_harm, genlaguerre
from scipy.constants import physical_constants
import plotly.graph_objects as go

a0 = physical_constants['Bohr radius'][0]

# Radial hydrogen wavefunction
def R_nlm(n, l, r):
    rho = 2 * r / (n * a0)
    # Associated Laguerre polynomial
    L = genlaguerre(n-l-1, 2*l+1)(rho)

    # Normalization constant
    num = 2.0 / (n**2 * a0)
    pref = num**1.5 * np.sqrt(math.factorial(n-l-1) / (2*n*math.factorial(n+l)))

    return pref * np.exp(-rho/2) * rho**l * L


# Angular part (Y_l^m)
# represents the angular component of the wavefunction in terms of theta and phi
def Y_lm(l, m, theta, phi):
    return sph_harm(m, l, phi, theta)


# Full separated wavefunction ψ = R(r) Y(θ,φ)
def psi_nlm(n, l, m, r, theta, phi):
    return R_nlm(n,l,r) * Y_lm(l,m,theta,phi)

def plot_orbital_3d(n, l, m, Rmax=15*a0, N=60, iso=0.1):
    # 3D grid
    x = np.linspace(-Rmax, Rmax, N)
    # y = np.linspace(-Rmax, Rmax, N)
    y = np.array([-Rmax,0,Rmax])
    z = np.linspace(-Rmax, Rmax, N)
    xx, yy, zz = np.meshgrid(x, y, z)

    r = np.sqrt(xx**2 + yy**2 + zz**2)
    theta = np.arccos(np.divide(zz, r, out=np.zeros_like(r), where=r!=0))
    phi = np.arctan2(yy, xx)

    psi = psi_nlm(n, l, m, r, theta, phi)
    val = np.abs(psi)**2

    fig = go.Figure(data=go.Isosurface(
        x=xx.flatten(),
        y=yy.flatten(),
        z=zz.flatten(),
        value=val.flatten(),
        isomin=iso*np.max(val),
        isomax=np.max(val),
        opacity=0.5,
        surface_count=5,
        colorscale="Viridis",
        cmin = 10 ** 25
    ))

    fig.update_layout(
        title=f"Hydrogen Orbital n={n}, l={l}, m={m}",
        scene=dict(aspectmode='data')
    )

    fig.show()

# Example: 2p orbital
plot_orbital_3d(3,0,0, N=100, Rmax = 20 * a0, iso=0.003)