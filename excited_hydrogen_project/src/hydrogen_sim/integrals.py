### Contains the radial and angular overlap integrals

import numpy as np
from scipy import integrate

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
