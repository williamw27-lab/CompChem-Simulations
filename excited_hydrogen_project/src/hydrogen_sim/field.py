### calculate electric field using sin^2 envelope and calculate hamiltonian 

import numpy as np

def make_E_of_t(t: np.ndarray, pulse_params):
    ts = np.copy(t)
    envelopes = np.zeros(shape=ts.shape)

    for time in range(len(ts)):
        if np.abs(ts[time] - pulse_params.t0) <= pulse_params.T() / 2:
            envelopes[time] = np.sin(np.pi*(ts[time]-(pulse_params.t0 - pulse_params.T() / 2)) / pulse_params.T())**2
        else:
            envelopes[time] = 0.

    elec_field = pulse_params.E0 * envelopes * np.cos(pulse_params.omega*(ts-pulse_params.t0)+pulse_params.phase)

    return elec_field

# def compute_hamiltonian(t, ops, pulse_params):
#     return ops.H0 - _sin2_field(t, pulse_params) * ops.D