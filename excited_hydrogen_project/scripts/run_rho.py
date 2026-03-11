### Simulate the excitation and relaxation of a hydrogen atom using light

from __future__ import annotations

import numpy as np

from hydrogen_sim.io import complete_save
from hydrogen_sim.field import make_E_of_t
from hydrogen_sim.liouvillian import compile_E1_spontaneous, build_L0_LE
from hydrogen_sim.steppers import make_E_callable, run_rho

from hydrogen_sim.projection import SliceGrid, GridProjector

from hydrogen_sim.config import *  
from hydrogen_sim.basis import make_hydrogen_basis
from hydrogen_sim.operators import Operators  


def build_config() -> SimulationConfig:
    """
    Create a config for this run.
    """
    return SimulationConfig(BasisConfig(),PulseConfig(),TimeGridConfig(),RelaxationConfig())


def main():
    # -------------------------
    # 0) Config
    # -------------------------
    cfg = build_config()

    # -------------------------
    # 1) Basis
    # -------------------------
    basis = make_hydrogen_basis(cfg.basis.nmax)
    N = basis.N

    idx_1s = basis.numbers_to_index(key=(1,0,0))

    # -------------------------
    # 2) Operators
    # -------------------------
    # Expect ops to expose H0, X, Y, Z (and maybe D)
    ops = Operators(basis,cfg)
    H0 = ops.H0
    X, Y, Z = ops.X, ops.Y, ops.Z

    # Combine polarization -> D
    D = ops.D

    # ------------------------- 
    # 3) Time grid
    # -------------------------
    dt = cfg.time.dt
    t = np.arange(cfg.time.t_start, cfg.time.t_end + dt, dt)

    store_every = getattr(cfg.time, "store_every", 10)
    rho_store_every = getattr(cfg.time, 'rho_store_every,', 20)

    # -------------------------
    # 4) Field E(t)
    # -------------------------
    E_array = make_E_of_t(t, cfg.pulse)
    E_of_t = make_E_callable(t,E_array)

    # -------------------------
    # 5) Collapse operators
    # -------------------------

    if cfg.relaxation.enabled:
        L_list, L_meta = compile_E1_spontaneous(
            basis=basis,
            X=X, Y=Y, Z=Z,
            gamma_scale=cfg.relaxation.gamma_scale,
            min_gamma=0.0,
        )
    else:
        L_list = []


    # -------------------------
    # 6) Liouvillian decomposition
    # -------------------------
    decomp = build_L0_LE(H0=H0, D=D, L_list=L_list)

    # -------------------------
    # 7) Initial rho
    # -------------------------
    rho0 = np.zeros((N, N), dtype=np.complex128)
    rho0[idx_1s, idx_1s] = 1.0 # ! Initial state: pure 1s

    # -------------------------
    # 8) Run rho evolution
    # -------------------------

    traj = run_rho(
        rho0=rho0,
        t=t,
        dt=dt,
        E_of_t=E_of_t,
        decomp=decomp,
        H0=H0,
        store_every=store_every,
        rho_store_every=rho_store_every,
        cleanup=True,
    )

    # -------------------------
    # 9) Save outputs
    # -------------------------

    arrays = {
        "t": traj.t,
        "energy_H0": traj.energy_H0,
        "pops": traj.pops,
        "trace": traj.trace,
        "purity": traj.purity,
        "herm_error": traj.herm_error,
        "t_snaps": traj.t_snaps,
        "rho_snaps": traj.rho_snaps,
        "positivity": traj.min_eig,
    }

    summary = {
        "basis":{
            "orbs": [orb.orb_to_string() for orb in basis.orbitals],
            "nmax": cfg.basis.nmax
            },
        "pulse": {
            "E0": cfg.pulse.E0,
            "omega": cfg.pulse.omega,
            "N_cycles": cfg.pulse.N_cycles,
            "t0": cfg.pulse.t0,
            "phase": cfg.pulse.phase,
            "polarization": cfg.pulse.polarization
            },
        "time": {
            "time_step": cfg.time.dt,
            "t_start": cfg.time.t_start,
            "t_end": cfg.time.t_end
            },
        "relaxation": {
            "enabled": cfg.relaxation.enabled,
            "gamma": cfg.relaxation.gamma_scale
            },
        "checks": { # TODO: add store every values
            "max |Tr-1|": float(np.max(np.abs(arrays["trace"]-1))),
            "max |sum(pops)-1|": float(np.max(np.abs(np.sum(arrays["pops"],axis=1)-1.0))),
            "min pops": float(np.min(arrays["pops"])),
            "purity min": float(np.min(arrays["purity"])),
            "purity max": float(np.max(arrays["purity"])),
            "final pops": list(float(i) for i in arrays["pops"][-1,:].tolist()),
            "herm error max": float(np.max(arrays["herm_error"])),
            "positivity min": float(np.min(arrays["positivity"]))
            },
        "script": "run_rho"
        }

    # if density_snaps is not None:
    #     arrays["density_t"] = density_snaps["t"]
    #     arrays["density_P"] = density_snaps["P"]

    complete_save(
        summary,
        t = arrays["t"],
        energy = arrays["energy_H0"],
        pops = arrays["pops"],
        trace = arrays["trace"],
        purity = arrays["purity"],
        herm_error = arrays["herm_error"],
        t_snaps = arrays["t_snaps"],
        rho_snaps = arrays["rho_snaps"],
        positivity = arrays["positivity"]
    )


if __name__ == "__main__":
    main()