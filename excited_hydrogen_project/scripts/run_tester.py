### run with basis coefficient evolution for comparison (NO DISSIPATION)

from __future__ import annotations

import numpy as np

from hydrogen_sim.io import complete_save
from hydrogen_sim.field import make_E_of_t, H_of_t
from hydrogen_sim.liouvillian import compile_E1_spontaneous, build_L0_LE
from hydrogen_sim.steppers import make_E_callable, run_rho, run_c

from hydrogen_sim.config import *  
from hydrogen_sim.basis import make_hydrogen_basis
from hydrogen_sim.operators import Operators  

from hydrogen_sim.projection import GridProjector, SliceGrid

run_c_rho = True

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
    if run_c_rho:
        cfg.relaxation.enabled = False # can only run c when no dissipation
    # cfg.time.dt = 0.05

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

    # ------------------------- TODO: fix organization and labels
    # 7) Time grid
    # -------------------------
    dt = cfg.time.dt
    t = np.arange(cfg.time.t_start, cfg.time.t_end + dt, dt)

    store_every = getattr(cfg.time, "store_every", 10)
    rho_store_every = getattr(cfg.time, 'rho_store_every,', 20)

    # -------------------------
    # 3) Field E(t)
    # -------------------------
    E_array = make_E_of_t(t, cfg.pulse)

    # -------------------------
    # 4) Collapse operators
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
    # 5) Liouvillian decomposition
    # -------------------------
    decomp = build_L0_LE(H0=H0, D=D, L_list=L_list)

    # -------------------------
    # 6) Initial rho
    # -------------------------
    if run_c_rho:
        c_i = np.array([1.0,0.0,0.0,0.0,0.0],dtype=np.complex128)
        rho0 = np.outer(c_i,c_i.conj())
    else:
        rho0 = np.zeros((N, N), dtype=np.complex128)
        rho0[idx_1s, idx_1s] = 1.0

    # -------------------------
    # 8) Run rho evolution
    # -------------------------

    E_of_t = make_E_callable(t,E_array)

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
    # Run coefficient evolution
    # -------------------------
    if run_c_rho:
        coeffs = run_c(
            c0=c_i,
            ts=t,
            dt=dt,
            H_of_t=H_of_t,
            ops=ops,
            E_of_t=E_of_t
        )
    else:
        coeffs = None

    # -------------------------
    # 10) Save outputs
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

    ## comparison checks

    default_grid = SliceGrid() # default = xz plane, 301 x 301 grid, -30 a0 to 30 a0 on both axis
    proj = GridProjector(basis=basis,grid=default_grid)

    n_steps = len(t)
    n_stores = n_steps // store_every + 1
    n_snaps = n_steps // rho_store_every + 1

    idx_stores = np.linspace(0, n_steps-1, n_stores, dtype=int)
    idx_snaps = np.linspace(0, n_steps-1, n_snaps, dtype=int)

    # pops error
    if coeffs is not None:
        p_c = np.abs(np.array([coeffs[idx_store,:] for idx_store in idx_stores]))**2
        p_rho = traj.pops

        pops_error = float(np.max(np.abs(p_c - p_rho)))

    # projection consistency
    if coeffs is not None:
        c_snaps = np.array([coeffs[idx_snap,:] for idx_snap in idx_snaps])
        rho_from_c = np.array([np.outer(c_snaps[i,:],c_snaps[i,:].conj()) for i in range(np.shape(c_snaps)[0])])

        proj_from_c_snaps = np.array([proj.density_from_c(c_snaps[i,:]) for i in range(np.shape(c_snaps)[0])])
        proj_from_rho_from_c = np.array([proj.density_from_rho(rho_from_c[i,:,:]) for i in range(np.shape(rho_from_c)[0])])

        max_proj_error = float(np.max(np.abs(proj_from_c_snaps - proj_from_rho_from_c)))

    ## summary

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
        "checks": {
            "max |Tr-1|": float(np.max(np.abs(arrays["trace"]-1))),
            "max |sum(pops)-1|": float(np.max(np.abs(np.sum(arrays["pops"],axis=1)-1.0))),
            "min pops": float(np.min(arrays["pops"])),
            "purity min": float(np.min(arrays["purity"])),
            "purity max": float(np.max(arrays["purity"])),
            "final pops": list(float(i) for i in arrays["pops"][-1,:].tolist()),
            "herm error max": float(np.max(arrays["herm_error"])),
            "positivity min": float(np.min(arrays["positivity"]))
            },
        "comparisons": {
            "max pops error": pops_error,
            "max proj error": max_proj_error
            },
        "script": "run_tester"
        }


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