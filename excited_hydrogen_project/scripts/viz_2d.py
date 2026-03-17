import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Literal
from hydrogen_sim.projection import SliceGrid, GridProjector
from hydrogen_sim.basis import make_hydrogen_basis, Basis

def plot_density_2d(
        basis, rho, t, 
        plane: Literal['xz', 'yz', 'xy']='xz', 
        extent=60.0, n=301) -> None: 
        # Default values: plane='xz', extent=30.0, n=301.
    '''
    Creates a log10 plot of a 2d snapshot of the function using the rho density

    params: 
        basis: Basis object describing a Hydrogen atom simulation
        rho: rho snapshot
        t: corresponding time snapshot from timescale
        plane: 2d plane used for snapshot (default xz plane)
        extent: half of axis length (a.u.)
        n: plot dimensions (n by n)

    returns:
        Function automatically shows the produced plot (no return)
    '''
    
    grid = SliceGrid(plane=plane, extent=extent, n=n)
    proj = GridProjector(basis, grid)
    P = proj.density_from_rho(rho)
    eps = 1e-18
    img = np.log10(P + eps)

    plt.figure()
    plt.imshow(img, origin="lower",
               extent=(-extent, extent, -extent, extent))
    plt.title(f"log10 density, t={t:.3f}")
    plt.xlabel(grid.plane[0])
    plt.ylabel(grid.plane[1])
    plt.colorbar()
    plt.show() # Figures will show one by one

def load_run_data() -> tuple[Basis, dict[str, np.ndarray]]:
    '''
    Loads necessary data from the latest run to produce plots

    params:
        None

    returns:
        Basis object and dictionary with trimmed t and rho arrays
    '''
    with open("excited_hydrogen_project/runs/latest/summary.json", 'r') as summary:
        summary_info = json.load(summary)
        nmax = summary_info["basis"]["nmax"]

    with np.load("excited_hydrogen_project/runs/latest/results.npz") as data:
        t_snaps = data["t_snaps"]
        rho_snaps = data["rho_snaps"]
        
    basis = make_hydrogen_basis(nmax)

    n_snaps = len(t_snaps)
    step = int(n_snaps / 10) # Total: 10 + 1 = 11 snapshots
    t = t_snaps.copy()[::step]
    rho = rho_snaps.copy()[::step,:,:]

    data_dict = {
        "t": t,
        "rho": rho,
    }
    
    return basis, data_dict

def main():
    basis, data = load_run_data()
    ts = data["t"]
    rhos = data["rho"]

    for snapshot in range(len(ts)):
        plot_density_2d(basis=basis, rho=rhos[snapshot,:,:], t=ts[snapshot])
        # This code creates new grid objects for each plot (not efficient)

if __name__ == "__main__":
    main()