import numpy as np
import plotly.graph_objects as go # TODO: Add dependency to package and update README
import json
import argparse

from hydrogen_sim.projection import VolumeGrid, GridProjector3D
from hydrogen_sim.basis import make_hydrogen_basis, Basis

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

def _plot_density_isosurface(
        X,
        Y,
        Z,
        P3d,
        title, 
        frac_min=0.1,
        frac_max=0.5,
        surface_count=3,
        opacity=0.6,
):
    value = P3d.reshape(-1)
    x = X.reshape(-1)
    y = Y.reshape(-1)
    z = Z.reshape(-1)

    vmax = float(np.max(value))
    isomin = frac_min * vmax
    isomax = frac_max * vmax

    fig = go.Figure(
        data=go.Isosurface(
            x=x,
            y=y,
            z=z,
            value=value,
            isomin=isomin,
            isomax=isomax,
            surface_count=surface_count,
            opacity=opacity,
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (a.u.)",
            yaxis_title="y (a.u.)",
            zaxis_title="z (a.u.)",
            aspectmode="cube",
        ),
    )
    return fig

def make_isosurface_fig(
        rho: np.ndarray, 
        t: float, 
        proj: GridProjector3D) -> go.Figure | None:
    '''
    Creates a plotly isosurface plotting electron density from rho at a given time

    params:
        rho: np.ndarray, rho of snapshot
        t: float, time of snapshot
        proj: GridProjector3D, density calculator dependent on basis set

    returns:
        go.Figure object representing the isosurface
    '''

    P3d = proj.density_from_rho(rho)

    title = f'Hydrogen electron density isosurface t = {t:.3f}'

    isosurface_plot = _plot_density_isosurface(
        X=proj.X,
        Y=proj.Y,
        Z=proj.Z,
        P3d=P3d,
        title=title
    ) 

    return isosurface_plot

def main():
    # load data (basis and t/rho snaps)
    basis, data = load_run_data()
    ts = data["t"]
    rhos = data["rho"]

    # Initialize grid and projector
    grid = VolumeGrid()
    proj = GridProjector3D(basis=basis,grid=grid)

    # Argparser to select time
    parser = argparse.ArgumentParser(
                description='Produce a plotly snapshot isosurface at a given t')
    parser.add_argument("t", type=float, choices=[snap_t for snap_t in ts], help='The time t of the snapshot')

    # Takes the inputted time and creates the isosurface
    args = parser.parse_args()
    snapshot_t = args.t

    for snap_idx in range(len(ts)):
        if ts[snap_idx] == snapshot_t:
            rho = rhos[snap_idx]
            break
        else:
            continue

    fig = make_isosurface_fig(rho, snapshot_t, proj)

    # Shows the plot
    if fig is not None:
        fig.show()

if __name__ == '__main__':
    main()