# Project: Hydrogen atom excitation and relaxation
=================================================

This python project contains a package and scripts to model the 
excitation and relaxation of a Hydrogen atom. Created by 
motivated students at Lakeside School in Seattle!

=================================================

## Project information

Hydrogen atom simulation in atomic units using a basis-set approach with:

- [x] hydrogenic orbitals (n,l,m)
- [x] field-free and driven dynamics
- [x] density-matrix evolution with Lindblad relaxation
- [x] dipole coupling in the electric dipole approximation
- [x] 2D visualization 
- [ ] 3D visualization and animation

## Project structure

```text
excited_hydrogen_project/
|
├── pyproject.toml
├── README.md
├── scripts/
│   ├── old_project.py (No longer used)
│   ├── run_rho.py
│   ├── viz_2d.py
│   ├── viz_3d.py
│   ├── animate_3d.py
│   └── run_tester.py
├── src/
│   └── hydrogen_sim/
│       ├── __init__.py
│       ├── basis.py
│       ├── config.py
│       ├── field.py
│       ├── integrals.py
│       ├── io.py
│       ├── liouvillian.py
│       ├── observables.py
│       ├── operators.py
│       ├── projection.py
│       ├── steppers.py
│       └── orbitals.py
└── runs/
```

## Package Installation

0. Clone the repository, in your terminal:
```bash
    git clone https://github.com/williamw27-lab/CompChem-Simulations.git
```
1. Activate a virtual environment in the repository directory
2. Install pip
3. Enter the excited_hydrogen_project/ directory and run:
```bash
    pip install -e .
```
Install the package in *editable* mode

## Package updates

If dependencies, package structure, or scripts changed, run:
```bash
pip install -e .
```

## Running

Run the main density-matrix simulation with:
```bash
python excited_hydrogen_project/scripts/run_rho.py
```

## Outputs

Simulation outputs are written to runs/ and typically include:
- compressed array data in .npz
- run metadata and diagnostics in .json
- a latest/ directory with the most recent run

Typical saved quantities (in .npz) include:
- stored times
- populations
- energy expectation values
- trace
- purity
- rho snapshots

## Dependencies

- numpy >= 2.3.5
- scipy >= 1.17.0
- plotly >= 6.5.0 *plotly dependency not in package yet*

## Visualization

Run the corresponding script:
```bash
python excited_hydrogen_project/scripts/viz_2d.py # matplotlib snapshots
``` 
```bash
python excited_hydrogen_project/scripts/viz_3d.py -h # plotly isosurface snapshots
```
```bash
python excited_hydrogen_project/scripts/animated_3d.py # plotly isosurface animation (WIP)
```