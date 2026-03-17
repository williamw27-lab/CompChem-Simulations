# CompChem-Simulations

This repository is used by students at Lakeside School in Seattle.
We are creating simulation projects to learn about computational
chemistry and practice our coding skills. Thus, this repository
is being used as a learning resource. 

## Repo structure

```text
CompChem-Simulations/
|
├── README.md
├── OrbitalVisualization.ipynb
├── excited_hydrogen_project/
│   ├── README.md
│   ├── pyproject.toml
│   ├── src/
│   ├── scripts/
│   └── runs/
├── skills_practice/
│   ├── bouncing_ball.py 
│   ├── gravity.py
│   ├── oscillating_field.py
│   ├── particle_field.py
│   ├── particle_pc_3d_field.py
│   ├── particle_pc_3d.py
│   ├── random_walk.py
│   └── simple_harmonic_oscillator.py
└── orbital_playground/
    └── orbital_visualization.py
```

## Project descriptions

### Orbital Visualization 

Contained in:
- OrbitalVisualization.ipynb
- orbital_playground/orbital_visualization.py

Our goal of this code was to gain familiarity with building and 
visualizing atomic orbitals. Currently, these are not being updated.
Additionally, they are no longer functioning due to scipy shift
from scipy.special.sph_harm to scipy.special.sph_harm_y.

### Skills Practice

Contained in:
- skills_practice/

Our goal of this code was to gain familiarity with core python
skills we might use while coding, such as visualization, animation,
and evolving differential equations. Currently, new scripts are
not being developed. Many are...incomplete...as well.

### Excited Hydrogen Atom

Contained in:
- excited_hydrogen_project/

Our goal of this code is to gain familiarity with building
larger projects and exploring python functionality. We built a
package and simulated a hydrogen atom using that package. This
project is currently being developed.