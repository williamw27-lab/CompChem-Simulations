### Create a code to simulate two bodies interacting through gravity in vacuum

'''
Import statements
Initial conditions
Differential equation
Solve diffeq
Plot/animate
'''

## Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.constants import G

## Initial conditions
m1 = 5.0 # mass of object 1 (kg)
m2 = 5.0 # mass of object 2 (kg)
v1_x = 1.0 # velocity of object 1 in x direction (m/s)
v1_y = 1.0 # velocity of object 1 in y direction (m/s)
v2_x = -1.0 # velocity of object 2 in x direction (m/s)
v2_y = -1.0 # velocity of object 2 in y direction (m/s)

print(G)
