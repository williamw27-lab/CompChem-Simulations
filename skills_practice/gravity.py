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
# m1 = 5.0 # mass of object 1 (kg)
# m2 = 5.0 # mass of object 2 (kg)

# x1, y1 = [0.,0.] 
# pos1 = [x1,y1] # position of object 1 (m from reference)

# x2, y2 = [5.0,5.0] 
# pos2 = [x2,y2] # position of object 2 (m from reference)

v1_x = 1.0 # velocity of object 1 in x direction (m/s)
v1_y = 1.0 # velocity of object 1 in y direction (m/s)
v2_x = -1.0 # velocity of object 2 in x direction (m/s)
v2_y = -1.0 # velocity of object 2 in y direction (m/s)

## Differential equation

# problem: the differential equations require the distance between objects, direction between them, and their positions before going to a next iteration
# idea: to use solve ivp, store all of the object information (both objects) in a large array 2 pos, 2 v per  

initial_array = np.array([0., 0., 5.0, 5.0, v1_x, v1_y, v2_x, v2_y])

def master_func(t,master_array):
    '''
    Returns the derivatives of position and velocity given an initial position and velocity array
    
    :param t: time evolution
    :param master_array: array of initial position and velocity values
    '''

    m1 = 5.0
    m2 = 5.0

    # unpack master_array
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = master_array

    # calculate distance between objects
    dx = x2 - x1
    dy = y2 - y1

    d = np.sqrt(dx**2 + dy**2)

    # calculate force on each object
    F = G * m1 * m2 / (d**2)

    Fx1 = F * dx / d
    Fy1 = F * dy / d
    Fx2 = -F * dx / d
    Fy2 = -F * dy / d

    # calculate acceleration on each object
    ax1 = Fx1 / m1
    ay1 = Fy1 / m1

    ax2 = Fx2 / m2
    ay2 = Fy2 / m2

    deriv_array = np.array([vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2])

    return deriv_array

time_array = np.linspace(0,10,100)

anim = solve_ivp(master_func,(0,10),initial_array,t_eval = time_array)
print(anim.y)