### Create a code to model the behavior of a charged particle in a uniform electric field, 2d

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

## describing the initial state
q = 1.0 # charge (C)
m = 1.0 # mass (kg)
x_i = 2.0 # position in a grid, x axis (m)
y_i = 2.0 # position in a grid, y axis (m)
v_i = np.array([-3.0,-3.0,0.]) # particle initial velocity as a vector [vx,vy,vz] (m/s)

## describing the electric (and magnetic) field 
E = np.array([4.0,4.0,0.]) # strength of electric field (N/C)
B = np.array([0.,0.,0.]) # strength of magnetic field (?)

## calculating the force (Lorentz Force)
def LorentzForce(E_field, B_field, velocity, charge):
    '''
    Calculates the Lorentz force acting on a particle
    
    :param E_field: array describing the vector of the electric field
    :param B_field: array describing the vector of the magnetic field
    :param velocity: array describing the particle's velocity
    :param charge: scalar, particle's charge
    '''

    return charge * (E_field + np.cross(velocity, B_field))

## calculating the acceleration
def Acceleration(force, mass):
    '''
    Calculates the acceleration of a particle with a force acting on it
    
    :param force: array describing the force acting on the particle
    :param mass: scalar, particle's mass
    '''
    return force / mass

## defining the differential equation
def evolution(t, state, E_field, B_field, charge, mass):
    '''
    Returns the derivatives of position and velocity given a particle state
    
    :param t: time
    :param state: variable containing the position and velocity of the particle
    '''
    x, y, vx, vy = state

    F = LorentzForce(E_field=E_field, B_field=B_field, velocity=np.array([vx,vy,0.]), charge=charge)
    a = Acceleration(force=F, mass=mass)

    return [vx,vy,a[0],a[1]]

## preparing to use solve_ivp
span = (0, 10) # time interval to evaluate
eval = np.linspace(0,10,101) # points to return 
system = (E, B, q, m)
initial_state = [x_i, y_i, v_i[0], v_i[1]]

## use solve_ivp
solution = solve_ivp(fun=evolution,t_span=span,y0=initial_state,args=system,t_eval=eval)
print(solution.y)