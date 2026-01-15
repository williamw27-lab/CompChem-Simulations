### Analyze the behavior of an electron and stationary proton in an oscillating external electric field

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.integrate import solve_ivp

## describing the initial state

k = 8.99 * 10**9 # Coulomb's constant (Nm^2/C^2)
e = 1.60 * 10**(-19) # elementary charge (C)
# a0 = 53 * 10**(-12) # bohr radius (m)
a0 = 0.01
v0 = 3 # initial speed (m/s)

q = -1.0*e # charge (C)
m = 9.11 * 10**(-31) # electron mass (kg)
x_i = a0 * np.array([1.0,0.,0.]) # initial position in a grid 
v_i = v0 * np.array([np.sqrt(3)/3,np.sqrt(3)/3,np.sqrt(3)/3]) # particle initial velocity as a vector [vx,vy,vz] (m/s)

point_charge = ((0.,0.,0.),1.0*e) # describe the point charge's position and charge

E0 = 10**(-6) * np.array([0.,-1.,1.]) # array describing the oscillating field's amplitude?

## master equation

# t, state, k, E0, E_ion, r_min, pc, w, charge, mass

def evolution(t, state, k, E0, E_ion, r_min, pc, w, charge, mass): 
    '''
    Returns the derivatives of position and velocity for solve_ivp
    
    :param t: time
    :param state: array carrying the position and velocity components of the particle
    :param k: scalar, Coulomb's constant
    :param E0: array describing the oscillating field's components
    :param E_ion: maximum simulation energy (physically an ionized particle)
    :param r_min: minimum distance between proton and electron (cutoff)
    :param pc: tuple describing the point charge (proton)
    :param w: scalar, angular frequency
    :param charge: scalar, particle charge
    :param mass: scalar, particle mass
    '''

    x, y, z, vx, vy, vz = state
    point_pos, point_crg = pc

    r = np.array([x,y,z]) - np.array(point_pos) # computes the vector from the particle to the point charge
    d = norm(r)

    E_coulomb = k * point_crg * r / d**3
    E_osc = E0 * np.cos(w*t)
    
    F = charge * (E_coulomb + E_osc) # B not included, negligible (ChatGPT)
    a = F / mass

    return [vx, vy, vz, a[0], a[1], a[2]]

## break conditions
def ionization(t, state, k, E0, E_ion, r_min, pc, w, charge, mass):
    '''
    Cutoff function when electron energy exceeds a value
    
    :param t: time
    :param state: array carrying the position and velocity components of the particle
    :param k: scalar, Coulomb's constant
    :param E0: array describing the oscillating field's components
    :param E_ion: maximum simulation energy (physically an ionized particle)
    :param r_min: minimum distance between proton and electron (cutoff)
    :param pc: tuple describing the point charge (proton)
    :param w: scalar, angular frequency
    :param charge: scalar, particle charge
    :param mass: scalar, particle mass
    '''

    x, y, z, vx, vy, vz = state
    point_pos, point_crg = pc

    r = np.array([x,y,z]) - np.array(point_pos) 
    d = norm(r)



    KE = 0.5 * mass * (norm(np.array([vx,vy,vz])))**2
    PE = k * point_crg * charge / d

    E = KE + PE

    return E_ion - E

def collision(t, state, k, E0, E_ion, r_min, pc, w, charge, mass):
    '''
    Cutoff function when the distance between electron and proton becomes too small
    
    :param t: time
    :param state: array carrying the position and velocity components of the particle
    :param k: scalar, Coulomb's constant
    :param E0: array describing the oscillating field's components
    :param E_ion: maximum simulation energy (physically an ionized particle)
    :param r_min: minimum distance between proton and electron (cutoff)
    :param pc: tuple describing the point charge (proton)
    :param w: scalar, angular frequency
    :param charge: scalar, particle charge
    :param mass: scalar, particle mass
    '''

    x, y, z, vx, vy, vz = state
    point_pos, point_crg = pc

    r = np.array([x,y,z]) - np.array(point_pos) 
    d = norm(r)

    return d - r_min

ionization.terminal = True
ionization.direction = -1
collision.terminal = True
collision.direction = -1

## preparing to use solve_ivp

system = (k, E0, 2.72*10**(-18), 0.1*a0, point_charge, 3*10**(15), q, m)
initial_state = [x_i[0], x_i[1], x_i[2], v_i[0], v_i[1], v_i[2]]
span = (0, 0.001) # time interval to evaluate
eval = np.linspace(0,0.001,101) # time values to return

## solution
solution = solve_ivp(fun=evolution,t_span=span,y0=initial_state,args=system,t_eval=eval,events=(ionization,collision))
print(solution)

## plotting
# Energy vs time
# distance vs time