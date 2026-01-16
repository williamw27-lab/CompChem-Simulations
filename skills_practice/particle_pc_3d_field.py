### Create a code to model the behavior of a charged particle interacting with a point charge (coulombic force), 3d

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

## describing the initial state

k = 8.99 * 10**9 # Coulomb's constant (Nm^2/C^2)
e = 1.60 * 10**(-19) # elementary charge (C)

q = -1.0*e # charge (C)
m = 9.11 * 10**(-31) # electron mass (kg)
x_i = 2.0 # position in a grid, x axis (m)
y_i = 2.0 # position in a grid, y axis (m)
z_i = 2.0 # position in a grid, z axis (m)
v_i = np.array([1.0,1.0,1.0]) # particle initial velocity as a vector [vx,vy,vz] (m/s)

point_charge = ((0.,0.,0.),1.0*e) # describe the point charge's position and charge

B = np.array([0.,0.,0.]) # strength of magnetic field (?)

## defining the evolution function
def evolution(t, state, r_min, B_field, charge, mass, pc, k):
    '''
    Returns the derivatives of position and velocity for solve_ivp

    :param t: time
    :param state: array holding the position and velocity of the particle
    :param B_field: array describing the vector of the magnetic field
    :param charge: scalar, particle's charge
    :param mass: scalar, particle's mass
    :param pc: tuple, describing the position and charge of point charge
    :param k: scalar, Coulomb's constant
    '''

    x, y, z, vx, vy, vz = state
    point_pos, point_crg = pc

    r = np.array([x,y,z]) - np.array(point_pos) # computes the vector from the particle to the point charge
    d = norm(r)

    if norm(np.array([state[0],state[1],state[2]])) - r_min <= 0.001: # because of classical physics, the two particles collapse during the simulation
        pass # when the collision occurs, terminate

    E_field = k * point_crg * r / d**3
    F = charge * (E_field + np.cross(np.array([vx,vy,vz]), B_field))
    a = F / mass

    return [vx, vy, vz, a[0], a[1], a[2]]

## preparing to use solve_ivp
span = (0, 1) # time interval to evaluate
eval = np.linspace(0,1,101) # points to return
system = (0.001, B, q, m, point_charge, k)
initial_state = [x_i, y_i, z_i, v_i[0], v_i[1], v_i[2]]

## defining a collision function
def stop(t, state, r_min, *args):
    return norm(np.array([state[0],state[1],state[2]])) - r_min

stop.terminal = True
stop.direction = -1

## use solve_ivp
solution = solve_ivp(fun=evolution,t_span=span,y0=initial_state,args=system,t_eval=eval,events=stop)

## plotting
xs, ys, zs, *vs = solution.y


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

proton = ax.scatter([0.],[0.],[0.])
electron = ax.scatter(xs[0],ys[0],zs[0])
velocity = ax.quiver(xs[0],ys[0],zs[0],xs[1]-xs[0],ys[1]-ys[0],zs[1]-zs[0], color='red')

def update(frame):
    x = xs[frame]
    y = ys[frame]
    z = zs[frame]

    # electron._offsets3d = (x,y,z)
    electron._offsets3d = ([x],[y],[z])
    try:
        u = xs[frame + 1]
        v = ys[frame + 1]
        w = zs[frame + 1]
        # print(u, v, w)
        velocity.set_segments([[[x, y, z], [u, v, w]]])
    except IndexError:
        velocity.set_segments([[[x, y, z], [0, 0, 0]]])
    return (electron, velocity,)

anim = animation.FuncAnimation(fig, update,frames=len(solution.t),interval=50)



plt.show()