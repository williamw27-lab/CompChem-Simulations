### Create a program to model a mass on a spring

'''
Import necessary libraries
Set initial conditions
Make time array
Solve the differential equation and graph the results
    1. use a loop over a time array
    2. scipy.integrate.solve_ivp
Make plots and/or animations
'''

## Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

## Set initial conditions
m = 5.0 # mass (kg)
k = 5.0 # spring constant (N/m)
x = 0.1 # displacement from equilibrium (m)
v = 0. # initial velocity (m/s)

# define differential equations (for solve_ivp method)

## time array
t_i = 0. # initial time = 0
t_f = 10.0 # final time = 10 (s)
dt = 0.1 # time resolution (s)
time_array = np.linspace(t_i+dt,t_f,int((t_f-t_i)/dt)) # makes the time array

## Solve the differential equation
displacements_loop = np.empty(len(time_array),dtype=np.float64) # creates an empty array to store the displacements over time

# using loops

for time_step in range(len(time_array)):
    displacements_loop[time_step] = x

    force = -k*x # defines the force based on the spring constant and displacement
    a = force/m # defines the acceleration based on the force and mass  
    v = v + a*dt # updates the velocity 
    x = x + v*dt # updates the position

# making plot and animation

fig1, axs1 = plt.subplots(1,2)
displacement_vs_time = axs1[0].plot(time_array,displacements_loop,c='r')


mass_on_spring = displacements_loop + 5.0 # plots position of a mass, rather than displacement from equilibrium, as if the spring was 5.0 m long
spring_anim = mass_on_spring 

mass_animated = axs1[1].scatter(time_array[0],mass_on_spring[0],c='r',s=10)
# spring_animated = axs1[1].plot(time_array[0],mass_on_spring[0])

def update(frame):
    time_step = time_array[0]
    mass_pos = mass_on_spring[frame]
    # spring_len = spring_anim[frame]

    mass_data = [time_step,mass_pos]
    # spring_data = [time_step,spring_len]

    mass_animated.set_offsets(mass_data)
    # spring_animated.set_offsets(spring_data)

    return (mass_animated,)

spring_plot = animation.FuncAnimation(fig=fig1,func=update,frames=len(displacements_loop),interval=100)
plt.show()

# using scipy.integrate.solve_ivp