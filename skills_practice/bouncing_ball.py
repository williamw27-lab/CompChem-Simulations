### Create a plot of a bouncing ball's height over time using matplotlib

'''
Import necessary libraries
Set initial conditions
Set time array
Loop to calculate height at each time step and store results
Make the plot
'''

## Import statements
import numpy as np
import matplotlib.pyplot as plt

## Initial conditions
y = 10.0 # initial height (m)
v = 0.0 # initial velocity (m/s)
g = -9.81 # gravitational acceleration (m/s^2)
e = 0.8 # elasticity coefficient
dt = 0.1 # time resolution (s)

## Time array
t_i = 0.0 # initial time, beginning at 0
t_f = 10.0 # total time (s)
time_array = np.linspace(t_i,t_f,int(t_f/dt)) # makes an array with t_max/dt terms between initial and final time

## Loop to calculate height at each time step and store results
heights = np.empty(int(t_f/dt),dtype=np.float64) # makes an empty array with elements for each time step, making sure the data type is float
for time_step in range(np.size(heights)):
    v = v + g*dt # change of velocity with time
    y = y + v*dt # change of height with time
    if y <= 0: # when the ball bounces
        v = -v * e # change the velocity direction and lose some energy
        y = 0 # correct the height to 0 (y >= 0)

    heights[time_step] = round(y,4)

## Plot the height vs. time graph
fig, ax = plt.subplots() # allow multiple subplots (with the animatmion later)
height_vs_time = ax.plot(time_array,heights,label='Height vs. Time graph') # set the x and y axes, respectively using time and height data, respectively
ax.set_xlabel('Time (s)') # label the x axis
ax.set_ylabel('Height (m)') # label the y axis
ax.set_title('Height (m) vs. Time (s)') # title the plot

## Making an animation using animation.FuncAnimation
import matplotlib.animation as animation

path_animated = ax.scatter(time_array[0],heights[0],s=5) # makes a scatter plot for the animation, with initial frame at time step 0
ball_animated = ax.scatter(time_array[0],heights[0],s=10,c='r') # animates the ball as it travels 

def update(frame):
    # for each frame, update the data stored
    x = time_array[:frame] # an array with frame # of time terms
    h = heights[:frame] # an array with frame # of height terms
    # update the scatter 
    data = np.stack([x,h]).T # an array with frame # of data points in the form (time,height), dimensions (frame,2)
    path_animated.set_offsets(data) # updates the scatter plot with the data points

    # update different scatter
    x2 = time_array[frame]
    h2 = heights[frame]

    ball_animated.set_offsets([x2,h2])

    return (path_animated,ball_animated) # comma must be included to show tuple

path_animation = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=100)

'''
animation.FuncAnimation
Takes a number of frames and the interval between each frame (in ms)
For each frame (1, 2, ..., n), uses the update function to update the points plotted from each frame
frames alters the total time
interval alters the framerate -- question: how does a difference between the interval and dt affect the animation?
'''

## show the plots
plt.show()