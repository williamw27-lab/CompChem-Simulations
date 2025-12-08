### Create a program to simulate 1D random walkers and their final positions in a histogram

'''
Import necessary libraries
Create a class for the random walker
Create functions for walkers
Plot a simulation of a few random walks
Plot a histogram of many random walks
'''

## Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

## Random walker class

class RandomWalker: # creates the class
    def __init__(self, position=0): # creates the position attribute (default position = 0)
        self.position = position
    
    def random_walk(self): # defines the random walk as a choice to either decrease or increase position by 1
        self.position += random.choice([-1,1])

# test case 

first_walker = RandomWalker() # creates the first walker in the RandomWalker class
first_walker.random_walk() # makes the walker walk
print(first_walker.position) # prints the position of the walker

first_walker.random_walk() # makes the walker walk 
print(first_walker.position) # prints the position of the walker

## defining functions to generate walkers and store their data 

def GenerateWalkers(number):
    '''
    Returns a dictionary with the number of walkers

    :param number: number of walkers
    '''

    # create an empty dictionary to store walkers
    stored_walkers = {}

    # loop to create number of walkers
    for counter in range(1,number+1):
        walker_key = f'walker_{counter}' # creates the key name
        generated_walker = RandomWalker() # creates a walker

        stored_walkers[walker_key] = generated_walker # adds the walker to the dictionary

    # returns the dictionary
    return stored_walkers

def WalkWalkers(walker_dict,steps):
    '''
    Takes a walker dictionary, walks the walkers, stores their position over the number of steps, returns a numpy array of coordinates

    :param walker_dict: dictionary of walkers
    :param steps: steps for each walker
    '''

    position_array = np.empty((len(walker_dict),steps+1),dtype=np.int64) # creates an empty array to store positions of walkers over time (each row = 1 walker, first column is time 0, then each column is after one step)

    for walker_num in range(len(walker_dict)): # iterates over the walker dictionary
        for step in range(1,steps+1): # for each step given
            walker_dict[f'walker_{walker_num+1}'].random_walk() # random walker walks once
            position_array[walker_num,step] = walker_dict[f'walker_{walker_num+1}'].position # sets the position after the step to the new position

    return position_array

def HistWalkers(walker_dict,steps):
    '''
    Creates a number of walkers and stores the final position for each in a returned numpy array
    
    :param walker_dict: dictionary of walkers
    :param steps: steps for each walker
    '''
    
    final_positions = np.empty(len(walker_dict),dtype=np.int64) # creates an empty array with one element per walker (for each final position)

    for walker_num in range(len(walker_dict)): # iterates over the walker dictionary
        for step in range(steps): # for each step, randomly step
            walker_dict[f'walker_{walker_num+1}'].random_walk()

        final_positions[walker_num] = walker_dict[f'walker_{walker_num+1}'].position # updates the final position for the walker

    return final_positions

# test case

first_walker_dict = GenerateWalkers(4)
# print(first_walker_dict)

first_walker_positions = WalkWalkers(first_walker_dict,9)
print(first_walker_positions)

first_walker_finals = HistWalkers(first_walker_dict,100)
print(first_walker_finals)

## matplotlib plotting and animation

# walkers for the animation
anim_walkers = GenerateWalkers(6) # creates 6 walkers
anim_walkers_positions = WalkWalkers(anim_walkers,8) # walks each walker 8 times
positions_by_step = np.swapaxes(anim_walkers_positions,0,1) # organizes the array with the positions of each walker by step #
x_values = np.array([1,2,3,4,5,6])

# walkers for the histogram
hist_walkers = GenerateWalkers(1000) # creates 1000 walkers
hist_walkers_positions = HistWalkers(hist_walkers,100) # walks each walker 100 times

# create the figure and axes objects
fig, axs = plt.subplots(1,2) # creates a figure with two plots side by side

# create the animation
positions_animation = axs[0].scatter(x_values,positions_by_step[0,]) # makes the scatter plot using the animation walkers

def update(frame):
    x = x_values # x values remain the same
    y = positions_by_step[frame] # y values are updated

    data = np.stack([x,y]).T # need to reformat the data into N by 2 array
    positions_animation.set_offsets(data) # updates the scatter plot

    return (positions_animation,)

axs[0].set_xlabel('Walkers') # make the xlabel
axs[0].set_ylabel('Position') # make the ylabel
axs[0].set_xlim(0,7) # set boundaries of x values
axs[0].set_ylim(-8,8) # set boundaries of y values
axs[0].set_title('Animated Walkers')

animated_walkers = animation.FuncAnimation(fig,update,frames=9,interval=1000)

# create the histogram
positions_histogram = axs[1].hist(hist_walkers_positions,bins=50,density=True)

# show the plot
plt.show()