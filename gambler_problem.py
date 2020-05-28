# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:36:29 2020

Implementation of Gambler's Problem from Chapter 4 of the 
Sutton & Barto book Reinforcement Learning An Introduction.

TODO:
- Implement the problem in a class (make it easier to read)

Aims:
- Investigate the optimal solutions to the problem under different conditions
- Understand why the optimal action policy takes the form that it does

@author: William Bankes
"""

#%%
#Imports

import numpy as np
import matplotlib.pyplot as plt
import random
    
#%%
#Write a really bare bones implementation for the gambler's problem:
#n.b. indexing from 1 instead of 0 because it makes the problem nicer
        
final_state = 99
reward = 1
p_h = 0.4
epochs = 1

states = [x for x in range(0, final_state)]
policy = [0 for x in range(0, final_state)]

vs = [0 for x in range(0, final_state)]

# The final state isn't in the states: draw the MDP for this problem

def create_win_loss_states(actions, state, final_state):
    """
    create win loss values creates the potential win and loss states for all
    actions taken from a certain state. Bounds max or min values
    """
    output = list()
    
    for a in actions:
        
        win = state + a
        loss = state - a 
            
        output.append((win, loss))
        
    return output

def bellman_equ(vs, win_state, loss_state, final_state, reward):
    """
    Output the results of the bellman_equ for a win_loss state pair. The final
    state and reward can also be edited as they affect the returned value.
    """

    if win_state == final_state:       
     
        win = p_h * (reward)
   
    else:
        win = p_h * vs[win_state]
             
    loss = (1 - p_h) * vs[loss_state]
     
    return win + loss
    
    
def max_vs(state, vs, reward, p_h, final_state):
    #find the maximum value_function for state s
    #max_a Sum p(s', r | s,a)[r + gammaV(s')] ... as the problem is episodic we can ignore gamma....
    
    #Possible actions:
    max_action = min([state, final_state - state])
    
    actions = [x for x in range(0, max_action + 1)]
     
    win_loss_states = create_win_loss_states(actions, state, final_state)
             
    bellman_res = [bellman_equ(vs, win, loss, final_state, reward)\
                for win, loss in win_loss_states]
    
    return max(bellman_res), bellman_res.index(max(bellman_res))



epochs = 10

#for e in range(epochs):

theta = 1000


while theta > 0.001: 
       
    theta = 0
    
    for s in states:
        
        vs_old = vs[s]
        
        vs[s], policy[s] = max_vs(s, vs, reward, p_h, final_state)
        
        theta = max([theta, abs(vs[s] - vs_old)])
        

#%%

fig, axes = plt.subplots(nrows=2, ncols=1)

axes[0].plot(vs)
axes[1].scatter(states, policy)

#Whilst the policy shown in the book is optimal, ties for actions in max value func mean that multiple less pretty functions exist.
        
        
    



        
    

            
        
    
    