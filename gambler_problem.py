# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:36:29 2020

Implementation of Gambler's Problem from Chapter 4 of the 
Sutton & Barto book Reinforcement Learning An Introduction.

TODO:
- Add an epoch option to the value_iteration so we can see how many iterations convergence takes
- Add an option to specify how many policy steps per value eval we take?

Aims:
- Investigate the optimal solutions to the problem under different conditions
- Understand why the optimal action policy takes the form that it does

@author: William Bankes
"""

#%%
#Imports

import numpy as np
import matplotlib.pyplot as plt

#%%

class gambler():
    
    def __init__(self, ph=0.4, reward=1, final_state=100):
        """
        ph -> probability of heads
        reward -> reward for reaching the win state (final_state)
        final_state -> the final win state of the problem
        """
        
        self.__ph = ph
        self.__r = reward
        self.__final_state = final_state
        self.__vs = [0 for x in range(0, final_state)]
        self.__policy = [0 for x in range(0, final_state)]
        self.__states = [x for x in range(0, final_state)]
        
    def value_iteration(self, epsilon=0.001):
        """
        Use value_iteration to find the optimal value function and policy. 
        """
        
        theta = 1000
        
        while theta > epsilon: 
           
            theta = 0
        
            for s in self.__states:
        
                vs_old = self.__vs[s]
        
                self.__vs[s], self.__policy[s] = self.__max_vs(s)
        
                theta = max([theta, abs(self.__vs[s] - vs_old)])
            
    def __max_vs(self, state):
        """
        Find the maximum expected reward for state s: 
                        max_a Sum p(s', r | s,a)[r + gammaV(s')]
        """
        
        #Possible actions:
        max_action = min([state, self.__final_state - state])
        
        actions = [x for x in range(0, max_action + 1)]
         
        win_loss_states = self.__create_win_loss_states(actions, state)
         
        bellman_res = [self.__bellman_equ(win, loss) for win, loss in win_loss_states]
        
        return max(bellman_res), bellman_res.index(max(bellman_res))


    def __create_win_loss_states(self, actions, state):
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


    def __bellman_equ(self, win_state, loss_state):
        """
        Output the results of the bellman_equ for a win_loss state pair. The final
        state and reward can also be edited as they affect the returned value.
        """
    
        if win_state == self.__final_state:       
         
            win = self.__ph * (self.__r)
       
        else:
            win = self.__ph * self.__vs[win_state]
                 
        loss = (1 - self.__ph) * self.__vs[loss_state]
         
        return win + loss  
    
    def plot_results(self):
        
        fig, axes = plt.subplots(nrows=2, ncols=1)
        
        axes[0].plot(self.__vs)
        axes[1].scatter(self.__states, self.__policy)
        
        
g = gambler()
g.value_iteration()
g.plot_results()
    