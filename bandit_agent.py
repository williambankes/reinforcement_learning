# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:43:42 2020

@author: William Bankes

Implementation of the k armed bandit testbed described in Chapter 2 of the 
Sutton & Barto book Reinforcement Learning An Introduction. The api is 
based off my so far limited knowledge of the gym python library.
"""
#%%
#imports
import numpy as np
import matplotlib.pyplot as plt

#%%

class dis_action_space():
    """
    A discrete action space implementation:
        minimum - min value of the space
        maximum - max value of the space
    To do: A better __str__ for larger spaces
    """
    
    def __init__(self, minimum, maximum):
        self.__min = minimum
        self.__max = maximum
        
    def sample(self):
        return np.random.randint(self.__min, self.__max)
    
    def get_min(self):
        return self.__min
    
    def get_max(self):
        return self.__max
    
    def __str__(self):
        return "action space: [" +\
            ",".join([str(i) for i in range(self.__min, self.__max)]) + "]"

class bandit():
    """
    A simple k armed bandit implementation:
        k - number of actions (arms)
        stationary - The rewards for each action remain stationary
        mean - The mean about which the rewards are centred
        std - standard deviation of the normal dist that generates rewards
        
    Todo: implement non-stantionary rewards
    """
    
    def __init__(self, k=10, stationary=True, mean=0, std=1):
        #Private Variables
        self.__k  = k
        self.__stationary = stationary
        self.__qa_star = np.random.normal(loc=mean, scale=std, size=k)
        
        #Public Variables
        self.action_space_ = dis_action_space(0, k)
        
        
    def step(self, action):
        if self.__stationary:
            return np.random.normal(loc=self.__qa_star[action], scale=1)
        else:
            return False

#%%
#Agent 

class agent():
    """
    The agent class, given a policy, update method and stores the q action-values
    after interacting with the env (in this case the bandit)
        policy - a policy function which returns an action from the action_space
        q_update - an update function which determines how the action-values are
                    updated after each iteration
        env - the environment the agent will interact with. This is so the  
                action-value space can be determined
    """
    
    def __init__(self, policy, q_update, env, bias=False):
        #private variables:
        self.__policy = policy
        self.__q_update = q_update
        self.__q = self.__init_q(bias, env.action_space_)
        
        
    def __init_q(self, bias, action_space_):
        return [bias for i in range(action_space_.get_min(), action_space_.get_max())]
    
    def action(self, env):
        action = self.__policy(env.action_space_, self.__q)
        reward = env.step(action)
        self.__q[action] = self.__q_update(reward, self.__q[action])
        return reward
    
#%%
#Policy functions
        
class Policy():
    """
    Policy class allows the input arguements to any policy to remain the same
    whilst varying the epsilon for different examples.
    
    Given only an e_greedy_policy is used here this isn't strictly necessary 
    but it was interesting to implement.
    """
    
    def __random_policy(self, action_space):
        return action_space.sample()
    
    def __greedy_policy(self, action_space, q):
        return np.argmax([q])
    
    def create_e_greedy_policy(self, epsilon):
        
        def e_greedy_policy(action_space, q):
        
            i = np.random.uniform(0,1)
            e = epsilon
            if i < e:
                return self.__random_policy(action_space)
            else:
                return self.__greedy_policy(action_space, q)
        
        return e_greedy_policy
    
#%%
#q_updates
    
class Update():
        
    def create_constant_update(self, alpha):
        def constant_update(reward, q):
            return q + (alpha * (reward - q))
        
        return constant_update

#%%
# Bandit agent testbed:
    
class testbed():
    """
    A testbed with <tests> no. of bandits and agents each learning with the same
    policy and q_update functions. 
        policy - policy function for deciding actions 
        q_update - q update function
        tests - the no. of tests to run
        steps - the no. of time steps for each test
        bias - the initial bias of the action-values
    
    Todo: improve the show_rewards and graphic side to allow multiple results
    to be displayed
    """
    
    def __init__(self, policy, q_update, tests=1000, steps=1000, bias=False):
        self.__envs = [bandit() for i in range(tests)]
        self.__agents = [agent(policy, q_update, self.__envs[i], bias) for i in range(tests)]
        self.__results = np.zeros(steps)
        
        self.__steps = steps
        self.__tests = tests
        
    def run(self):
        for i in range(self.__steps):
            self.__results[i] = np.mean([self.__agents[i].action(self.__envs[i])\
                                        for i in range(self.__tests)])
        return self.__results
    
    def show(self):
        
        fig, ax = plt.subplots()
        ax.plot(self.__results)
        
        #X label and title
        ax.set_xlabel('Time step')
        ax.set_ylabel('Average Reward')
        ax.set_title('Testbed results')
        plt.show()
        
        return (fig, ax)
    
    
    def show_rewards(self, rewards):
        fig, ax = plt.subplots()
        
        for r in rewards:
            ax.plot(r)
        
        ax.set_xlabel('Time step')
        ax.set_ylabel('Average Reward')
        ax.set_title('Testbed results')
                
#%%
#Run the testbed

t = testbed(Policy().create_e_greedy_policy(0),
            Update().create_constant_update(0.1), tests=2000)
t1 = testbed(Policy().create_e_greedy_policy(0),
            Update().create_constant_update(0.1), tests=2000)

t1.show_rewards([t.run(), t1.run()])











