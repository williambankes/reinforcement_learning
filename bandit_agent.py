# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:43:42 2020

@author: William Bankes

Implementation of the k armed bandit testbed described in Chapter 2 of the 
Sutton & Barto book Reinforcement Learning An Introduction. The api is 
based off my so far limited knowledge of the gym python library.

To do:
    
- Optimal Actions passed after a step
- Non-stationary bandit behaviour (exercise 2.5)

"""
#%%
#imports
import numpy as np
import matplotlib.pyplot as plt
import time
import progressbar

#%%

class dis_action_space():
    """
    A discrete action space implementation:
        minimum - min value of the space
        maximum - max value of the space
    To do: A better __str__ for larger spaces
    """
    
    def __init__(self, k):
        self.__k = k
        
    def sample(self):
        return np.random.randint(self.__k)
    
    def get_k(self):
        return self.__k
    
    def __str__(self):
        return "action space: [" +\
            ",".join([str(i) for i in range(self.__k)]) + "]"

class bandit():
    """
    A simple k armed bandit implementation:
        k - number of actions (arms)
        stationary - The rewards for each action remain stationary
        mean - The mean about which the rewards are centred
        std - standard deviation of the normal dist that generates rewards
        
    Todo: implement non-stantionary rewards
    """
    
    def __init__(self, k=10, stationary=True, mean=0, std=1, init=0):
        #Private Variables
        self.__k  = k
        self.__stationary = stationary
        
        if stationary:
            self.__qa_star = np.random.normal(loc=mean, scale=std, size=k)
            self.__qa_star += init
        else:
            self.__qa_star = [init for i in range(k)]
        
        #Public Variables
        self.action_space_ = dis_action_space(k)

    def get_qa(self):
        return ",".join([str(q) for q in self.__qa_star])
        
    def step(self, action):
        if self.__stationary:
            return np.random.normal(loc=self.__qa_star[action], scale=1)
        else:
            #add random values to each step:
            self.__qa_star += np.random.normal(loc=0, scale=0.01, size=self.__k)
            return np.random.normal(loc=self.__qa_star[action], scale=1)

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
        #init count of actions taken:
        self.__n = [0 for i in range(env.action_space_.get_k())]
        
    def __init_q(self, bias, action_space_):
        if not bias:
            return [0 for i in range(action_space_.get_k())]
        else:
            return [bias for i in range(action_space_.get_k())]
        
    def get_q(self):
        return ",".join([str(q) for q in self.__q])        
    
    def action(self, env):
        
        #find action and increment count:
        action = self.__policy(env.action_space_, self.__q)
        self.__n[action] += 1        
        reward = env.step(action)
        self.__q[action] = self.__q_update(reward, self.__q[action], n=self.__n[action])
        
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
        return np.argmax(q)
    
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
        def constant_update(reward, q, **kwargs):
            return q + (alpha * (reward - q))
        
        return constant_update
    
    def create_average_update(self):
        def average_update(reward, q, **kwargs):
            n = kwargs['n']            
            return q + ((1/(n))*(reward - q))
        return average_update

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
    
    def __init__(self, policy, q_update, tests=1000, steps=1000,
                 k=10, stationary=True, mean=0, std=1, bias=False, init=0):
        self.__envs = [bandit(k=k, stationary=stationary,
                              mean=mean, std=std, init=init) for i in range(tests)]
        self.__agents = [agent(policy, q_update, self.__envs[i], bias) for i in range(tests)]
        self.__results = np.zeros(steps)
        
        self.__steps = steps
        self.__tests = tests
        
    def run(self):
        for i in progressbar.progressbar(range(self.__steps)):
            self.__results[i] = np.mean([self.__agents[i].action(self.__envs[i])\
                                        for i in range(self.__tests)])
        return self.__results
    
    
    def debug_estimates(self):
        i = np.random.randint(0, self.__tests)
        
        print(self.__agents[i].get_q())
        print(self.__envs[i].get_qa())
    
    def show(self):
        
        fig, ax = plt.subplots()
        ax.plot(self.__results)
        
        #X label and title
        ax.set_xlabel('Time step')
        ax.set_ylabel('Average Reward')
        ax.set_title('Testbed results')
        plt.show()
        
        return (fig, ax)
    
 
#%%
#General ustility functions:
  
def show_rewards(rewards):
    fig, ax = plt.subplots()
    
    for r in rewards:
        ax.plot(r)
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Average Reward')
    ax.set_title('Testbed results')
                

#%%
#Run the testbed
    
def fig_2_2():

    t = testbed(Policy().create_e_greedy_policy(0.1),
                Update().create_average_update(), tests=2000)
    t1 = testbed(Policy().create_e_greedy_policy(0.01),
                Update().create_average_update(), tests=2000)
    t2 = testbed(Policy().create_e_greedy_policy(0),
                 Update().create_average_update(), tests=2000)
    
    figsize = (10,5)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    line_t, = ax.plot(t.run(), label='e=0.1')
    line_t1, = ax.plot(t1.run(), label='e=0.01')
    line_t2, = ax.plot(t2.run(), label='greedy')
    

    plt.legend([line_t, line_t1, line_t2], ['e=0.1', 'e=0.01', 'greedy'])
    
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Average Reward')
    ax.set_title('Testbed results')
    plt.show()
    
    
fig_2_2()

#%%
"""
Exercise 2.5: qa_star start equal and then take a random walk with std=0.01 per
step. Two runs one with sample averaging the other with a constant update param
a=0.1.
"""

def ex_2_5():
    
    avg = testbed(Policy().create_e_greedy_policy(0.1),
                  Update().create_average_update(), tests=2000, steps=10000, stationary=False)
    const = testbed(Policy().create_e_greedy_policy(0.1),
                    Update().create_constant_update(0.1), tests=2000, steps=10000, stationary=False)
    
    figsize = (10,5)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    line_avg, = ax.plot(avg.run())
    line_const, = ax.plot(const.run())
    
    plt.legend([line_avg, line_const], ['averaging update policy',
               'constant update policy a=0.1'])
           
    ax.set_xlabel('Time step')
    ax.set_ylabel('Average Reward')
    ax.set_title('Testbed results')
    plt.show()        
    
ex_2_5()

#%%
#Parallelisation test:










