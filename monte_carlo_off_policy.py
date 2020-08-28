# -*- coding: utf-8 -*-
"""
An introduction to Reinforcement Learning: Chapter 5 Monte Carlo methods

An implementation of the off-policy monte carlo simulation e.g. example 5.4 

To run ensure the active directory contains 'blackjack_edited.py'

Goals:
    
    - Use the gym blackjack env to learn the gym rl api
    - Implement the off-policy estimation algorithm with importance sampling
    - Implement the off-policy estimation algorithm with weighted importance
        sampling

Future Ideas:
    
    Update to calculate the action value function q. From there the optimal
    policy for the init_state could be calculated. The importance sampling would
    change slightly to not include the first time step of the trajectory.
    
Update formula:
    
 q = q + ((1/(n))*(reward - q))
    
BLACKJACK

observation:
 The observation of a 3-tuple of: the players current sum,
 the dealer's one showing card (1-10 where 1 is ace),
 and whether or not the player holds a usable ace (0 or 1).
 
 
action:
The player can request additional cards (hit=1) until they decide to stop
(stick=0) or exceed 21 (bust).

Upon taking a step the env returns (observation, reward, done, info) tuple.


reward: 

Created on Sun Aug  2 21:16:36 2020

@author: William Bankes
"""

#imports:
import os
import gym
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

os.chdir('D:/William/Documents/Programming/reinforcement_learning/')

from blackjack_edited import BlackjackEnv #Edited openai-gym blackjack class

#%%

class Policy():
    """
    Parent policy class
    """

    def __init__(self, env):
        """
        env -> openai gym style environment
        """
        
        self.action_space = env.action_space
        
    def action(self, state):
        """        
        Samples a random action from the action space
        
        state -> not used in parent class
        
        returns action sampled from sample space, format depends upon the env 
        """
        
        return self.action_space.sample()
    
class SB_Policy(Policy):
    """
    A policy defined by Sutton and Barto, stick on scores higher than 20 and 
    hit on anything else
    """
    
    def __init__(self, env):
        super().__init__(env)
        
    def action(self, state):        
        """
        If player's hand is >= 20 stick
        
        state -> (tuple) observation tuple from openai gym's blackjack env
        
        return (int) action
        """
        if state[0] >= 20:
            return 0
        else:
            return 1
        
    def get_policy(self, state):
        """
        Give the state (observation tuple) return the probability of the policy
        picking an action
        
        state -> (tuple) observation tuple from openai gym's blackjack env
        
        return (list of doubles) probability of action a is at index a 
        """        
        
        if state[0] >= 20:
            return [1, 0]
        else:
            return [0, 1]
    
    
class Soft_Policy(Policy):
    """
    A soft policy that is used to actually sample the data. The probability of 
    the policy picking an action can be customised
    """
    
    
    def __init__(self, env, prob):

        """
        prob -> (list of doubles) probability of the policy picking action a
                is at index a. Prob must be the same size as the discrete 
                action space and sum to 1. 
        """
        
        super().__init__(env)
        self.prob = prob
        
        #Assertions
        prob_length_error = """length of weights should be equal to the size
        of the discrete action space"""
        prob_sum_error = """Sum of the probabilities should add up to 1"""
        
        assert (len(self.prob) == self.action_space.n), prob_length_error 
        assert (sum(self.prob) == 1), prob_sum_error
        
        
    def action(self, state):
        """
        state -> (tuple) observation tuple from openai gym's blackjack env
        
        return (int) action chosen according to the probabilites in prob
        """
    
        actions = range(0, len(self.prob))
        
        return random.choices(actions, self.prob)[0]
    
    def get_policy(self, state):
        """
        state -> (tuple) observation tuple from openai gym's blackjack env
        
        return (list of doubles) probability of action a is at index a 
        """

        return self.prob


def sample_traj(env, sample_policy, init_state = False):
    
    """
    Sample a trajector from the environment according to the sample policy, 
    init_state can be used to sample from the same start state everytime
    
    env - openai gym environment (edited so the reset method takes init_state)
    
    sample_policy - (Policy) the policy which determines the trajectory
    
    return (list of tuples) (state, action, reward, next_state, done)
    """
    
    done = False
    
    if not init_state:
    
        state = env.reset()
    
    else:
        
        state = env.reset(player_state = init_state['player'],
                          dealer_state = init_state['dealer'])
    
    traj = list()
    
    while not done:
        
        action = sample_policy.action(state)
        
        next_state, reward, done, _ = env.step(action)
    
        traj.append((state, action, reward, next_state, done))
        
        state = next_state
        
    return traj
    

def calc_G(traj):
    
    """
    traj -> (list of tuple) output of the sample_traj function
    
    return (int) sum of the rewards of the trajectory
    """
    
    rewards = list(map(lambda x: x[2], traj))
    
    return sum(rewards)


def calc_ro(traj, sample_policy, target_policy):
    
    """
    Calculate the product of the ratios of the probabilities of a trajectory 
    under the sample and target policies for each time step of the trajectory
    
    traj -> (list of tuples) from sample_traj
    
    sample_policy -> (Policy) the policy via which the trajectory was generated
    
    target_policy -> (Policy) the policy we're trying to evaluate
    
    return (double) the trajectory's weighting    
    """
    
    sample_probs = np.array(list(map(lambda x: sample_policy.get_policy(x[0])[x[1]],
                            traj)))
    
    trgt_probs = np.array(list(map(lambda x: target_policy.get_policy(x[0])[x[1]],
                          traj)))
    
    denom = np.prod(sample_probs)
    
    if denom == 0:
        
        output = 0 
        
    else:
        
        output = np.prod(trgt_probs/sample_probs)
        
    return output


#%%

def importance_sample_eval(env, sample_policy, target_policy, 
                           num_episodes=10_000):
    
    """
    Evaluate the value function for the init_state by sampling 10_000 trajectories
    using the sample policy and applying importance sampling to evaluate the 
    target policy. The value function is dynamically updated.
       
    env - openai gym environment (edited so the reset method takes init_state)
    
    sample_policy -> (Policy) the policy via which the trajectory was generated
    
    target_policy -> (Policy) the policy we're trying to evaluate
    
    num_episodes -> (int) the number of trajectories sampled
    
    return (numpy array) incremental updates of the value function for each 
            trajectory
    """
    
    init_state = {'player':(1,2),
                  'dealer':2}
    
    v_pi = [0]
        
    for i in range(num_episodes):
        
        t = sample_traj(env, sample_policy, init_state = init_state)

        G = calc_G(t)
        
        imp_weighting = calc_ro(t, sample_policy, target_policy)
        
        #Calculate V_pi for that trajectory:
        
        v = imp_weighting * G
                
        v_ = v_pi[-1]
        v_pi.append(v_ + (1/(i + 1))*(v - v_))
              
    return np.array(v_pi[1:])

def weighted_importance_sample_eval(env, sample_policy, target_policy,
                                    num_episodes=10_000):
    
    """
    Evaluate the value function for the init_state by sampling 10_000 trajectories
    using the sample policy and applying weighted importance sampling to evaluate the 
    target policy. The value function is dynamically updated.
       
    env - openai gym environment (edited so the reset method takes init_state)
    
    sample_policy -> (Policy) the policy via which the trajectory was generated
    
    target_policy -> (Policy) the policy we're trying to evaluate
    
    num_episodes -> (int) the number of trajectories sampled
    
    return (numpy array) incremental updates of the value function for each 
            trajectory
    """
        
    init_state = {'player':(1,2),
                  'dealer':2}
    
    v_pi = [0]
    imp = 0
        
    for i in range(num_episodes):
        
        t = sample_traj(env, sample_policy, init_state = init_state)

        G = calc_G(t)
        
        imp_w = calc_ro(t, sample_policy, target_policy)
        
        imp_t_1 = imp
        imp = imp_t_1 + imp_w
        
        v_ = v_pi[-1]
        
        if imp == 0:        
            v_pi.append(0)    
        else:    
            v_pi.append(((imp_w * G)/imp) + (imp_t_1/imp)*v_)
              
    return np.array(v_pi[1:])

def on_policy_estimate(env, target_policy, num_episodes=10_000):
    
    """
    Evaluate the value function of the init_state using an on-policy monte
    carlo method. The value function is updated dynamically 
       
    env - openai gym environment (edited so the reset method takes init_state)
    
    target_policy -> (Policy) the policy we're trying to evaluate
    
    num_episodes -> (int) the number of trajectories sampled
    
    return (numpy array) incremental updates of the value function for each 
            trajectory
    
    on running for 100,000,000 -> -0.27716
    """
    
    init_state = {'player':(1,2),
                  'dealer':2}
    
    v_pi = [0]
    
    for i in range(num_episodes):
        
        t = sample_traj(env, target_policy, init_state=init_state)
        
        G = calc_G(t)
    
        v_ = v_pi[-1]
        
        v_pi.append(v_ + (1/(i + 1))*(G - v_))
        
    return np.array(v_pi[1:])

#%%
#Some experiments:

def ordinary_importance_sampling_variance(env, sample_policy, target_policy):
    """
    Plots the dynamic update of the evaluation for num_of_runs number of evals
    
    *Would be interesting to do for a high number and plot how the variance
    evolves over the evaluation 
    
    """
    
    
    num_of_runs = 10
    
    results = [importance_sample_eval(env, sample_policy, target_policy)\
               for n in range(num_of_runs)]

    x = [i for i in range(0, 10_000)]
    
    fig, axs = plt.subplots()
    
    for n in range(num_of_runs):
        
        axs.plot(x, results[n])
    
    plt.title('Plot of weighted importance sample estimates per iteration')
    plt.ylim([-1, 1])
    plt.xscale('log')
    
    
    
    
def weighted_imp_sampling_variance(env, sample_policy, target_policy):
    """
    See ordinary_importance_sampling_variance
    
    *Interesting to do for a high number and plot both the variance and bias
    of the evaluations (see S&Barto for more info)
    """
    
        
    num_of_runs = 10
    
    results = [weighted_importance_sample_eval(env, sample_policy, target_policy)\
               for n in range(num_of_runs)]

    x = [i for i in range(0, 10_000)]
    
    fig, axs = plt.subplots()
    
    for n in range(num_of_runs):
        
        axs.plot(x, results[n])
    
    plt.title('Plot of weighted importance sample estimates per iteration')
    plt.ylim([-1, 1])
    plt.xscale('log')
    
    
def mean_squared_error(env, sample_policy, target_policy):
    """
    Calculating the actual value using an on-policy MC method, then calculate the
    average mean squared error averaged over 200 different evaluations for 10_000 
    trajectories. Plots average mean squared error over 10_000 elements.
    
    env -> openai gym environment (edited so the reset method takes init_state)
    
    sample_policy -> (Policy) the policy via which the trajectory was generated
    
    target_policy -> (Policy) the policy we're trying to evaluate
    
    return 
    """
    
    
    num_runs = 200
    num_episodes = 10_000
    
    #Using the on_policy estimate find a close approximation of the answer:    
    actual_value = on_policy_estimate(env, target_policy, num_episodes=100_000)[-1]        
        
    #Eval sampling error num_run times:
    ordinary = np.array([np.power(importance_sample_eval(env,
                                      sample_policy,
                                      target_policy) - actual_value, 2)\
                                      for i in range(num_runs)])    
        
    weight = np.array([np.power(weighted_importance_sample_eval(env,
                                             sample_policy,
                                             target_policy) - actual_value, 2)\
                                             for i in range(num_runs)])
        
    #find the mean squared error
    ordinary_mse = np.mean(ordinary, axis=0)
    weight_mse = np.mean(weight, axis=0)   
    
    fig, axs = plt.subplots()
    
    x = [i for i in range(num_episodes)]
    
    axs.plot(x, weight_mse, label='weighted importance sampling')
    axs.plot(x, ordinary_mse, label='ordinary importance sampling')
    
    axs.set_xlabel('iterations')
    axs.set_ylabel('mean squared error')
    
    plt.legend()
    plt.xscale('log')
    plt.ylim([-1, 5])
    
    return  
        
    
#%%
#Lets run some experiments:

if __name__ == '__main__':
    
    env = BlackjackEnv()
        
    sp = Soft_Policy(env, [0.5, 0.5])
    tg = SB_Policy(env)
    
    ordinary_importance_sampling_variance(env, sp, tg)
    weighted_imp_sampling_variance(env, sp, tg)
    output = mean_squared_error(env, sp, tg)
    
    
    
    