# reinforcement_learning

Primarily exercises and example from Reinforcement Learning an Introduction (Sutton and Barto 2018). I also hope to upload my own projects once I get far enough through the book to start apply the ideas to more interesting games.

## bandit_agent.py 
  - An implementation of the multi-armed bandit problem from Chapter 2. It aims to reproduce the results shown in the book to ensure an understanding of the content.
  - A testbed class runs multiple simulations of the problem and plots of the results evolving dynamically are produced
  - Two different update methods mentioned in the book are re-created
  
  ![image](https://github.com/williambankes/reinforcement_learning/blob/master/figures/bandit_rewards.png?raw=true)
  
## gambler_problem.py 
  - An implementation of the gambler problem from Chapter 4
  - Re-creates the environment and dynamic programming method used to estimate the action value function for each state
  - The extact policy shown in the book is hard to re-create exactly as multiple different optimal policies exist for this setup
  - Ensuring the environment is exactly the same as that specified in the book caught me out
  
 ## monte_carlo_off_policy.py
  - An implementation of the blackjack example touched upon in Chapter 5. 
  - Using weighted and ordinary importance sampling methods to evaluate a set initial state in blackjack
  - The OpenAi gym blackjack environment was edited so that the .reset method allowed one to set the state it was reset to (see blackjack_edited.py)
  - To run ensure blackjack_edited.py is in the current active directory, this can be done by changing the path in the os.chdir() function in the first few lines
  
  ### blackjack_edited.py
  
  - Edited the OpenAI gym environment reset method to accept an initial state of the form: `init_state={'player':(1,2), 'dealer':(1)}`
  
  
