# Reinforcement-learning-in-financial-applications

## Project Syllabus

Instructor: Prof. Chak WONG

Group Members: HE Xinyi & ZHU Zheng

Contact: xhebm@connect.ust.hk & zheng.zhu@connect.ust.hk

Aim: Reinforcement Learning in the financial applications

### Assignments

1. Consider the discrete-time asset allocation example in section 8.4 of Rao and Jelvis.  Suppose the single-time-step return of the risky asset from time t to t+1 as Y_t=a, prob=p,  and b, prob=(1−p) .  Suppose that T =10, use the TD method to find the Q function, and hence the optimal strategy.

2. In a binomial model of a single stock with non-zero interest rate, assume that we can hedge any fraction of a stock, use policy gradient to train the optimal policy of hedging an ATM American put option with maturity T = 10.  When do you early exercise the option? Is your solution same as what you obtain from delta hedging? 
Optional, as a substitute for 3) For the really advanced students:  suppose the stock follows a GBM, construct an algorithm to train a NN that hedges an ATM American put option.  
Optional bonus:  after solving the above optional question,  use the Soft Actor Critic algorithm in TF agent to solve the problem again in the colab.research.google.com environment.   Compare your results. 


### Textbooks

Rao, “Foundations of Reinforcement Learning with Applications in Finance”, manuscripts, https://stanford.edu/~ashlearn/


