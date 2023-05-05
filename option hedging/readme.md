We finish the option hedging which the underlying stock follows binomial and GBM in both European option and American option.

For binomial stock, we use DDPG to find the optimal action and compare with the true delta of the option.

For GBM stock, we use DDPG and SAC to find the optimal action and compare with the true delta of the option.

In addition, we analyze the time when exercising early in the American put option for both binomial stock and GBM stock.

The utils.py represents the test function for European option training, and the utils_american.py represents the test function for American option training.

The stock.py represents the binomial stock and GBM stock under European option and the stock_american.py represents the binomial and GBM stock under American option.

The ddpg.py and SAC.py represent the two agent we use in this assignment.

The ipynb files under each folder show the results of different experiments.
