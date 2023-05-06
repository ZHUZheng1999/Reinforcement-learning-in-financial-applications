# import pandas as pd
import numpy as np
# from utils import put_option_price
from scipy.stats import norm

class Binomial_stock(object):

    def __init__(self, S0, K, rf, sigma, n_step):
        self.S0 = S0
        self.K = K
        self.rf = rf
        self.sigma = sigma
        self.n_step = n_step
        self.trading_days_per_year = 252
        self.R = np.exp(rf/self.trading_days_per_year)
        self.u = np.exp(sigma * np.sqrt(1/self.trading_days_per_year))
        self.d = 1/self.u
        self.prob = (self.R - self.d)/(self.u - self.d)

        self.stock_price = np.zeros((2*self.n_step+1, self.n_step+1))
        self.option_price = np.zeros((2*self.n_step+1, self.n_step+1))

        # stock price
        for j in range((self.n_step+1)):
            for i in range(self.n_step-j, self.n_step+1+j, 2):
                if i < self.n_step:
                    up_times = self.n_step - i + (j - (self.n_step - i))/2
                else:
                    up_times = (j + (self.n_step - i))/2
                down_times = j - up_times
                price = self.S0 * self.u ** up_times * self.d ** down_times
                self.stock_price[i, j] = price

        # payoff at maturity
        for i in range(0, 2*self.n_step+1, 2):
            self.option_price[i, self.n_step] = np.maximum(self.K - self.stock_price[i, self.n_step], 0)
        # option price
        for j in reversed(range(self.n_step)):
            for i in range(self.n_step - j, self.n_step+1+ j, 2):
                up_option_price = self.option_price[i-1, j+1]
                down_option_price = self.option_price[i+1, j+1]
                option_price = (self.prob*up_option_price+(1-self.prob)*down_option_price)/self.R
                self.option_price[i, j] = option_price
    
    def reset(self):
        state = np.array([0, self.S0])
        # state = np.array([0, self.n_step])
        return state

    def true_delta(self):
        S0 = self.S0
        fu = self.option_price[self.n_step+1, 1]
        fd = self.option_price[self.n_step-1, 1]
        delta = (fd-fu) / (S0 * self.u - S0 * self.d)
        return delta


    def step(self, action, state):
        proba = np.random.binomial(1, self.prob)
        this_stock_price = state[1]
        pos = np.argwhere(self.stock_price[:, int(state[0])] == state[1])[0, 0]
        # position = [pos, state[0]]
        this_option_price = self.option_price[pos, int(state[0])]
        # this_stock_price = self.stock_price[state[1], state[0]]
        # this_option_price = self.option_price[state[1], state[0]]
        if proba == 0:
            next_stock_price = self.stock_price[pos+1, int(state[0])+1]
            next_state = np.array([int(state[0])+1, next_stock_price])
        else:
            next_stock_price = self.stock_price[pos-1, int(state[0])+1]
            next_state = np.array([int(state[0])+1, next_stock_price])
        # delta = self.action_map[int(action)]
        # delta = ac .tion
        next_pos = np.argwhere(self.stock_price[:, int(next_state[0])] == next_state[1])[0, 0]
        next_option_price = self.option_price[next_pos, int(next_state[0])]
        reward = next_option_price - this_option_price + action * (next_stock_price - this_stock_price)
        # reward = self.option_price[next_state[1], next_state[0]] - action / 2 * self.stock_price[next_state[1], next_state[0]]
        reward = -np.abs(reward)
        
        if state[0] + 1 == self.n_step:
            done = True
        else:
            done = False
        # next_state = next_state.tolist()
        return next_state, reward, done


class GBM_stock(object):

    def __init__(self, S0, K, rf, sigma, n_step):
        self.S = S0
        self.K = K
        self.rf = rf
        self.trading_days_per_year = 252
        self.dt = 1 / self.trading_days_per_year
        self.T = n_step / self.trading_days_per_year
        self.sigma = sigma
        self.n_step = n_step
        # self.actions_num = 100
        # self.action_map = {i: i / self.actions_num for i in range(self.actions_num + 1)}

    def put_option_price(self, S, K, r, sigma, tau):
        if tau == 0:
            return max(K - S, 0)
        else:
            d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * tau) / (sigma * np.sqrt(tau))
            d2 = d1 - sigma * np.sqrt(tau)
        return float(-S * norm.cdf(-d1) + K * np.exp(-r * tau) * norm.cdf(-d2))

    def reset(self):
        state = [self.T, self.S]
        return state

    def put_delta(self, S, K, r, sigma, tau):
        if tau == 0:
            return float(np.where(K >= S, 1, 0))
        else:
            d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * tau) / (sigma * np.sqrt(tau))
        return float(-norm.cdf(-d1))

    def step(self, action):
        this_option = self.put_option_price(self.S, self.K, self.rf, self.sigma, self.T)
        this_stock = self.S
        # delta = self.action_map[int(action)]
        # delta = action
        self.T -= self.dt
        self.S = self.S * np.exp(
            (self.rf - self.sigma ** 2 / 2) * self.dt + self.sigma * np.random.normal() * np.sqrt(self.dt))
        next_option = self.put_option_price(self.S, self.K, self.rf, self.sigma, self.T)
        reward = next_option - this_option + action * (self.S - this_stock)
        reward = -abs(reward)
        next_state = [self.T, self.S]

        if self.T <= 0 or np.isclose(self.T, 0):
            done = True
        else:
            done = False

        return next_state, reward, done


# Binomial_stock = Binomial_stock(S0=50, K=50, rf=0.05, sigma=0.05, n_step=5)
# print(Binomial_stock.stock_price)
# print(Binomial_stock.option_price)
# t_delta = []
# for i in range(400,600):
#     S = i/10
#     env = Binomial_stock(S, K=50, rf=0.05, sigma=0.3, n_step=20)
#     # t_delta.append(env.true_delta())
#     print(S, env.true_delta())
