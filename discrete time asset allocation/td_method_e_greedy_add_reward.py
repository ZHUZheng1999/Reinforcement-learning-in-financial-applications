import numpy as np
import pandas as pd
from tqdm import trange


class asset_allocation():
    def __init__(self, period, ALPHA=0.5, GAMMA=1, epsilon=0.1, r=0.05, a=1, price=1, W0=10):
        self.ALPHA = ALPHA  # step size of td-method
        self.GAMMA = GAMMA  # discount rate of td-method
        self.r = r  # riskfree asset return
        self.a = a  # 效用函数里的a
        self.price = price  # initial stock price
        self.W0 = W0  # initial asset
        self.actions = [0, 1]  # possible actions: 0: buy 0 unit of stocks, 1: buy 1 unit of stocks
        self.epsilon = epsilon
        self.up = 1.1  # the upside ratio of stock movement in binominal model
        self.down = 0.9  # the downside ratio of stock movement in binominal model
        self.period = period

    def this_action(self, state, price, time, v_value, epsilon=0.1):
        if np.random.rand() < epsilon:
            # exploration
            action = np.random.choice(self.actions)
        else:
            # exploitation
            next_states = [str(float(state) * (1+self.r)),
                           str((float(state) - price) * (1 + self.r) + price * self.up),
                           str((float(state) - price) * (1 + self.r) + price * self.down)]
            values = []
            for next_state in next_states:
                if next_state not in v_value.columns:
                    v_value[next_state] = 0
                values.append(v_value.loc[time, next_state])
            values = list(value for value in values if value < 0)
            if len(values) == 0:
                action = np.random.choice(self.actions)
            else:
                max_value = np.max(values)
                max_index = list(i for i,j in enumerate(values) if j == max_value)
                max_state = []
                for index in max_index:
                    max_state.append(next_states[index])
                # max_state = next_states[list(i for i,j in enumerate(values) if j == max_value)]
                # max_state = values[values.values == max_value].index
                max_actions = []
                for next_state in max_state:
                    if float(next_state) == float(state) * (1+self.r):
                        max_actions.append(0)
                    else:
                        max_actions.append(1)
                max_actions = np.unique(max_actions)
                action = np.random.choice(max_actions)

        return action

    def step(self, state, time, action, price):
        # get the next state and reward
        move = np.random.binomial(1, 0.5)
        if move == 1:
            next_price = price * self.up
        else:
            next_price = price * self.down

        if action == 0:
            next_state = float(state) * (1 + self.r)
        else:
            next_state = (float(state) - price) * (1 + self.r) + next_price

        # if time == self.period-1:
        #     reward = -np.exp(-self.a * next_state) / self.a
        # else:
        #     reward = 0

        # price = next_price
        return str(next_state), next_price

    def TD_method(self, v_value, epsilon):
        state = str(float(self.W0))
        time = 0
        price = self.price

        while time < self.period:
            if state not in v_value.columns:
                v_value[state] = 0

            if time < (self.period - 1):
                action = self.this_action(state, price, time, v_value, epsilon)
                next_state, next_price = self.step(state, time, action, price)
                if next_state not in v_value.columns:
                    v_value[next_state] = 0
                # rewards = -np.exp(-self.a * float(state)) / self.a
                rewards = float(next_state) - float(state)
                v_value.loc[time, state] += self.ALPHA * (
                        rewards + self.GAMMA * v_value.loc[time + 1, next_state]
                        - v_value.loc[time, state])
                state = next_state
                price = next_price
            else:
                reward = -np.exp(-self.a * float(state)) / self.a
                v_value.loc[time, state] += self.ALPHA * (reward + 0 - v_value.loc[time, state])

            time += 1

        return v_value

    def TD_train(self, epsilon, epoch=20000):
        v_value = pd.DataFrame()
        v_value.index = list(range(self.period))

        for k in range(epoch):
            v_value = self.TD_method(v_value, epsilon)

        self.v_value = v_value

        return

    def get_v_value(self):
        return self.v_value

    def multi_train(self, times, period):
        all_result = pd.DataFrame()
        for k in trange(times):
            self.TD_train(epsilon=0.1)
            v_value = self.get_v_value()
            optimal_state = []
            for i in range(period):
                series = v_value.iloc[i, :]
                series = series[series.values != 0]
                a = v_value.columns.values[v_value.iloc[i, :] == np.max(series)]
                optimal_state.append(float(a))
            all_result[k] = optimal_state
        return all_result.mode(axis=1)

    def get_optimal_policy(self, optimal_value):
        v_value = self.v_value
        optimal_policy = []
        # optimal_value = self.multi_train(self.times,self.period)
        for time in range(v_value.shape[0] - 1):
            if optimal_value.iloc[time + 1, 0] / optimal_value.iloc[time, 0] == 1.05:
                optimal_policy.append(0)
            else:
                optimal_policy.append(1)
        return optimal_policy


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    period = 5
    solver = asset_allocation(period=period, r=0.05, ALPHA=0.5)
    result = solver.multi_train(times=5, period=period)
    print(result)
    optimal_policy = solver.get_optimal_policy(result)
    print(optimal_policy)