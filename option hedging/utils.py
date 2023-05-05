from stock import Binomial_stock, GBM_stock
import numpy as np

def episodes_50(agent, S, sigma, r, n_step):

    # this function will test 50 episodes under the current hedging strategy

    scores = []
    for i in range(50):
        env = Binomial_stock(S, r, sigma, n_step)
        done = False
        score = 0
        state = env.reset()
        while not done:
            action = agent.final_action(state)
            next_state, reward, done = env.step(action, state)
            score += reward
            state = next_state

        scores.append(score)

    return np.mean(scores)




def episodes_50_gbm(agent, S, sigma, r, n_step):

    scores = []
    for i in range(50):
        env = GBM_stock(S, r, sigma, n_step)
        done = False
        score = 0
        state = env.reset()
        while not done:
            action = agent.final_action(state)
            next_state, reward, done = env.step(action)
            score += reward
            state = next_state

        scores.append(score)

    return np.mean(scores)


