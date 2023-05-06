from stock_American_new import Binomial_stock, GBM_stock
import numpy as np

def episodes_50_american(agent, S, K, sigma, r, n_step):
    '''
    evaluation function for discrete hedging environment.
    this function will simmulate 50 episodes under the current hedging strategy
    '''

    scores = []
    early_exercise = []
    for i in range(50):
        early_exer_per_epoch = []
        env = Binomial_stock(S, K, r, sigma, n_step)
        done = False
        already_strike = False
        score = 0
        state = env.reset()
        while not done:
            action = agent.final_action(state)
            next_state, reward, done, already_strike = env.step(action, state, already_strike)
            score += reward
            state = next_state
            if already_strike:
                if state[0] < n_step:
                    early_exer_per_epoch.append(state[0])
        scores.append(score)
        if len(early_exer_per_epoch) > 0:
            early_exercise.append(np.min(early_exer_per_epoch))
    print('exercise early average at time', np.mean(early_exercise), 'early exercise ratio is', len(early_exercise) / 50)

    return np.mean(scores)

def episodes_50_gbm_american(agent, S, K, sigma, r, n_step):
    '''
    evaluation function for discrete hedging environment.
    this function will simmulate 50 episodes under the current hedging strategy
    '''

    scores = []
    early_exercise = []
    for i in range(50):
        early_exer_per_epoch = []
        env = GBM_stock(S, K, r, sigma, n_step)
        done = False
        already_strike = False
        score = 0
        state = env.reset()
        while not done:
            action = agent.final_action(state)
            next_state, reward, done, already_strike = env.step(action, already_strike)
            score += reward
            state = next_state
            if already_strike:
                step = n_step - state[0]*252
                if step < n_step:
                    early_exer_per_epoch.append(step)
        scores.append(score)
        if len(early_exer_per_epoch) > 0:
            early_exercise.append(np.min(early_exer_per_epoch))
    print('exercise early average at time', np.mean(early_exercise), 'early exercise ratio is', len(early_exercise)/50)

    return np.mean(scores)
