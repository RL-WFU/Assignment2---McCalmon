import numpy as np
import gym
import matplotlib
from collections import defaultdict
from lib import plotting
matplotlib.style.use('ggplot')

env = gym.make("Blackjack-v0")

observation = env.reset()
#This returns a tuple of player score, dealer first, usable ace
print(observation)

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    # Args:
    # Policy: a function that maps an observation to action probabilities
    # env: OpenAI gym environment
    # num_episodes: number of episodes to sample
    # discount_factor: gamma discount factor

    # Returns:
    #   A dictionary that maps from state -> value
    #   The state is a tuple and the value is a float

    # These keep track of sum and count of returns for each state to calculate
    # an average. We could use an array to save all returns, but that is
    # memory inefficient
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # Final value function
    V = defaultdict(float)

    for i in range(num_episodes):
        ep_history = []
        state = env.reset()
        for j in range(100):
            action = policy(state)
            nextState, r, d, _ = env.step(action)
            ep_history.append((state, action, r))
            if d:
                break
            state = nextState

        states = []
        for k, triple in enumerate(ep_history):
            states.append(triple[0])

        visits = []
        for k, state in enumerate(states):
            if state not in visits:
                visits.append(state)
                first_visit = k
                G = 0
                for j, triple in enumerate(ep_history[first_visit:]):
                    G += triple[2] * (discount_factor**j)
                returns_sum[state] += G
                returns_count[state] += 1

            V[state] = returns_sum[state] / returns_count[state]




    return V


def sample_policy(observation):
    # A policy that sticks if the player score is >20 and hits otherwise

    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


#V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
#plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")