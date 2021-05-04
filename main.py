import irlc
from irlc.ex09.rl_agent import TabularAgent, TabularQ
from irlc.ex09.rl_agent import TabularAgent
from qtrain import qtrain
import gym
from irlc import main_plot
import matplotlib.pyplot as plt
from irlc import savepdf
from irlc.ex11.q_agent import QAgent, RAgent
from rasmusmus import *


# Set total number of episodes
n_episodes = 5000

epsilon = 1 # Exploration rate
decay_epsilon = (True, 1.45, n_episodes // 100)

alpha = 0.1 # Learning Rate
betas = [0.2]
gammas = [0.9] # Discount Factor


max_runs = 10
max_steps = 10000000

grid_sizes = [[10,10],[15,15],[20,20]]
def qsnake(grid_size, gamma):
    q_exp = f"experiments/grid{grid_size[0]}x{grid_size[0]}/q_gamma{gamma}"
    # Make environment instance
    env = Snake_env(grid_size)
    agent = QAgent(env, gamma=gamma, epsilon=epsilon, alpha=alpha)

    stats, trajectories, agent = qtrain(env, agent, q_exp, num_episodes=n_episodes, max_runs=max_runs,
                                       return_agent=True, max_steps=max_steps, decay_epsilon=decay_epsilon)

    # print(stats)
    return env, agent


def rsnake(grid_size, beta):
    q_exp = f"experiments/grid{grid_size[0]}x{grid_size[0]}/r_beta{beta}"
    # Make environment instance
    env = Snake_env(grid_size)
    agent = RAgent(env, alpha=alpha, beta=beta, epsilon=epsilon)

    stats, trajectories, agent = qtrain(env, agent, q_exp, num_episodes=n_episodes, max_runs=max_runs,
                                        return_agent=True, max_steps=max_steps, decay_epsilon=decay_epsilon)

    return env, agent


for grid_size in grid_sizes:
    for _ in range(5):
        for gamma in gammas:
            qsnake(grid_size, gamma)
        for beta in betas:
            rsnake(grid_size, beta)


# while True:
#     observation = env.reset()  # Constructs an instance of the game
#     snakes_remaining = 1
#     while snakes_remaining != 0:
#         env.render()
#         action = agent.Q.get_optimal_action(observation)
#         observation, reward, done, info = env.step(action)
#         snakes_remaining = info['snakes_remaining']
#         # print('OBS: ' , observation)
#         print(observation)
#         # print('Reward: ' , reward)
#         # print('Done: ' , done)
#         # print('Info: ' , info)
#
#         env.close()