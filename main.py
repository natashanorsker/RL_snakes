import irlc
from irlc.ex09.rl_agent import TabularAgent, TabularQ
from irlc.ex09.rl_agent import TabularAgent
from qtrain import qtrain
import gym
from irlc import main_plot
import matplotlib.pyplot as plt
from irlc import savepdf
from irlc.ex11.q_agent import QAgent
from rasmusmus import *


# Set total number of episodes
n_episodes = 100000

epsilon = 1 # Exploration rate
decay_epsilon = (True, 2, 500)

alpha = 0.5 # Learning Rate
gamma = 0.99 # Discount Factor


max_runs = 10
max_steps = 10000

grid_size = [30, 30]
def qsnake():
    # Make environment instance
    env = Snake_env()
    env.grid_size = grid_size
    agent = QAgent(env, gamma=gamma, epsilon=epsilon, alpha=alpha)

    stats, trajectories, agent = qtrain(env, agent, num_episodes=n_episodes, max_runs=max_runs,
                                       return_agent=True, max_steps=max_steps, decay_epsilon=decay_epsilon)
    print(agent.epsilon)
    # print(stats)
    return env, agent


env, agent = qsnake()

while True:
    observation = env.reset()  # Constructs an instance of the game
    snakes_remaining = 1
    while snakes_remaining != 0:
        env.render()
        action = agent.Q.get_optimal_action(observation)
        observation, reward, done, info = env.step(action)
        snakes_remaining = info['snakes_remaining']
        # print('OBS: ' , observation)
        print(observation)
        # print('Reward: ' , reward)
        # print('Done: ' , done)
        # print('Info: ' , info)

        env.close()