import irlc
from irlc.ex09.rl_agent import TabularAgent, TabularQ
from irlc.ex09.rl_agent import TabularAgent
from irlc import train
import gym
from irlc import main_plot
import matplotlib.pyplot as plt
from irlc import savepdf
from irlc.ex11.q_agent import QAgent
from rasmusmus import *


# Set total number of episodes
n_episodes = 1000000

epsilon = 0.1
max_runs = 10
max_steps = 5000

alpha = 0.5

grid_size = [30, 30]
def qsnake():
    # Make environment instance
    env = Snake_env()
    env.grid_size = grid_size
    agent = QAgent(env, epsilon=epsilon, alpha=alpha)

    stats, trajectories, agent = train(env, agent, num_episodes=n_episodes//2, max_runs=max_runs,
                                       return_agent=True, max_steps=max_steps)
    agent.epsilon = 0
    stats, trajectories, agent = train(env, agent, num_episodes=n_episodes//2, max_runs=max_runs,
                                       return_agent=True, max_steps=max_steps)
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