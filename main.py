import irlc
from irlc import Agent, train
from irlc.ex09.rl_agent import TabularAgent, TabularQ
from irlc.ex11.q_agent import QAgent
import gym
import gym_snake


# set grid size of snake map
# N =

# Set total number of iterations
num_episodes = 10  # change this :)
max_runs = 10

# Make environment instance
env = gym.make('snake-v0')

# Make Agent instance
agent = QAgent(env, gamma=1.0, alpha=0.5)

# Train Agent
snake_exp = f"experiments/snake"
train(env, agent, snake_exp, num_episodes=num_episodes, max_runs=max_runs)
