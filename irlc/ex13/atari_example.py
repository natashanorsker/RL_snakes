"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import matplotlib.pyplot as plt
import gym
from irlc import main_plot
from irlc.ex01.agent import train
from irlc.ex13.deepq_agent import linear_interp
from irlc.ex13.duel_deepq_agent import DuelQAgent
from irlc.ex13.deepq_agent import framework, USE_KERAS
if USE_KERAS: from irlc.ex13.keras_networks import KerasDuelNetworkAtari as DuelNetworkAtari
else: from irlc.ex13.torch_networks import TorchDuelNetworkAtari as DuelNetworkAtari

def mk_agent_atari():
    env = gym.make("SpaceInvaders-v0")
    delay_training = 50000
    max_episodes = 1000000
    from baselines.common.retro_wrappers import wrap_deepmind_retro
    env = wrap_deepmind_retro(env)
    epsilon_duel_atari = linear_interp(maxval=1.0, minval=0.1, delay=delay_training, miniter=500000)
    network = DuelNetworkAtari
    agent = DuelQAgent(env, network=network, gamma=0.99, epsilon=epsilon_duel_atari, tau=0.08,
                       replay_buffer_size=200000, replay_buffer_minreplay=delay_training)
    return env, agent, max_episodes

if __name__ == "__main__":
    env,agent,max_episodes = mk_agent_atari()
    ex = f"experiments/{framework}_atari_{agent}_{max_episodes}"
    episodes_per_run = 1000
    for k in range(max_episodes // episodes_per_run):
        print(k, f"Running atari training for another {episodes_per_run} episodes...")
        train(env, agent, experiment_name=ex, num_episodes=100, saveload_model=True)

    main_plot([ex], smoothing_window=None)
    plt.show()
