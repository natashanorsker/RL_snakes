"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import gym
import matplotlib.pyplot as plt
from irlc import main_plot
from irlc.ex01.agent import train
from irlc.ex13.double_deepq_agent import DoubleQAgent
from irlc.ex13.deepq_agent import framework, USE_KERAS
if USE_KERAS: from irlc.ex13.keras_networks import KerasDuelNetwork as DuelNetwork
else: from irlc.ex13.torch_networks import TorchDuelNetwork as DuelNetwork
from irlc.ex13.buffer import BasicBuffer
from irlc.ex13.double_deepq_agent import cartpole_doubleq_options

class DuelQAgent(DoubleQAgent):
    def __init__(self, env, network=None, buffer=None, gamma=0.99, epsilon=None, alpha=0.001, tau=0.1, batch_size=32,
                    replay_buffer_size=2000, replay_buffer_minreplay=500):
        network = DuelNetwork if network is None else network # Only relevant change
        buffer = buffer if buffer is not None else BasicBuffer(max_size=500000)
        super().__init__(env, network=network, buffer=buffer, gamma=gamma,epsilon=epsilon, alpha=alpha, tau=tau,batch_size=batch_size,
                         replay_buffer_size=replay_buffer_size, replay_buffer_minreplay=replay_buffer_minreplay)
        self.target.update_Phi(self.Q)

    def __str__(self):
        return f"DuelQ_{self.gamma}"

def mk_cartpole():
    env = gym.make("CartPole-v0")
    agent = DuelQAgent(env, **cartpole_doubleq_options)
    return env, agent

if __name__ == "__main__":
    env,agent = mk_cartpole()
    ex = f"experiments/{framework}_cartpole_duel_dqn3"
    train(env, agent, experiment_name=ex, num_episodes=200)
    main_plot([f"experiments/{framework}_cartpole_dqn", f"experiments/{framework}_cartpole_double_dqn", ex], smoothing_window=None)
    savepdf("cartpole_duel_dqn")
    plt.show()
