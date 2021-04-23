"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import gym
import numpy as np
import os
from irlc.ex13.deepq_agent import DeepQAgent
from matplotlib import pyplot as plt
from irlc.ex13.deepq_agent import framework, USE_KERAS
if USE_KERAS: from irlc.ex13.keras_networks import KerasNetwork as QNetwork
else: from irlc.ex13.torch_networks import TorchNetwork as QNetwork

class DoubleQAgent(DeepQAgent):
    def __init__(self, env, network=None, buffer=None, gamma=0.99, epsilon=0.2, alpha=0.001, tau=0.1, batch_size=32,
                    replay_buffer_size=2000, replay_buffer_minreplay=500):
        super().__init__(env, network=network, buffer=buffer, gamma=gamma,epsilon=epsilon, alpha=alpha, batch_size=batch_size,
                         replay_buffer_size=replay_buffer_size, replay_buffer_minreplay=replay_buffer_minreplay)
        # The target network play the role of q_{phi'} in the slides.
        self.target = QNetwork(env, learning_rate=alpha, trainable=False) if network is None else network(env, learning_rate=alpha, trainable=False)
        self.tau = tau # Rate at which the weights in the target network is updated (see slides)

    def train(self, s, a, r, sp, done=False):
        self.memory.push(s, a, r, sp, done)
        if len(self.memory) > self.replay_buffer_minreplay:
            self.experience_replay()
            # TODO: 1 lines missing.
            raise NotImplementedError("update Phi here in the self.target network")
        self.steps, self.episodes = self.steps + 1, self.episodes + done

    def experience_replay(self):
        """ Update the double-Q method, i.e. make sure to select actions a' using self.Q
        but evaluate the Q-values using the target network (see slides).
        In other words,
        > self.target(s)
        is a Q-function network which evaluates
        > q-hat_{\phi'}(s,:).
        Asides this, the code will be nearly identical to the basic DQN agent """
        s,a,r,sp,done = self.memory.sample(self.batch_size)
        # TODO: 5 lines missing.
        raise NotImplementedError("")
        self.Q.fit(s, target=target)

    def save(self, path):
        super().save(path)
        self.target.save(os.path.join(path, "Q_target")) # also save target network

    def load(self, path):
        loaded = super().load(path)
        if loaded:
            self.Q.load(os.path.join(path, "Q_target")) # also load target network
        return loaded


    def __str__(self):
        return f"doubleDQN_{self.gamma}"

from irlc.ex13.deepq_agent import cartpole_dqn_options
cartpole_doubleq_options = {**cartpole_dqn_options, 'tau': 0.08}

def mk_cartpole():
    env = gym.make("CartPole-v0")
    agent = DoubleQAgent(env, **cartpole_doubleq_options)
    return env, agent

if __name__ == "__main__":
    from irlc import main_plot

    env_id = "CartPole-v0"
    MAX_EPISODES = 200
    for j in range(1):
        env, agent = mk_cartpole()
        from irlc.ex01.agent import train
        ex = f"experiments/{framework}_cartpole_double_dqn"
        train(env, agent, experiment_name=ex, num_episodes=MAX_EPISODES)
        main_plot([f"experiments/{framework}_cartpole_dqn", ex], estimator=None, smoothing_window=None)
        savepdf("cartpole_double_dqn")
        plt.show()
