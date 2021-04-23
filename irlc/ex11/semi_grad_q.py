"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc.ex01.agent import train
import gym
from irlc import main_plot
import matplotlib.pyplot as plt
from irlc.ex11.q_agent import QAgent
from irlc.ex11.feature_encoder import LinearQEncoder
from irlc import savepdf

class LinearSemiGradQAgent(QAgent): 
    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1, q_encoder=None):
        """ The Q-values, as implemented using a function approximator, can now be accessed as follows:

        >> self.Q(s,a) # Compute q-value
        >> self.Q.x(s,a) # Compute gradient of the above expression wrt. w
        >> self.Q.w # get weight-vector.

        I would recommend inserting a breakpoint and investigating the above expressions yourself; check also the implementing class,
        although the code is a bit messy :-).
        """
        super().__init__(env, gamma, epsilon=epsilon, alpha=alpha)
        self.Q = LinearQEncoder(env, tilings=8) if q_encoder is None else q_encoder 

    def train(self, s, a, r, sp, done=False): 
        # TODO: 4 lines missing.
        raise NotImplementedError("Implement function body")

    def __str__(self):
        return f"LinearSemiGradQ{self.gamma}_{self.epsilon}_{self.alpha}"

num_of_tilings = 8
alpha = 1 / num_of_tilings
episodes = 300
x = "Episode"
experiment_q = "experiments/mountaincar_semigrad_q"

if __name__ == "__main__":
    import gym
    from irlc.ex09 import envs
    env = gym.make("MountainCar500-v0")
    for _ in range(10):
        agent = LinearSemiGradQAgent(env, gamma=1, alpha=alpha, epsilon=0)
        train(env, agent, experiment_q, num_episodes=episodes, max_runs=10)
    main_plot(experiments=[experiment_q], x_key=x, y_key='Length', smoothing_window=30, resample_ticks=100)
    savepdf("semigrad_q")
    plt.show()
