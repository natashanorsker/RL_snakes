"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf). 
"""
import numpy as np
from irlc.ex01.agent import train
import gym
from irlc import main_plot
import matplotlib.pyplot as plt
from irlc import savepdf
from irlc.ex11.sarsa_agent import SarsaAgent
from irlc.ex11.q_agent import QAgent
from irlc.ex13.maze_dyna_environment import MazeEnvironment

class DynaQ(QAgent):
    """
    Implement the tabular dyna-Q agent (SB18, Section 8.7).
    """
    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1, n=5):
        super().__init__(env, gamma, alpha=alpha, epsilon=epsilon)
        """
        Model is a list of experience, i.e. of the form
        Model = [ (s_t, a_t, r_{t+1}, s_{t+1}, done_t), ...] 
        """
        self.Model = []
        self.n = n # number of planning steps

    def q_update(self, s, a, r, sp, done): 
        """
        Update the Q-function self.Q[s][a] as in regular Q-learning
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def train(self, s, a, r, sp, done=False):
        self.q_update(s,a,r,sp,done)
        self.Model.append( (s,a, r,sp, done))
        for _ in range(self.n): 
            """ Obtain a random transition from the replay buffer. You can use np.random.randint 
            then call self.q_update on the random sample. """
            # TODO: 2 lines missing.
            raise NotImplementedError("Implement function body")

    def __str__(self):
        return f"DynaQ_{self.gamma}_{self.epsilon}_{self.alpha}_{self.n}"


def dyna_experiment(env, env_name='maze',num_episodes=50,epsilon=0.1, alpha=0.5, gamma=1.0):
    for _ in range(50):
        agents = [QAgent(env, epsilon=epsilon, alpha=alpha,gamma=gamma),
                  SarsaAgent(env, epsilon=epsilon, alpha=alpha, gamma=gamma),
                  DynaQ(env, epsilon=epsilon, alpha=alpha,gamma=gamma,n=5),
                  DynaQ(env, epsilon=epsilon, alpha=alpha,gamma=gamma, n=50),
                  ]

        experiments = []
        for agent in agents:
            expn = f"experiments/b{env_name}_{str(agent)}"
            train(env, agent, expn, num_episodes=num_episodes, max_runs=100)
            experiments.append(expn)
    return experiments

if __name__ == "__main__":
    env = MazeEnvironment()
    # TODO: 1 lines missing.
    raise NotImplementedError("")
    main_plot(experiments, smoothing_window=None, y_key="Length")
    plt.ylim([0, 500])
    plt.title("Dyna Q on simple Maze (Figure 8.2)")
    savepdf("dynaq_maze_8_2")
    plt.show()

    # Part 2: Cliffwalking old
    env = gym.make('CliffWalking-v0')
    gamma, alpha, epsilon = 1, 0.5, 0.1
    # TODO: 1 lines missing.
    raise NotImplementedError("")
    main_plot(experiments, smoothing_window=5)
    plt.ylim([-150, 0])
    plt.title("Dyna-Q learning on " + env.spec._env_name)
    savepdf("dyna_cliff")
    plt.show()
