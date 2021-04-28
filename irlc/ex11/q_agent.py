"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf). 
"""
from irlc.ex09.rl_agent import TabularAgent
from irlc import train
import gym
from irlc import main_plot
import matplotlib.pyplot as plt
from irlc import savepdf
from irlc.ex09.value_iteration_agent import ValueIterationAgent

class QAgent(TabularAgent):
    """
    Implement the Q-learning agent (SB18, Section 6.5)
    Note that the Q-datastructure already exist, as do helper functions useful to compute an epsilon-greedy policy
    (see TabularAgent class for more information)
    """
    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1):
        self.alpha = alpha
        super().__init__(env, gamma, epsilon)

    def pi(self, s, k=None):
        """
        Return current action using epsilon-greedy exploration. Look at the TabularAgent class
        for ideas.
        """
        return self.pi_eps(s)
        # raise NotImplementedError("Implement function body")

    def train(self, s, a, r, sp, done=False):
        """
        Implement the Q-learning update rule, i.e. compute a* from the Q-values.
        As a hint, note that self.Q[sp,a] corresponds to q(s_{t+1}, a) and
        that what you need to update is self.Q[s, a] = ...

        You may want to look at self.Q.get_optimal_action(state) to compute a = argmax_a Q[s,a].
        """
        astar = self.Q.get_optimal_action(sp)
        maxQ = self.Q[sp,astar]
        self.Q[s, a] = self.Q[s, a] + self.alpha * (r + self.gamma * maxQ - self.Q[s, a])
        # raise NotImplementedError("Implement function body")

    def __str__(self):
        return f"QLearner_{self.gamma}_{self.epsilon}_{self.alpha}"

q_exp = f"experiments/cliffwalk_Q"
epsilon = 0.1
max_runs = 10
alpha = 0.5
def cliffwalk():
    env = gym.make('CliffWalking-v0')
    agent = QAgent(env, epsilon=epsilon, alpha=alpha)

    train(env, agent, q_exp, num_episodes=200, max_runs=max_runs)
    vi_exp = "experiments/cliffwalk_VI"
    Vagent = ValueIterationAgent(env, epsilon=epsilon)
    train(env, Vagent, vi_exp, num_episodes=200, max_runs=max_runs)

    vi_exp_opt = "experiments/cliffwalk_VI_optimal"
    Vagent_opt = ValueIterationAgent(env, epsilon=0)
    train(env, Vagent_opt, vi_exp_opt, num_episodes=200, max_runs=max_runs)
    exp_names = [q_exp, vi_exp, vi_exp_opt]
    return env, exp_names

if __name__ == "__main__":
    for _ in range(10):
        env, exp_names = cliffwalk()
    main_plot(exp_names, smoothing_window=10)
    plt.ylim([-100, 0])
    plt.title("Q-learning on " + env.spec._env_name)
    savepdf("Q_learning_cliff")
    plt.show()
