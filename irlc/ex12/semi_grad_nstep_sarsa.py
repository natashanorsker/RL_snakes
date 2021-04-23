"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc.ex01.agent import train
import gym
from irlc.ex11.semi_grad_sarsa import LinearSemiGradSarsa
from irlc.ex11.nstep_sarsa_agent import SarsaNAgent

class LinearSemiGradSarsaN(SarsaNAgent, LinearSemiGradSarsa): 
    def __init__(self, env, gamma=0.99, alpha=0.5, epsilon=0.1, q_encoder=None, n=1):
        """
        Note you can access the super-classes as:
        >> SarsaNAgent.pi(self, s) # Call the pi(s) as implemented in SarsaNAgent
        Alternatively, just inherit from Agent and set up data structure as required.
        """
        SarsaNAgent.__init__(self, env, gamma, alpha=alpha, epsilon=epsilon, n=n)
        LinearSemiGradSarsa.__init__(self, env, gamma, alpha=alpha, epsilon=epsilon, q_encoder=q_encoder)
        # self.Q = None  # Inherited from SarsaNAgent but of no use to us 

    def pi(self, s, k=None):
        return SarsaNAgent.pi(self, s, k)

    def _q(self, s, a): 
        """
        Return Q(s,a) using the linear function approximator with weights self.w; i.e. use self.q
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def _upd_q(self, s, a, delta): 
        """
        Update the weight-vector w using the appropriate rule (see exercise description). I.e. the update
        should be of the form

        self.w += self.alpha * delta * (gradient of Q(s,a;w)

        where
           delta = (G^n - Q(s,a;w)
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def __str__(self):
        return f"LinSemiGradSarsaN{self.gamma}_{self.epsilon}_{self.alpha}_{self.n}"


experiment_nsarsa = "experiments/mountaincar_SarsaN"
if __name__ == "__main__":
    from irlc.ex12.semi_grad_sarsa_lambda import alpha, plot_including_week10, experiment_sarsaL, episodes
    import irlc.ex09.envs
    env = gym.make("MountainCar500-v0")
    for _ in range(1):
        agent = LinearSemiGradSarsaN(env, gamma=1, alpha=alpha, epsilon=0, n=4)
        train(env, agent, experiment_nsarsa, num_episodes=episodes, max_runs=10)
    # plot while including the results from last week for Sarsa and Q-learning
    plot_including_week10([experiment_sarsaL, experiment_nsarsa],output="semigrad_sarsan")
