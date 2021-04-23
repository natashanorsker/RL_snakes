"""
Code for frozen lake environment: https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
"""
import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from irlc import savepdf
from irlc.ex01.dp_model import DPModel
from irlc.ex01.frozen_lake import to_rc  # we need a little help from our friends.
from irlc.ex02.dp import DP_stochastic
from irlc import Agent, train

class Gym2DPModel(DPModel):
    def __init__(self, gym_env, N=None):
        """
        Converts a Discrete gym environment into a DP problem. This is possible since the gym environment explicitly store the transition probabilities internally

        See https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
        for definition of P.

        Specifically, look at

        https://github.com/openai/gym/blob/8e5a7ca3e6b4c88100a9550910dfb1a6ed8c5277/gym/envs/toy_text/discrete.py#L16

        for how it is converted into a step() function.

        *** Hint ***
        When you implement this function, pay close attention to

        self.P[x][u]

        which encodes the possible outcomes (new state, reward, probability)
        of taking action u in state x. To get an idea of this notation, if
        'w' is a possible random disturbance, then

        self.P[x][u][w]

        is a valid way to index the structure and can be used to extract new state, reward and probability of w occuring.

        See also the aforementioned step function above (this was how I figured it out).
        """
        P = gym_env.P if hasattr(gym_env, 'P') else gym_env.unwrapped.P
        N = gym_env._max_episode_steps if N is None else N
        self.P = P
        self.observation_space = gym_env.observation_space
        super(Gym2DPModel, self).__init__(N)

    def f(self, x, u, w, k): 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def g(self, x, u, w, k): 
        """ Remember the DP environment has a (positive) reward, we want a negative cost. Multiply reward by -1. """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def gN(self, x):
        """
        Since N is so big let's just not care what happens at the last step.
        """
        return 0

    def S(self, k):
        return range(self.observation_space.n)

    def A(self, x, k):
        return set(self.P[x].keys())

    def Pw(self, x, u, k): 
        """
        at step k, given x_k, u_k, compute the set of random noise disturbances w
        and their probabilities as a dictionary {..., w_i: p(w_i), ...}

        The possible values of w corresponds to the index of the transitions. I.e.
        we define w as the index:

        self.P[x][u][w]

        with probability given as:

        pw = self.P[x][u][w][0]
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

def plot_value_function(env, J, figsize=8, ncol=None):
    if ncol is None:
        ncol = env.ncols  # columns.
    S = env.observation_space.n  # number of states (squares)
    nrow = S // ncol  # rows

    A = np.zeros((nrow, ncol))  # plot the value function as a matrix
    for i in range(S):
        nr, nc = to_rc(i, ncol)
        # TODO: 1 lines missing.
        raise NotImplementedError("Update A[nr,nc] to contain the value function.")

    if figsize is not None:
        plt.figure(figsize=(figsize, figsize))
    sns.heatmap(A, cmap="YlGnBu", annot=True, cbar=False, square=True,fmt='g')

if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    frozen_lake = Gym2DPModel(gym_env=env)
    J, pi = DP_stochastic(frozen_lake)
    """
    Wrap the policy we just computed into a handy function.
    """
    T = 10000
    s = env.reset() # Get initial state s=0
    # To test the method, we use the DPAgent class we implemented earlier:
    from irlc.ex02.dp_agent import DynamicalProgrammingAgent
    agent = DynamicalProgrammingAgent(env, frozen_lake)
    # Evaluate the policy by simulation.
    stats, _ = train(env, agent, num_episodes=T)
    Er = np.mean([stat['Accumulated Reward'] for stat in stats])

    print("Estimated reward using trained policy and MC rollouts", Er)  
    print("Reward as computed using DP", -J[0][s])  

    """
    Plot the value function, i.e. the value function in each state. Remember we start in (0,0)
    and end in (4,4).
    
    We want to reproduce the plot at the very bottom of: 
    
    https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
    
    (note, you can also try the 8x8 environment if you like)
    """
    # plt.figure(figsize=(8, 8))
    plot_value_function(env.env, {k: -J[0][k] for k in J[0]})
    plt.title("Value function J(x). Note we start in (0,0) and terminate in (4,4)")
    savepdf("frozen_DP_J")
    plt.show()
