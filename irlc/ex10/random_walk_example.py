"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf). 
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from irlc import savepdf
from irlc.ex10.td0_evaluate import TD0ValueAgent
from irlc.ex10.mc_evaluate import MCEvaluationAgent
import seaborn as sns
import pandas as pd
from irlc.ex01.agent import train
from irlc.ex09.mdp import MDP2GymEnv, MDP

class ChainMRP(MDP):
    def __init__(self, length=6):
        """
        Build the "Chain MRP" old from (SB18). Terminal states are [0,6],
        all states are [0,1,2,3,4,5,6] and initial state is 3. (default settings).
        """
        self.max_states = length
        super().__init__(initial_state=length // 2)

    def is_terminal(self, state):
        return state == 0 or state == self.max_states

    def A(self, s): # 0: left, 1: right.
        return [0,1]

    def Psr(self, s, a): 
        # TODO: 2 lines missing.
        raise NotImplementedError("Implement function body")

class ChainEnvironment(MDP2GymEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(mdp=ChainMRP(*args, **kwargs))

if __name__ == "__main__":
    """ plot results as in (SB18, Example 6.2) """
    env = ChainEnvironment()
    V_init = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    V_true = np.array([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
    states = range(1,6)
    """
    This is a bit janky. In the old, the value-function is initialized at 
    0.5, however (see (SB18)) the value function must be initialized at 
    0 in terminal states. We make a function to initialize the value function
    and pass it along to the ValueAgent; the ValueAgent then uses a subclassed 
    defaultdict which can handle a parameterized default value. Good times, but no good alternative. """
    v_init_fun = lambda x: 0.5

    fig, ax = plt.subplots(figsize=(15, 6), ncols=2)
    """ Make TD plot """
    td_episodes = [0, 1, 10, 100]
    V_current = np.copy(V_init)
    xticks = ['A', 'B', 'C', 'D', 'E']

    for i, episodes in enumerate(td_episodes):
        agent = TD0ValueAgent(env, v_init_fun=v_init_fun)
        train(env, agent, num_episodes=episodes,verbose=False)
        vs = [agent.value(s) for s in states]
        ax[0].plot(vs, label=f"{episodes} episodes", marker='o')

    ax[0].plot(V_true, label='true values', marker='o')
    ax[0].set(xlabel='State', ylabel='Estimated Value', title='Estimated Values TD(0)',
              xticks=np.arange(5), xticklabels=['A','B','C','D','E'])
    ax[0].legend()

    """ Make TD vs. MC plot """
    td_alphas = [0.05, 0.15, 0.1]
    mc_alphas = [0.01, 0.03]
    episodes = 100
    runs = 200

    def eval_mse(agent):
        errors = []
        for i in range(episodes):
            V_ = [agent.value(s) for s in states] #list(map(agent.value, states))
            train(env, agent, num_episodes=1, verbose=False)
            z = np.sqrt(np.sum(np.power(V_ - V_true, 2)) / 5.0)
            errors.append(z)
        return errors

    methods = [(TD0ValueAgent, 'TD', alpha) for alpha in td_alphas]
    methods += [(MCEvaluationAgent, 'MC', alpha) for alpha in mc_alphas]

    dfs = []
    for AC,method,alpha in tqdm(methods):
        TD_mse = []
        for r in range(runs):
            agent = AC(env, alpha=alpha, gamma=1, v_init_fun=v_init_fun)
            err_ = eval_mse(agent)
            TD_mse.append( np.asarray(err_))

        # Happy times with pandas. Let's up the production value by also plotting 1 std.
        for u,mse in enumerate(TD_mse):
            df = pd.DataFrame(mse, columns=['rmse'])
            df.insert(len(df.columns), 'Unit', u)
            df.insert(len(df.columns), 'Episodes', range(episodes))
            df.insert(len(df.columns), 'Condition', f"{method} $\\alpha$={alpha}")
            dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    sns.lineplot(data=data, x='Episodes', y='rmse', hue="Condition", ci=95, estimator='mean')
    plt.ylabel("RMS error (averaged over states)")
    plt.title("Empirical RMS error, averaged over states")
    savepdf("random_walk_example")
    plt.show()
