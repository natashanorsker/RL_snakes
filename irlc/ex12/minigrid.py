"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc import main_plot
from irlc.ex01.agent import train
import matplotlib.pyplot as plt
import gym
from irlc import savepdf
from irlc.ex11.semi_grad_q import LinearSemiGradQAgent
from irlc.ex11.q_agent import QAgent
from irlc.ex11.sarsa_agent import SarsaAgent
from irlc.ex11.semi_grad_sarsa import LinearSemiGradSarsa

if __name__ == "__main__":
    env0 = gym.make("MiniGrid-Empty-5x5-v0") # Try the MiniGrid-DoorKey-5x5-v0 environment for a more difficult challenge
    from gym_minigrid.wrappers import ImgObsWrapper
    from irlc.ex12.minigrid_wrappers import HashableImgObsWrapper, ProjectObservationSpaceWrapper
    

    """ Strip away the (middle) slice of the 7 x 7 x 3 tensor of observations """ 
    env0 = ProjectObservationSpaceWrapper(env0, dims=(0,2))
    """ Use tight upper/lower bounds on the tensor and remove mission string """
    env_linear = ImgObsWrapper(env0) 
    """ Create a hashable observation (i.e. a tuple) suitable for tabular methods """ 
    env_tabular = HashableImgObsWrapper(env0) 

    alpha = 0.2 / 8
    epsilon = 0.2

    exps_tabular = [(f"experiments/mg_q_b", QAgent(env_tabular, epsilon=epsilon, alpha=alpha), env_tabular),
                    (f"experiments/mg_sarsa_b", SarsaAgent(env_tabular, epsilon=epsilon, alpha=alpha), env_tabular)]

    lenv = env_linear
    exps_lin = []
    for alpha in [0.001, 0.005, 0.01]:
        exps_lin += [
                    (f"experiments/mg_sg_sarsa_b_{alpha:3}", LinearSemiGradSarsa(lenv, epsilon=epsilon, alpha=alpha), lenv),
                    (f"experiments/mg_sg_q_b_{alpha:3}", LinearSemiGradQAgent(lenv, epsilon=epsilon, alpha =alpha), lenv)
                    ]

    for exp, agent, env_linear in exps_tabular + exps_lin:
        _, stats, _ = train(env_linear, agent, experiment_name=exp, num_episodes=200)

    main_plot([exp for exp,_, _ in exps_tabular+exps_lin], ci=None, linewidth=4)

    plt.ylim([-.05, 1.1])
    plt.xlim([0, 350])
    savepdf("minigrid5_sarsa_q_semigrad")
    plt.show()
