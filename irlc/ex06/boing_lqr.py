"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
import matplotlib.pyplot as plt
from irlc import savepdf
from irlc import train
from irlc.ex04.model_boing import BoingEnvironment
from irlc.ex06.lqr_agent import DiscreteLQRAgent

def boing_simulation():
    env = BoingEnvironment(Tmax=10, output=[10,0])
    agent = DiscreteLQRAgent(env, model=env.discrete_model)
    stats, trajectories = train(env, agent, return_trajectory=True)
    return stats, trajectories, env


def boing_experiment():
    stats, trajectories, env = boing_simulation()
    cmod = env.discrete_model.continuous_model
    t = trajectories[-1]
    out = t.state @ cmod.P.T

    plt.plot(t.time, out[:, 0], '-', label=env.observation_labels[0])
    plt.plot(t.time, out[:, 1], '-', label=env.observation_labels[1])
    plt.grid()
    plt.legend()
    plt.xlabel("Time/seconds")
    plt.ylabel("Output")
    savepdf("boing_lqr_output")
    plt.show()

    plt.plot(t.time[:-1], t.action[:, 0], '-', label=env.action_labels[0])
    plt.plot(t.time[:-1], t.action[:, 1], '-', label=env.action_labels[1])
    plt.xlabel("Time/seconds")
    plt.ylabel("Control action")
    plt.grid()
    plt.legend()
    savepdf("boing_lqr_action")
    plt.show()



if __name__ == "__main__":
    boing_experiment()
