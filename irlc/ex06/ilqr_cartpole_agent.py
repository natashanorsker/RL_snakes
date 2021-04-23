"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
from irlc.ex06.ilqr_agent import ILQRAgent
from irlc import train
from irlc import savepdf
import matplotlib.pyplot as plt
from irlc.ex04.model_cartpole import GymSinCosCartpoleEnvironment
from irlc import VideoMonitor

def cartpole_experiment(N=12, use_linesearch=True, figex="", animate=True):
    np.random.seed(2)
    Tmax = .9
    dt = Tmax/N

    env = GymSinCosCartpoleEnvironment(dt=dt, Tmax=Tmax, supersample_trajectory=True)
    agent = ILQRAgent(env, env.discrete_model, N=N, ilqr_iterations=200, use_linesearch=use_linesearch)
    if animate:
        env =VideoMonitor(env)
    stats, trajectories = train(env, agent, num_episodes=1, return_trajectory=True)

    agent.use_ubar = True
    stats2, trajectories2 = train(env, agent, num_episodes=1, return_trajectory=True)
    env.close()

    xb = agent.xbar
    tb = np.arange(N+1)*dt
    plt.figure(figsize=(8,6))
    F = 3
    plt.plot(trajectories[0].time, trajectories[0].state[:,F], 'k-', label='Closed-loop $\\pi$')
    plt.plot(trajectories2[0].time, trajectories2[0].state[:,F], '-', label='Open-loop $\\bar{u}_k$')

    plt.plot(tb, xb[:,F], '.-', label="iLQR rediction $\\bar{x}_k$")
    plt.xlabel("Time/seconds")
    plt.ylabel("$\cos(\\theta)$")
    plt.title(f"Pendulum environment $T={N}$")

    plt.grid()
    plt.legend()
    ev = "pendulum"
    savepdf(f"irlc_cartpole_theta_N{N}_{use_linesearch}{figex}")
    plt.show()

def plt_cartpole():
    cartpole_experiment(N=50, use_linesearch=True, animate=True)

if __name__ == '__main__':
    plt_cartpole()
