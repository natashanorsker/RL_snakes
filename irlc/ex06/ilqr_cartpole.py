"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import matplotlib.pyplot as plt
import numpy as np
from irlc.ex04.model_cartpole import GymSinCosCartpoleEnvironment
import time
from irlc.ex06.ilqr_rendovouz_basic import ilqr
from irlc import savepdf

# Number of steps.
N = 100
def cartpole(use_linesearch):
    env = GymSinCosCartpoleEnvironment()
    x0 = env.reset()
    xs, us, J_hist, L, l = ilqr(env.discrete_model, N, x0, n_iter=300, use_linesearch=use_linesearch)
    # xs0 = xs.copy()

    plot_cartpole(env, xs, us, use_linesearch=use_linesearch)

def plot_cartpole(env, xs, us, J_hist=None, use_linesearch=True):
    animate(xs, env)
    env.close()
    # Transform actions/states using build-in functions.
    def gapply(f, xm):
        usplit = np.split(xm, len(xm))
        u2 = [f(u.flat) for u in usplit]
        us = np.stack(u2)
        return us
    us = gapply(env.discrete_model.discrete_actions2continious_actions, us)
    xs = gapply(env.discrete_model.discrete_states2continious_states, xs)

    t = np.arange(N + 1) * env.dt
    x = xs[:, 0]
    theta = np.unwrap(xs[:, 2])  # Makes for smoother plots.
    theta_dot = xs[:, 3]
    pdf_ex = '_linesearch' if use_linesearch else ''
    ev = 'cartpole_'

    plt.plot(theta, theta_dot)
    plt.xlabel("theta (rad)")
    plt.ylabel("theta_dot (rad/s)")
    plt.title("Orientation Phase Plot")
    plt.grid()
    savepdf(f"{ev}theta{pdf_ex}")
    plt.show()

    _ = plt.plot(t[:-1], us)
    _ = plt.xlabel("time (s)")
    _ = plt.ylabel("Force (N)")
    _ = plt.title("Action path")
    plt.grid()
    savepdf(f"{ev}action{pdf_ex}")
    plt.show()

    _ = plt.plot(t, x)
    _ = plt.xlabel("time (s)")
    _ = plt.ylabel("Position (m)")
    _ = plt.title("Cart position")
    plt.grid()
    savepdf(f"{ev}position{pdf_ex}")
    plt.show()
    if J_hist is not None:
        _ = plt.plot(J_hist)
        _ = plt.xlabel("Iteration")
        _ = plt.ylabel("Total cost")
        _ = plt.title("Total cost-to-go")
        plt.grid()
        savepdf(f"{ev}J{pdf_ex}")
        plt.show()

def animate(xs0, env):
    render = True
    if render:
        for i in range(2):
            render_(xs0, env.discrete_model)
            time.sleep(1)
        # env.viewer.close()

def render_(xs, env):
    for i in range(xs.shape[0]):
        x = xs[i]
        env.render(x=x)

if __name__ == "__main__":
    cartpole(use_linesearch=True)
