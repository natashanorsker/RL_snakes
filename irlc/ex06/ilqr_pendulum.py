"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
from irlc.ex04.model_pendulum import GymSinCosPendulumModel
import matplotlib.pyplot as plt
import time
from irlc.ex06.ilqr_rendovouz_basic import ilqr
from irlc import savepdf

def pendulum(use_linesearch):
    print("> Using iLQR to solve Pendulum swingup task. Using linesearch?", use_linesearch)
    dt = 0.02
    model = GymSinCosPendulumModel(dt, cost=None)
    N = 250
    x0 = model.reset() # Get start position.
    n_iter = 200 # Use 200 iLQR iterations.
    # xs, us, J_hist, L, l = ilqr(model, ...) Write a function call like this, but with the correct parametesr
    # TODO: 1 lines missing.
    raise NotImplementedError("Call iLQR here (see hint above).")

    render = True
    if render:
        for i in range(2):
            render_(xs, model)
            time.sleep(2) # Sleep for two seconds between simulations.
    model.close()
    xs = np.asarray([model.discrete_states2continious_states(x) for x, u in zip(xs, us)]) # Convert to Radians. We use the build-in functions to change coordinates.
    xs, us = np.asarray(xs), np.asarray(us)

    t = np.arange(N) * dt
    theta = np.unwrap(xs[:, 0])  # Makes for smoother plots.
    theta_dot = xs[:, 1]

    pdf_ex = '_linesearch' if use_linesearch else ''
    stitle = "(using linesearch)" if use_linesearch else "(not using linesearch) "
    ev = 'pendulum_'
    _ = plt.plot(theta, theta_dot)
    _ = plt.xlabel("$\\theta$ (rad)")
    _ = plt.ylabel("$d\\theta/dt$ (rad/s)")
    _ = plt.title(f"Phase Plot {stitle}")
    plt.grid()
    savepdf(f"{ev}theta{pdf_ex}")
    plt.show()

    _ = plt.plot(t, us)
    _ = plt.xlabel("time (s)")
    _ = plt.ylabel("Force (N)")
    _ = plt.title(f"Action path {stitle}")
    plt.grid()
    savepdf(f"{ev}action{pdf_ex}")
    plt.show()

    _ = plt.plot(J_hist)
    _ = plt.xlabel("Iteration")
    _ = plt.ylabel("Total cost")
    _ = plt.title(f"Total cost-to-go {stitle}")
    plt.grid()
    savepdf(f"{ev}J{pdf_ex}")
    plt.show()

def render_(xs, env):
    for i in range(xs.shape[0]):
        env.render(xs[i])

if __name__ == "__main__":
    pendulum(use_linesearch=False)
    pendulum(use_linesearch=True)
