"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
import matplotlib.pyplot as plt
from irlc import savepdf
# from irlc.ex05orig.ilqr_rendovouz_basic import ilqr
# from irlc.ex06.ilqr import
from irlc.ex06.ilqr import ilqr_basic, ilqr_linesearch
from irlc.ex06.model_rendevouz import DiscreteRendevouzModel

def ilqr(env, N, x0, n_iter, use_linesearch, verbose=True):
    if not use_linesearch:
        xs, us, J_hist, L, l = ilqr_basic(env, N, x0, n_iterations=n_iter,verbose=verbose) 
    else:
        xs, us, J_hist, L, l = ilqr_linesearch(env, N, x0, n_iterations=n_iter, tol=1e-6,verbose=verbose)
    xs, us = np.stack(xs), np.stack(us)
    return xs, us, J_hist, L, l

def plot_vehicles(x_0, y_0, x_1, y_1, linespec='-', legend=("Vehicle 1", "Vehicle 2")):
    _ = plt.title("Trajectory of the two omnidirectional vehicles")
    _ = plt.plot(x_0, y_0, "r"+linespec, label=legend[0])
    _ = plt.plot(x_1, y_1, "b"+linespec, label=legend[1])

Tmax = 20
def solve_rendovouz(use_linesearch=False):
    env = DiscreteRendevouzModel()
    x0 = env.continuous_model.x0 # Starting position
    N = int(Tmax/env.dt)
    return ilqr(env, N, x0, n_iter=10, use_linesearch=use_linesearch), env

def plot_rendevouz(use_linesearch=False):
    (xs, us, J_hist, _, _), env = solve_rendovouz(use_linesearch=use_linesearch)
    N = int(Tmax / env.dt)
    dt = env.dt
    x_0 = xs[:, 0]
    y_0 = xs[:, 1]
    x_1 = xs[:, 2]
    y_1 = xs[:, 3]
    x_0_dot = xs[:, 4]
    y_0_dot = xs[:, 5]
    x_1_dot = xs[:, 6]
    y_1_dot = xs[:, 7]

    pdf_ex = '_linesearch' if use_linesearch else ''
    ev = 'rendevouz_'
    plot_vehicles(x_0, y_0, x_1, y_1, linespec='-', legend=("Vehicle 1", "Vehicle 2"))
    plt.legend()
    savepdf(f'{ev}trajectory{pdf_ex}')
    plt.show()

    t = np.arange(N + 1) * dt
    _ = plt.plot(t, x_0, "r")
    _ = plt.plot(t, x_1, "b")
    _ = plt.xlabel("Time (s)")
    _ = plt.ylabel("x (m)")
    _ = plt.title("X positional paths")
    _ = plt.legend(["Vehicle 1", "Vehicle 2"])
    savepdf(f'{ev}vehicles_x_pos{pdf_ex}')
    plt.show()

    _ = plt.plot(t, y_0, "r")
    _ = plt.plot(t, y_1, "b")
    _ = plt.xlabel("Time (s)")
    _ = plt.ylabel("y (m)")
    _ = plt.title("Y positional paths")
    _ = plt.legend(["Vehicle 1", "Vehicle 2"])
    savepdf(f'{ev}vehicles_y_pos{pdf_ex}')
    plt.show()

    _ = plt.plot(t, x_0_dot, "r")
    _ = plt.plot(t, x_1_dot, "b")
    _ = plt.xlabel("Time (s)")
    _ = plt.ylabel("x_dot (m)")
    _ = plt.title("X velocity paths")
    _ = plt.legend(["Vehicle 1", "Vehicle 2"])
    savepdf(f'{ev}vehicles_vx{pdf_ex}')
    plt.show()

    _ = plt.plot(t, y_0_dot, "r")
    _ = plt.plot(t, y_1_dot, "b")
    _ = plt.xlabel("Time (s)")
    _ = plt.ylabel("y_dot (m)")
    _ = plt.title("Y velocity paths")
    _ = plt.legend(["Vehicle 1", "Vehicle 2"])
    savepdf(f'{ev}vehicles_vy{pdf_ex}')
    plt.show()

    _ = plt.plot(J_hist)
    _ = plt.xlabel("Iteration")
    _ = plt.ylabel("Total cost")
    _ = plt.title("Total cost-to-go")
    savepdf(f'{ev}cost_to_go{pdf_ex}')
    plt.show()


if __name__ == "__main__":
    plot_rendevouz(use_linesearch=False)

# No way to see how many Qs a TA has answered on forum
# No way to automatically associate a study nr with a group.
