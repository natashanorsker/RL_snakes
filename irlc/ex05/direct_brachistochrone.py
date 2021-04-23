"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
import matplotlib.pyplot as plt
from irlc import savepdf
from irlc.ex05.model_brachistochrone import ContiniouBrachistochrone, brachistochrone
from irlc.ex05.direct import direct_solver, get_opts
from irlc.ex05.direct_plot import plot_solutions

def plot_brachistochrone_solutions(env,solutions, out=None):
    plot_solutions(env, solutions, animate=False, pdf=out)
    for index, sol in enumerate(solutions):
        x_res = sol['grid']['x']
        plt.figure(figsize=(5,5))
        plt.plot( x_res[:,0], x_res[:,1])
        bounds = env.simple_bounds()
        xF = bounds['xF'].lb
        plt.plot([0, 0], [0, xF[1]], 'r-')
        plt.plot([0, xF[0]], [xF[1], xF[1]], 'r-')
        # plt.title("Curve in x/y plane")
        plt.xlabel("$x$-position")
        plt.ylabel("$y$-position")

        if env.h is not None:
            # add dynamical constraint.
            xc = np.linspace(0, env.x_dist)
            yc = -xc/2 - env.h
            plt.plot(xc, yc, 'k-', linewidth=2)
        plt.grid()
        # plt.gca().invert_yaxis()
        plt.gca().axis('equal')
        if out:
            savepdf(f"{out}_{index}")
        plt.show()
    pass

def compute_unconstrained_solutions():
    env = ContiniouBrachistochrone(h=None, x_dist=1)
    _, _, guess = brachistochrone(x_B=1) # to obtain a guess
    options = [get_opts(N=10, ftol=1e-3, guess=guess),
               get_opts(N=30, ftol=1e-6)]
    # solve without constraints
    solutions = direct_solver(env, options)
    return env, solutions

def compute_constrained_solutions():
    env_h = ContiniouBrachistochrone(h=0.1, x_dist=1)
    _, _, guess = brachistochrone(x_B=1)  # to obtain a guess
    options = [get_opts(N=10, ftol=1e-3, guess=guess),
               get_opts(N=30, ftol=1e-6)]
    solutions_h = direct_solver(env_h, options)
    return env_h, solutions_h

if __name__ == "__main__":
    """ 
    For further information see:
    http://www.hep.caltech.edu/~fcp/math/variationalCalculus/variationalCalculus.pdf
    """
    env, solutions = compute_unconstrained_solutions()
    #
    # env = ContiniouBrachistochrone(h=None, x_dist=1)
    # _, _, guess = brachistochrone(x_B=1) # to obtain a guess
    # options = [get_opts(N=10, ftol=1e-3, guess=guess),
    #            get_opts(N=30, ftol=1e-6)]
    #
    # # solve without constraints
    # solutions = direct_solver(env, options)
    plot_brachistochrone_solutions(env, solutions[-1:], out="brachi")

    # solve with dynamical (sloped planc) constraint at height of h.
    env_h, solutions_h = compute_constrained_solutions()
    plot_brachistochrone_solutions(env_h, solutions_h[-1:], out="brachi_h")
