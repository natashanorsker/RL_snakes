import matplotlib.pyplot as plt
from irlc.ex04.model_cartpole import ContiniousCartpole, kelly_swingup
from irlc.ex04.cost_continuous import SymbolicQRCost
from irlc.ex05.direct import direct_solver, get_opts
import numpy as np
from scipy.optimize import Bounds
from irlc import savepdf

def make_cartpole_kelly17():
    """
    Creates Cartpole problem. Details about the cost function can be found in \cite[Section 6]{kelly17}
    and details about the physical parameters can be found in \cite[Appendix E, table 3]{kelly17}.
    """
    # this will generate a different carpole environment with an emphasis on applying little force u.
    duration = 2.0
    # TODO: 2 lines missing.
    raise NotImplementedError("")
    # Initialize the cost-function above. You should do so by using a call of the form:
    # cost = SymbolicQRCost(Q=..., R=...) # Take values from Kelly
    # The values of Q, R can be taken from the paper.

    _, bounds, _ = kelly_swingup(maxForce=20, dist=1.0) # get a basic version of the bounds (then update them below).
    # TODO: 1 lines missing.
    raise NotImplementedError("Update the bounds so the problem will take exactly tF=2 seconds.")

    # Instantiate the environment as a ContiniousCartpole environment. The call should be of the form:
    # env = ContiniousCartpole(...)
    # Make sure you supply all relevant physical constants (maxForce, mp, mc, l) as well as the cost and bounds. Check the
    # ContiniousCartpole class definition for details.
    # TODO: 1 lines missing.
    raise NotImplementedError("")
    guess = env.guess()
    guess['tF'] = duration # Our guess should match the constraints.
    return env, guess

def compute_solutions():
    env, guess = make_cartpole_kelly17()
    print("cartpole mp", env.mp)
    options = [get_opts(N=10, ftol=1e-3, guess=guess),
               get_opts(N=40, ftol=1e-6)]
    solutions = direct_solver(env, options)
    return env, solutions

def direct_cartpole():
    env, solutions = compute_solutions()
    from irlc.ex05.direct_plot import plot_solutions
    print("Did we succeed?", solutions[-1]['solver']['success'])
    plot_solutions(env, solutions, animate=True, pdf="direct_cartpole_force")
    env.close()

if __name__ == "__main__":
    direct_cartpole()
