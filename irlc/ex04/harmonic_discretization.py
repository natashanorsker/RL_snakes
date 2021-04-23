"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.

"""
from irlc.ex04.model_harmonic import HarmonicOscilatorEnvironment
import numpy as np
import matplotlib.pyplot as plt
from irlc import savepdf

def solve_harmonic(cont_model, d=1, tF=10):
    m = cont_model.m
    k = cont_model.k
    omega = np.sqrt(k / m)
    c1 = d
    c2 = 0
    ts = np.linspace(0, tF, 400)
    xs = c1 * np.cos(omega * ts) + c2 * np.sin(omega * ts)
    return ts, xs

def setax():
    plt.xlabel('Seconds'), plt.ylabel('$x$-position')
    plt.title(f"Simulation of Harmonic Oscillator")

k, m, T = 0.1, 2, 100  # Parameters for oscillator.
def partA_simulation():
    """  Setup environment and obtain the path of the true solution """
    env = HarmonicOscilatorEnvironment(dt=2, k=k, m=m, discretization_method='euler')
    model = env.discrete_model.continuous_model
    x0 = env.reset()
    N = 100
    ts, x_true = solve_harmonic(model, d=x0[0], tF=100)

    """
    Part 1: Obtain simulation paths using Euler and RK4 integration. See (Her21, Section 8.1)
    """
    N = 500
    plt.plot(ts, x_true, 'k-', label="True solution")
    x, u, tt = model.simulate(x0=x0, u_fun=0, t0=0, tF=T, N_steps=N, method='euler')
    plt.plot(tt, x[:, 0], '-', label=f"Euler integration, $N={N}$")

    N = 40
    x, u, tt = model.simulate(x0=x0, u_fun=0, t0=0, tF=T, N_steps=N, method='rk4') # RK4 is the default
    plt.plot(tt, x[:, 0], 'o-', label=f"RK4 integration, $N={N}$")
    setax()
    plt.legend()
    savepdf("harmonicB.pdf")
    plt.show()

def partB_discretization():
    """
    Part 2: Comparing Euler discretization and Exponential discretization for the Harmonic Oscillator.
    See (Her21, Subsection 8.2.6), and keep in mind that the following is the level of precision that your 
    discrete planning method will experience. We will therefore use the
    
    > model.f_discrete(x, u) 
    
    method which corresponds to f_k(x_k, u_k).
    """

    plt.figure()
    N = 20
    env = HarmonicOscilatorEnvironment(dt=T / N, k=k, m=m)  # By default, the linear models will default to exponential discretization.
    dmodel = env.discrete_model  # Get the discrete model; this is what we will use for planning.
    x_ei = [dmodel.reset()]  # Get starting state
    steps = N
    u = [0]  # No action
    for i in range(N):
        x_ei.append(dmodel.f_discrete(x_ei[-1], u))
    x_ei = np.stack(x_ei)

    # Plotting:
    ts, x_true = solve_harmonic(dmodel.continuous_model, d=dmodel.reset()[0], tF=100)
    plt.plot(ts, x_true, 'k-', label="True solution")
    plt.plot(np.linspace(0, T, x_ei.shape[0]), x_ei[:, 0], 'o', label=f"using $f_k$ (Exponential integrator), $N={steps}$")

    # Same as above, but force system to use the Euler discretization
    N = 500
    env = HarmonicOscilatorEnvironment(dt=T / N, k=k, m=m, discretization_method='euler', Tmax=100)  # By default, the linear models will default to exponential discretization.
    dmodel = env.discrete_model  # Get the discrete model; this is what we will use for planning.
    x_euler = [dmodel.reset()]  # Get starting state
    steps = N
    u = [0]  # No action
    for i in range(N):
        x_euler.append(dmodel.f_discrete(x_euler[-1], u))
    x_euler = np.stack(x_euler)
    plt.plot(np.linspace(0, T, x_euler.shape[0]), x_euler[:, 0], '-',label=f"Using Euler discretization, $N={steps}$")
    setax()
    plt.legend()
    savepdf("harmonicC.pdf")
    plt.show()

if __name__ == "__main__":
    partA_simulation()
    partB_discretization()
