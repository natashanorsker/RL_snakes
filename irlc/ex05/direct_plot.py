"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import matplotlib.pyplot as plt
import numpy as np
from irlc.ex04.continuous_time_model import plot_trajectory, make_space_above
from irlc import savepdf

"""
Helper function for plotting.
"""
def plot_solutions(env, solutions, animate=True, pdf=None, plot_defects=True, Ix=None, animate_repeats=1, animate_all=False):

    for k, sol in enumerate(solutions):
        grd = sol['grid']
        x_res = sol['grid']['x']
        u_res = sol['grid']['u']
        ts = sol['grid']['ts']
        u_fun = lambda x, t: sol['fun']['u'](t)
        N = len(ts)
        if pdf is not None:
            pdf_out = f"{pdf}_sol{N}"


        x_sim, u_sim, t_sim = env.simulate(x0=grd['x'][0, :], u_fun=u_fun, t0=grd['ts'][0], tF=grd['ts'][-1], N_steps=1000)
        if animate and (k == len(solutions)-1 or animate_all):
            for _ in range(animate_repeats):
                env.animate_rollout(x0=grd['x'][0, :], u_fun=u_fun, t0=grd['ts'][0], tF=grd['ts'][-1], N_steps=1000, fps=30)

        eqC_val = sol['eqC_val']
        labels = env.state_labels

        if Ix is not None:
            labels = [l for k, l in enumerate(labels) if k in Ix]
            x_res = x_res[:,np.asarray(Ix)]
            x_sim = x_sim[:,np.asarray(Ix)]

        print("Initial State: " + ",".join(labels))
        print(x_res[0])
        print("Final State:")
        print(x_res[-1])

        ax = plot_trajectory(x_res, ts, lt='ko-', labels=labels, legend="Direct state prediction $x(t)$")
        plot_trajectory(x_sim, t_sim, lt='-', ax=ax, labels=labels, legend="RK4 exact simulation")
        # plt.suptitle("State", fontsize=14, y=0.98)
        # make_space_above(ax, topmargin=0.5)

        if pdf is not None:
            savepdf(pdf_out +"_x")
        plt.show()
        print("plotting...")
        plot_trajectory(u_res, ts, lt='ko-', labels=env.action_labels, legend="Direct action prediction $u(t)$")
        print("plotting... B")
        # plt.suptitle("Action", fontsize=14, y=0.98)
        # print("plotting... C")
        # make_space_above(ax, topmargin=0.5)
        # print("plotting... D")
        if pdf is not None:
            savepdf(pdf_out +"_u")
        plt.show()
        if plot_defects:
            plot_trajectory(eqC_val, ts[:-1], lt='-', labels=labels)
            plt.suptitle("Defects (equality constraint violations)")
            if pdf is not None:
                savepdf(pdf_out +"_defects")
            plt.show()

    return x_sim, u_sim, t_sim
