"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import matplotlib.pyplot as plt
import numpy as np


def get_by_ace(V,ace=False):
    dd = V.copy()
    dd.clear()
    for (p,d,ac),val in V.items():
        if ac == ace:
            dd[ (p,d)] = val
    return dd

def plot_surface_2(X,Y,Z,fig=None, ax=None, **kwargs):
    if fig is None and ax is None:
        fig = plt.figure(figsize=(20, 10))
    if ax is None:
        ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=1, edgecolors='k', **kwargs)
    ax.view_init(ax.elev, -120)
    if fig is not None:
        fig.colorbar(surf, shrink=0.5, aspect=5)
    return ax

def to_matrix(V):
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    Z_ace = np.zeros_like(X,dtype=np.float)
    for j,(x, y) in enumerate( zip( X.flat, Y.flat)):
        Z_ace.flat[j] = V[(x,y)]
    return X, Y, Z_ace

def plot_blackjack_value(V, title="Value Function", pdf_out=None):
    """
    Plots the value function as a surface plot.
    """
    for lbl, ac in zip(["Usable ace", "No usable ace"], [True, False]):
        w = get_by_ace(V,ace=ac)
        X,Y,Z = to_matrix(w)
        ax = plot_surface_2(X, Y, Z)
        ax.set_zlabel("Value")
        ax.set_title(title)
        if pdf_out is not None:
            savepdf(pdf_out+"_"+lbl.replace(" ", "_"))

def plot_blackjack_policy(V, title):
    plt.figure(figsize=(18, 12))
    for lbl, ac in zip(["Usable ace", "No usable ace"], [True, False]):
        w = get_by_ace(V,ace=ac)
        X, Y, Z = to_matrix(w)
        plt.subplot(1,2,1+ac)
        plt.imshow(Z.T)
        plt.title(f"{title} ({lbl})")
        plt.gca().invert_yaxis()
        plt.ylabel('Player Sum')
        plt.xlabel('Dealer Showing')
        plt.colorbar()

def policy20(s): 
    # TODO: 1 lines missing.
    raise NotImplementedError("Implement the rule where we stick if we have a score of 20 or more.")

if __name__ == "__main__":
    from irlc.ex10.mc_evaluate import MCEvaluationAgent
    from irlc.ex01.agent import train
    import gym
    from irlc import main_plot, savepdf

    nenv = "Blackjack-v0"
    env = gym.make(nenv)
    episodes = 50000
    gamma = 1
    experiment = f"experiments/{nenv}_first_{episodes}"
    """ Instantiate the agent and call the training method here. Make sure to pass the policy=policy20 function to the MCEvaluationAgent
     and set gamma=1. """
    # TODO: 2 lines missing.
    raise NotImplementedError("")
    main_plot(experiment, smoothing_window=episodes//100, resample_ticks=200)
    plt.ylim([-0.5, 0])
    plt.title("Blackjack using first-visit MC")
    savepdf("blackjack_stick20_first")
    plt.show()

    pdf = "blackjack_stick20_valuefun"
    plot_blackjack_value(agent.v, title="MC first-visit value function", pdf_out=pdf)
    savepdf("blackjack_stick20_valuefun")
    plt.show()
