"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from irlc import savepdf
from irlc.ex01.agent import train
from irlc.ex12.semi_grad_nstep_sarsa import LinearSemiGradSarsaN
import gym
from irlc import main_plot
from irlc.ex10.mc_evaluate_blackjack import plot_surface_2
from irlc.ex12.semi_grad_sarsa_lambda import LinearSemiGradSarsa


def plot_mountaincar_value_function(env, value_function, ax):
    """
    3d plot
    """
    grid_size = 40
    low = env.unwrapped.observation_space.low
    high = env.unwrapped.observation_space.high
    X,Y = np.meshgrid( np.linspace(low[0], high[0], grid_size), np.linspace(low[1], high[1], grid_size)  )
    Z = X*0
    for i, (x,y) in enumerate(zip(X.flat, Y.flat)):
        Z.flat[i] = value_function([x,y])

    plot_surface_2(X,Y,Z,ax=ax)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go')

def figure_10_1():
    episodes = 9000
    plot_episodes = [1, 99, episodes - 1]
    scale = 8
    fig = plt.figure(figsize=(4*scale, scale))
    axes = [fig.add_subplot(1, len(plot_episodes), i+1, projection='3d') for i in range(len(plot_episodes))]
    num_of_tilings = 8
    alpha = 0.3

    env = gym.make("MountainCar-v0")
    agent = LinearSemiGradSarsa(env, gamma=1, alpha=alpha/num_of_tilings, epsilon=0)
    for ep in tqdm(range(episodes)):
        train(env, agent, num_episodes=1, max_steps=np.inf, verbose=False)
        if ep in plot_episodes:
            v = lambda s: -agent.v(s)
            ax = axes[plot_episodes.index(ep)]
            plot_mountaincar_value_function(env, v, ax=ax)
            ax.set_title('Episode %d' % (ep + 1))

    from irlc import savepdf
    savepdf("semigrad_sarsa_10-1")
    plt.show()

def figure_10_2():
    episodes = 500
    num_of_tilings = 8
    alphas = [0.1, 0.2, 0.5]
    env = gym.make("MountainCar500-v0")

    experiments = []
    for alpha in alphas:
        agent = LinearSemiGradSarsa(env, gamma=1, alpha=alpha / num_of_tilings, epsilon=0)
        experiment = f"experiments/mountaincar_10-2_{agent}_{episodes}"
        train(env, agent, experiment_name=experiment, num_episodes=episodes,max_runs=10)
        experiments.append(experiment)

    main_plot(experiments=experiments, y_key="Length")
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.title(env.spec._env_name + " - Semigrad Sarsa - Figure 10.2")
    savepdf("mountaincar_10-2")
    plt.show()

def figure_10_3():
    from irlc.ex12.semi_grad_sarsa_lambda import LinearSemiGradSarsaLambda
    from irlc.ex11.semi_grad_q import LinearSemiGradQAgent

    max_runs = 10
    episodes = 500
    num_of_tilings = 8
    alphas = [0.5, 0.3]
    n_steps = [1, 8]

    env = gym.make("MountainCar500-v0")
    experiments = []

    """ Plot results of experiments here. """
    # TODO: 16 lines missing.
    raise NotImplementedError("")

    main_plot(experiments=experiments, y_key="Length")
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.title(env.spec._env_name + " - Semigrad N-step Sarsa - Figure 10.3")
    savepdf("mountaincar_10-3")
    plt.show()

def figure_10_4():
    from irlc import log_time_series
    alphas = np.arange(0.25, 1.75, 0.25)
    n_steps = np.power(2, np.arange(0, 5))
    episodes = 50
    env = gym.make("MountainCar500-v0")
    experiments = []
    num_of_tilings = 8
    max_asteps = 500
    run = True
    for n_step_index, n_step in enumerate(n_steps):
        aexp = []
        did_run = False
        for alpha_index, alpha in enumerate(alphas):
            if not run:
                continue
            if (n_step == 8 and alpha > 1) or (n_step == 16 and alpha > 0.75):
                # In these cases it won't converge, so ignore them
                asteps = max_asteps #max_steps * episodes
            else:
                n = n_step
                agent = LinearSemiGradSarsaN(env, gamma=1, alpha=alpha / num_of_tilings, epsilon=0, n=n)
                _, stats, _ = train(env, agent, num_episodes=episodes)
                asteps = np.mean( [s['Length'] for s in stats] )
                did_run = did_run or stats is not None

            aexp.append({'alpha': alpha, 'average_steps': asteps})

        experiment = f"experiments/mc_10-4_lsgn_{n_step}"
        experiments.append(experiment)
        if did_run:
            log_time_series(experiment, aexp)

    main_plot(experiments, x_key="alpha", y_key="average_steps", ci=None)
    plt.xlabel('alpha')
    plt.ylabel('Steps per episode')
    plt.title("Figure 10.4: Semigrad n-step Sarsa on mountain car")
    plt.ylim([150, 300])
    savepdf("mountaincar_10-4")
    plt.show()

if __name__ == '__main__':
    # figure_10_1()
    # figure_10_2()
    figure_10_3()
    # figure_10_4()
