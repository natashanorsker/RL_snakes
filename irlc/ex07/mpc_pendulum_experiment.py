"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment
from irlc.utils.video_monitor import VideoMonitor
from irlc.ex04.cost_discrete import goal_seeking_qr_cost, DiscreteQRCost
from irlc.ex01.agent import train
import numpy as np
from irlc.ex07.lqr_learning_agents import MPCLearningAgentLocalOptimize, MPCLocalLearningLQRAgent
from irlc import plot_trajectory, main_plot
import matplotlib.pyplot as plt


def mk_mpc_pendulum_env(Tmax=10):
    """
    Initialize pendulum model suitable for MPC learning.

    If you try to replicate this experiment for another environment, please note I had to tweak the
    parameters quite a lot to make things work.
    """
    env_pendulum = GymSinCosPendulumEnvironment(Tmax=Tmax, dt=0.08, transform_actions=False)
    model = env_pendulum.discrete_model
    Q = np.eye(model.state_size)
    Q = Q * 0
    Q[1, 1] = 1.0  # up-coordinate
    q = np.zeros((model.state_size,))
    q[1] = -1
    cost2 = goal_seeking_qr_cost(model, Q=Q, x_target=model.x_upright) * 1
    cost2 += DiscreteQRCost(model, Q=np.eye(model.state_size), R=np.eye(model.action_size)) * 0.03
    model.cost = cost2
    return env_pendulum
L = 12

def main_pendulum():
    env_pendulum = mk_mpc_pendulum_env(Tmax=10)
    """ Run Local Optimization/MPC agent using the parameters
    L = 12 
    neighboorhood_size=50
    min_buffer_size=50 
    """
    agent = MPCLocalLearningLQRAgent(env_pendulum, horizon_length=L, neighbourhood_size=50, min_buffer_size=50)
    env_pendulum = VideoMonitor(env_pendulum)

    experiment_name = f"pendulum{L}"
    stats, trajectories = train(env_pendulum, agent, experiment_name=experiment_name, num_episodes=16,return_trajectory=True)
    plt.show()
    for k in range(len(trajectories)):
        plot_trajectory(trajectories[k], env_pendulum)
        plt.title(f"Trajectory {k}")
        plt.show()

    env_pendulum.close()
    main_plot(experiment_name)
    plt.show()

if __name__ == "__main__":
    main_pendulum()


