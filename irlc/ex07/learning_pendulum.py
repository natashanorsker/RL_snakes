"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment
from irlc.utils.video_monitor import VideoMonitor
from irlc.ex04.cost_discrete import goal_seeking_qr_cost, DiscreteQRCost
from irlc.ex01.agent import train
import numpy as np
import matplotlib.pyplot as plt
from irlc.ex07.lqr_learning_agents import MPCLocalLearningLQRAgent, MPCLearningAgentLocalOptimize

def main_pendulum():
    env_pendulum = GymSinCosPendulumEnvironment(Tmax=5, dt=0.08, transform_actions=False, max_torque=7)
    mk_new_cost = 1
    if mk_new_cost == 1:
        model = env_pendulum.discrete_model
        Q = np.eye(model.state_size)
        Q[0,0] = 0
        Q[2,2] = 0.05
        R = np.eye(1)*0.01
        cost2 = goal_seeking_qr_cost(model, Q=Q, x_target=model.x_upright)
        # cost2 += goal_seeking_qr_cost(model, xN_target=model.x_upright) * 1000
        cost2 += DiscreteQRCost(model, R=R)
        model.cost = cost2
    elif mk_new_cost == 2:
        model = env_pendulum.discrete_model
        Q = np.eye(model.state_size)
        Q = Q * 0
        Q[1, 1] = 1.0  # up-coordinate
        q = np.zeros((model.state_size,))
        q[1] = -1
        cost2 = goal_seeking_qr_cost(model, Q=Q, x_target=model.x_upright) * 1
        cost2 += DiscreteQRCost(model, Q=np.eye(model.state_size), R=np.eye(model.action_size)) * 0.1 * 0.1
        cost2 += DiscreteQRCost(model, R=np.eye(model.action_size)) * 1 * 0
        model.cost = cost2


    agent = MPCLocalLearningLQRAgent(env_pendulum, horizon_length=18, neighbourhood_size=100, min_buffer_size=300)
    # agent = MPCLearningAgentLocalOptimize(env_pendulum, horizon_length=18, neighbourhood_size=100, min_buffer_size=300)
    env_pendulum = VideoMonitor(env_pendulum)

    stats, trajectories = train(env_pendulum, agent, num_episodes=40,return_trajectory=True, temporal_policy=True)
    tt =  [agent._t_nearest, agent._t_linearizer, agent._t_solver]

    print("Nearest, linear, solver: ", [t/sum(tt) for t in tt])
    for k, t in enumerate(trajectories):
        if (k+1)%5 != 0:
            continue
        plt.plot(t.time, t.state[:,0], label='sin(theta)')
        plt.plot(t.time, t.state[:,1], label='cos(theta)')
        plt.plot(t.time, t.state[:,2], label='d theta/dt')
        plt.plot(t.time[:-1], t.action[:, 0], label='action u')
        plt.show()
    env_pendulum.close()

if __name__ == "__main__":
    main_pendulum()
