"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
from irlc import savepdf
from irlc.ex04.pid import PID
from irlc import Agent

class PIDCarAgent(Agent):
    def __init__(self, env, v_target=0.5, use_both_x5_x3=True):
        """
        Define two pid controllers: One for the angle, the other for the velocity.

        self.pid_angle = PID(dt=self.discrete_model.dt, Kp=x, ...)
        self.pid_velocity = PID(dt=self.discrete_model.dt, Kp=z, ...)

        I did not use Kd/Ki, however you need to think a little about the targets.
        """
        # self.pid_angle = ...
        # TODO: 2 lines missing.
        raise NotImplementedError("Define PID controllers here.")
        self.use_both_x5_x3 = use_both_x5_x3 # Using both x3+x5 seems to make it a little easier to get a quick lap time, but you can just use x5 to begin with.
        super().__init__(env)

    def pi(self, x, t=None):
        # Call PID controller. The steering angle controller can either just use x5, use both x5 and x3 as input.
        # TODO: 2 lines missing.
        raise NotImplementedError("Compute action here. No clipping necesary.")
        return u


if __name__ == "__main__":
    from irlc.ex01.agent import train
    from irlc.utils.video_monitor import VideoMonitor
    from irlc.car.sym_car_model import CarEnvironment
    import matplotlib.pyplot as plt

    env = CarEnvironment(noise_scale=0,Tmax=30, max_laps=1)
    env = VideoMonitor(env)
    agent = PIDCarAgent(env, v_target=1.0)

    stats, trajectories = train(env, agent, num_episodes=1, return_trajectory=True)
    env.close()
    t = trajectories[0]
    plt.clf()
    plt.plot(t.state[:,0], label="velocity" )
    plt.plot(t.state[:,5], label="s (distance to center)" )
    plt.xlabel("Time/seconds")
    plt.legend()
    savepdf("pid_car_agent")
    plt.show()
