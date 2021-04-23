"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import gym
import numpy as np
from irlc.ex04.continuous_time_model import ensure_policy

class ContiniousTimeEnvironment(gym.Env):
    """
    Converts a discretized model into an environment. Usage:

    >>> dmod = DiscreteModel() # some sort of discrete mdoel
    >>> env = ContiniousTimeEnvironment(dmod, Tmax=5)
    >>> xp, reward, done, info = env.step(action)

    `Tmax` is the time until the environment terminates (done=True) measured in simulation time (seconds).

    This is mainly about implementing the step-function. Note the step-function
    uses the (relatively exact) RK4 method to integrate the environment over the timespan dt,
    and not the (approximate) x_{k+1} = f_k(x_k,u_k) method in the symbolic environment;
    this is because we actually want the environment to reflect what the robot does.

    We still use the variable transformations from the discrete environment though.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    def __init__(self, discrete_model, Tmax=None, supersample_trajectory=False):
        self.dt = discrete_model.dt  # Discretization time
        self.state = None            # the current state
        self.time = 0                # Current global time index
        self.discrete_model = discrete_model
        self.observation_space = discrete_model.observation_space
        self.action_space = discrete_model.action_space
        self.Tmax = Tmax
        self.state_labels = discrete_model.state_labels
        self.action_labels = discrete_model.action_labels
        self.supersample_trajectory = supersample_trajectory


    def clip_action(self, u):
        return np.clip(u, a_max=self.action_space.high, a_min=self.action_space.low)

    def step(self, u):
        u = self.clip_action(u)
        if u not in self.action_space:
            raise Exception("Action", u, "not contained in action space", self.action_space)
        # N=20 is a bit arbitrary; should probably be a parameter to the environment.
        xx, uu, tt = self.discrete_model.simulate(x0=self.state, policy=ensure_policy(u), t0=self.time, tF=self.time + self.discrete_model.dt, N=20)
        self.state = xx[-1]
        self.time = tt[-1]
        cc = [self.discrete_model.c(x, u, i=None) for x, u in zip(xx[:-1], uu[:-1])]
        done = False
        if self.time + self.discrete_model.dt/2 > self.Tmax:
            cc[-1] += self.discrete_model.cN(xx[-1])
            done = True
        metadata = {'dt': self.discrete_model.dt}  # Allow the train() function to figure out the simulation time step size
        if self.supersample_trajectory:   # This is only for nice visualizations.
            from irlc.ex01.agent import Trajectory
            traj = Trajectory(time=tt, state=xx.T, action=uu.T, reward=np.asarray(cc), env_info=[])
            metadata['supersample'] = traj # Supersample the trajectory
        reward = -sum(cc)  # To be compatible with openai gym we return the reward as -cost.
        if self.state not in self.observation_space:
            print("> state", self.state)
            print("> observation space", self.observation_space)
            raise Exception("State no longer in observation space", self.state)

        return self.state, reward, done, metadata

    def reset(self):
        self.state = self.discrete_model.reset()
        self.time = 0
        return self.state

    def render(self, mode='human', **kwargs):
        return self.discrete_model.render(x=self.state, mode=mode, **kwargs)

    @property
    def state_size(self):
        return np.prod(self.observation_space.shape)

    @property
    def action_size(self):
        return np.prod(self.action_space.low.shape)

    def close(self):
        self.discrete_model.close()
