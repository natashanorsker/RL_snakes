"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [Kel17] Matthew Kelly. An introduction to trajectory optimization: how to do your own direct collocation. SIAM Review, 59(4):849–904, 2017. (See kelly2017.pdf). 
"""
from scipy.optimize import Bounds
import sympy as sym
from irlc.ex04.continuous_time_model import ContiniousTimeSymbolicModel
from irlc.ex04.cost_continuous import SymbolicQRCost
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
import gym
from gym.spaces.box import Box
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
import numpy as np

"""
SEE: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
"""
class ContiniousPendulumModel(ContiniousTimeSymbolicModel): 
    state_labels= [r"$\theta$", r"$\frac{d \theta}{dt}$"]
    action_labels = ['Torque $u$']
    x_upright, x_down = np.asarray([0.0, 0.0]), np.asarray([np.pi, 0.0])

    def __init__(self, l=1., m=.8, g=9.82, friction=0.0, max_torque=6.0, simple_bounds=None, cost=None): 
        self.g, self.l, self.m = g, l, m
        self.friction=friction
        self.max_torque = max_torque
        self.action_space = Box(low=np.array([-max_torque]), high=np.array([max_torque]), dtype=float) 
        self.observation_space = Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), dtype=float) 

        if simple_bounds is None:
            simple_bounds = {'tF': Bounds([0.5], [4]), 
                             't0': Bounds([0], [0]),
                             'x': Bounds([-2 * np.pi, -np.inf], [2 * np.pi, np.inf]),
                             'u': Bounds([-max_torque], [max_torque]),
                             'x0': Bounds([np.pi, 0], [np.pi, 0]),
                             'xF': Bounds([0, 0], [0, 0])} 
        if cost is None:
            # Initialize to very basic cost compatible with (Kel17) (i.e. only account for action u)
            cost = SymbolicQRCost(R=np.ones( (1,1) ))

        self.u_prev = None                        # For rendering
        self.cp_render = gym.make("Pendulum-v0")  # environment only used for rendering
        self.cp_render.max_time_limit = 10000
        self.cp_render.reset()
        super().__init__(cost=cost, simple_bounds=simple_bounds)

    def render(self, x, mode="human"):
        self.cp_render.env.last_u = self.u_prev
        self.cp_render.env.state = np.asarray(x) # environment is wrapped
        return self.cp_render.render(mode=mode)

    def reset(self):
        return np.asarray([np.pi-0.01,0.01])

    def sym_f(self, x, u, t=None): 
        g, l, m = self.g, self.l, self.m
        theta_dot = x[1]  # Parameterization: x = [theta, theta']
        theta_dot_dot =  g/l * sym.sin(x[0]) + 1/(m*l**2) * u[0]
        return [theta_dot, theta_dot_dot] 

    def close(self):
        self.cp_render.close()

guess = {'t0': 0,
         'tF': 2.5,
         'x': [np.asarray([0, 0]), np.asarray([np.pi, 0])],
         'u': [np.asarray([0]), np.asarray([0])] }

def _pendulum_cost(model):
    from irlc.ex04.cost_discrete import goal_seeking_qr_cost, DiscreteQRCost
    Q = np.eye(model.state_size)
    Q[0, 1] = Q[1, 0] = model.l
    Q[0, 0] = Q[1, 1] = model.l ** 2
    Q[2, 2] = 0.0
    R = np.array([[0.1]]) * 10
    cost2 = goal_seeking_qr_cost(model, Q=Q, x_target=model.x_upright)
    cost2 += goal_seeking_qr_cost(model, xN_target=model.x_upright) * 1000
    cost2 += DiscreteQRCost(model, R=R)
    return cost2 * 2

class GymSinCosPendulumModel(DiscretizedModel): 
    state_labels =  ['$\sin(\theta)$', '$\cos(\theta)$', '$d\theta/dt$']
    action_labels = ['Torque $u$']

    def __init__(self, dt=0.02, cost=None, transform_actions=True, **kwargs): 
        model = ContiniousPendulumModel(**kwargs)
        self.max_torque = model.max_torque
        self.transform_actions = transform_actions
        self.observation_space = Box(low=np.array([-np.inf]*3), high=np.array([np.inf]*3),dtype=np.float)
        if transform_actions:
            self.action_space = Box(low=np.array([-np.inf]), high=np.array([np.inf]),dtype=np.float)  
        else:
            self.action_space = model.action_space

        super().__init__(model=model, dt=dt, cost=cost)

        self.x_upright = np.asarray(self.continious_states2discrete_states(model.x_upright))
        self.l = model.l
        if cost is None:
            cost = _pendulum_cost(self)
        self.cost = cost

    def sym_discrete_xu2continious_xu(self, x, u):
        sin_theta, cos_theta, theta_dot = x[0], x[1], x[2]
        torque = sym.tanh(u[0]) * self.max_torque if self.transform_actions else u[0]
        theta = sym.atan2(sin_theta, cos_theta)  # Obtain angle theta from sin(theta),cos(theta)
        return [theta, theta_dot], [torque]

    def sym_continious_xu2discrete_xu(self, x, u):
        theta, theta_dot = x[0], x[1]
        torque = sym.atanh(u[0]/self.max_torque) if self.transform_actions else u[0]
        return [sym.sin(theta), sym.cos(theta), theta_dot], [torque] 


class GymSinCosPendulumEnvironment(ContiniousTimeEnvironment): 
    def __init__(self, *args, Tmax=5, supersample_trajectory=False, transform_actions=True, **kwargs):
        discrete_model = GymSinCosPendulumModel(*args, transform_actions=transform_actions, **kwargs)
        super().__init__(discrete_model, Tmax=Tmax, supersample_trajectory=supersample_trajectory) 

    def step(self, u):
        self.discrete_model.continuous_model.u_prev = u
        return super().step(u)


if __name__ == "__main__":
    model = ContiniousPendulumModel(l=1, m=1)
    print(f"Pendulum with g={model.g}, l={model.l}, m={model.m}") 
    x = [1,2]
    u = [0] # Input state/action.
    # x_dot = ...
    # TODO: 1 lines missing.
    raise NotImplementedError("Compute dx/dt = f(x, u, t=0) here using the model-class defined above")
    # x_dot_numpy = ...
    # TODO: 1 lines missing.
    raise NotImplementedError("Compute dx/dt = f(x, u, t=0) here using numpy-expressions you write manually.")

    print(f"Using model-class: dx/dt = f(x, u, t) = {x_dot}")
    print(f"Using numpy: dx/dt = f(x, u, t) = {x_dot_numpy}") 
