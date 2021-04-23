"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
import numpy as np
from irlc.ex04.model_linear_quadratic import LinearQuadraticModel

"""
SEE: https://github.com/anassinator/ilqr/blob/master/examples/rendezvous.ipynb
"""
class ContiniousRendevouzModel(LinearQuadraticModel): 
    state_labels= ["x0", "y0", "x1", "y1", 'Vx0', "Vy0", "Vx1", "Vy1"]
    action_labels = ['Fx0', 'Fy0', "Fx1", "Fy1"]
    x0 = np.array([0, 0, 10, 10, 0, -5, 5, 0])  # Initial state.

    def __init__(self, m=10.0, alpha=0.1, simple_bounds=None, cost=None): 
        m00 = np.zeros((4,4))
        mI = np.eye(4)

        A = np.block( [ [m00, mI], [m00, -alpha/m*mI] ] )
        B = np.block( [ [m00], [mI/m]] )

        state_size = len(self.x0)
        action_size = 4

        self.m = m
        self.alpha = alpha

        if simple_bounds is None:
            """ 
            simple_bounds = {'tF': Bounds([0.5], [2.5]), 
                             't0': Bounds([0], [0]),
                             'x': Bounds([-2 * np.pi, -np.inf], [2 * np.pi, np.inf]),
                             'u': Bounds([-max_torque], [max_torque]),
                             'x0': Bounds([np.pi, 0], [np.pi, 0]),
                             'xF': Bounds([0, 0], [0, 0])} 
            """
        Q = np.eye(state_size)
        Q[0, 2] = Q[2, 0] = -1
        Q[1, 3] = Q[3, 1] = -1
        R = 0.1 * np.eye(action_size)
        super().__init__(A=A, B=B, Q=Q*20, R=R*20)

    def render(self, x, mode="human"):
        pass # When there is time...
        return None

    def reset(self):
        return self.x0

    def close(self):
        pass


class DiscreteRendevouzModel(DiscretizedModel): 
    def __init__(self, dt=0.1, cost=None, transform_actions=True, **kwargs):
        model = ContiniousRendevouzModel(**kwargs)
        super().__init__(model=model, dt=dt, cost=cost) 

class RendevouzEnvironment(ContiniousTimeEnvironment): 
    def __init__(self, Tmax=20, **kwargs):
        discrete_model = DiscreteRendevouzModel(**kwargs)
        super().__init__(discrete_model, Tmax=Tmax) 
