"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
from irlc.ex04.model_linear_quadratic import LinearQuadraticModel

R = np.eye(2)
class ContiniousBoingModel(LinearQuadraticModel):
    """
    Boing 747 level flight example.

    See: https://books.google.dk/books?id=tXZDAAAAQBAJ&pg=PA147&lpg=PA147&dq=boeing+747+flight+0.322+model+longitudinal+flight&source=bl&ots=L2RpjCAWiZ&sig=ACfU3U2m0JsiHmUorwyq5REcOj2nlxZkuA&hl=en&sa=X&ved=2ahUKEwir7L3i6o3qAhWpl4sKHQV6CdcQ6AEwAHoECAoQAQ#v=onepage&q=boeing%20747%20flight%200.322%20model%20longitudinal%20flight&f=false
    Also: https://web.stanford.edu/~boyd/vmls/vmls-slides.pdf
    """
    state_labels = ["Longitudinal velocity (x) ft/sec", "Velocity in y-axis ft/sec", "Angular velocity", "angle wrt. horizontal"]
    action_labels = ['Elevator', "Throttle"]
    observation_labels = ["Airspeed", "Climb rate"]
    def __init__(self):
        A = [[-0.003, 0.039, 0, -0.322],
             [-0.065, -.319, 7.74, 0],
             [.02, -.101, -0.429, 0],
             [0, 0, 1, 0]]
        B = [[.01, 1],
             [-.18, -.04],
             [-1.16, .598],
             [0, 0]]

        A, B = np.asarray(A), np.asarray(B)
        self.u0 = 7.74  # speed in hundred feet/seconds
        self.P = np.asarray( [[1, 0, 0, 0], [0, -1, 0, 7.74]])  # Projection of state into airspeed
        self.Q_obs = np.eye(2)
        Q = self.P.T @ self.Q_obs @ self.P
        super().__init__(A=A,B=B,Q=Q,R=R)

    def state2outputs(self, x):
        return self.P @ x

class DiscreteBoingModel(DiscretizedModel):
    def __init__(self):
        model = ContiniousBoingModel()
        dt = 0.1
        self.observation_space = model.observation_space
        self.action_space = model.action_space
        cost = model.cost.discretize(self, dt=dt)
        super().__init__(model=model, dt=dt, cost=cost)


class BoingEnvironment(ContiniousTimeEnvironment):
    @property
    def observation_labels(self):
        return self.discrete_model.continuous_model.observation_labels

    def __init__(self, Tmax=10, output=None, **kwargs):
        """
        output is the desired end-state of the aircraft in the form: [airspeed, climb-rate].
        Try e.g. output=[10, 0]
        """
        model = DiscreteBoingModel()
        cmod = model.continuous_model  # Get continuous model
        if output is not None:
            # output = [10, 0]  # Desired output = env.P @ x (x: state).
            model.cost.q = -np.asarray(output) @ cmod.Q_obs @ cmod.P
            model.cost.Q = cmod.P.T @ cmod.Q_obs @ cmod.P
            model.cost.R = np.eye(2)

        self.dt = model.dt
        super().__init__(discrete_model=model, Tmax=Tmax, **kwargs)
