"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc.ex06.dlqr import LQR
from irlc import Agent

class DiscreteLQRAgent(Agent):
    def __init__(self, env, model):
        self.model = model
        N = int(env.Tmax / env.dt) # Obtain the planning horizon
        """ Define A, B as the list of A/B matrices here. I.e. x[t+1] = A x[t] + B x[t].
        You should use the function model.f to do this, which has build-in functionality to compute Jacobians which will be equal to A, B """
        # TODO: 1 lines missing.
        raise NotImplementedError("")
        Q, q, R = self.model.cost.Q, self.model.cost.q, self.model.cost.R
        """ Define self.L, self.l here as the (lists of) control matrices. """
        # TODO: 1 lines missing.
        raise NotImplementedError("")
        self.dt = env.dt
        super().__init__(env)

    def pi(self,x, t=None):
        """
        Compute the action here using u = L_k x + l_k.
        You should use self.L, self.l to get the control matrices (i.e. L_k = self.L[k] ),
        but you have to compute k from t and the environment's discretization time. I.e. t will be a float, and k should be an int.
        """
        # TODO: 2 lines missing.
        raise NotImplementedError("Compute current action here")
        return u
