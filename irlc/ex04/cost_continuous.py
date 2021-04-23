"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import sympy as sym
from irlc.ex04.cost_discrete import DiscreteQRCost
import numpy as np

class SymbolicMayerLagrangeCost:
    """
    Symbolic MayerLagrange cost function. The environment is assumed to terminate at time t=t_f
    and have cost:

    J(x_0, t_0) = cf(x(t_f), t_f) + \int_{t_0}^{t_F} c(x(t), u(t), t) dt

    (if the environment does not terminate, simply let cf=0). We specify these as symbolic expressions
    to allow us to compute derivatives later.
    """
    def sym_cf(self, t0, tF, x0, xF):
        # compute Mayer term
        raise NotImplementedError()

    def sym_c(self, x, u, t):
        # compute Lagrange term
        raise NotImplementedError()

def mat(x):
    return sym.Matrix(x) if x is not None else x

def sym2np(x):
    if x is None:
        return x
    f = sym.lambdify([], x)
    return f()

class SymbolicQRCost(SymbolicMayerLagrangeCost):
    def __init__(self, Q=None, R=None, x_target=None, c=0, xF_linear=None, x_linear=None):
        self.Q = sym.Matrix(Q) if Q is not None else Q
        self.R = sym.Matrix(R) if R is not None else R
        self.x_target = mat(x_target)
        self.x_linear = mat(x_linear)
        self.xF_linear = sym.Matrix(xF_linear) if xF_linear is not None else xF_linear
        self.c = c
        # as numpy arrays for fast access
        self.Q_np = sym2np(self.Q)
        self.R_np = sym2np(self.R)
        self.c_np = sym2np(self.c)
        self.x_target_np = sym2np(self.x_target)
        self.x_linear_np = sym2np(self.x_linear)

    def sym_cf(self, t0, tF, x0, xF):
        sxF = sym.Matrix(xF)
        J = sym.Matrix( [[0.0]] )

        if self.xF_linear is not None:
            J = J + sxF.transpose() @ self.xF_linear
        return J[0,0]

    def sym_c(self, x, u, t):
        '''
        Implements:

        w = 0.5 * ((x-xt)' * Q * (x-xt) + u' * R * u) + c

        '''
        um = sym.Matrix(u)
        xm = sym.Matrix(x)

        w = sym.Matrix( [[0.0]] ) + sym.Matrix([[self.c]])
        if self.x_target is not None:
            xm = xm - self.x_target

        if self.R is not None:
            w += 0.5 * um.transpose() @ self.R @ um
        if self.Q is not None:
            w += 0.5 * xm.transpose() @ self.Q @ xm
        w = w[0,0]
        return w

    def discretize(self, env, dt):
        """ Discreteize the cost function. Note not all terms are discretized; it is good enough for this course, but
        it would be worth re-visiting it later if the examples are extended. """
        n = lambda x: np.asarray(x.tolist(),dtype=np.float)*dt if x is not None else None
        return DiscreteQRCost(env=env, Q=n(self.Q), R=n(self.R), qc=self.c_np)
