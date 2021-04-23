"""
Quadratic cost functions
"""
import numpy as np
from irlc.ex04.continuous_time_discretized_model import symv
import sympy as sym
from sympy import hessian, derive_by_array

class DiscreteCost:
    """
    Discrete Cost. At this point this class only acts as an interface since I only have one implementation

    The cost function has the form:


    """
    def c(self, x, u, i=None):
        """
        Compute instantaneous cost function and derivatives up to second order

        Returns:
            Instantaneous cost (scalar) and higher order derivatives
        """
        raise NotImplementedError

    def cN(self, x):
        """
        Terminal cost terms + derivatives.
        """
        raise NotImplementedError()

class BasicDiscretizeCost:
    def __init__(self, env, continious_cost):
        self.ccost = continious_cost
        self.env = env

        u = symv("u", env.action_size)
        x = symv('x', env.state_size)
        xc, uc = env.sym_discrete_xu2continious_xu(x, u)
        z = list(x) + list(u)

        # create the two cost functions
        w = continious_cost.sym_cf(xc, uc, 0) * env.dt
        self.c = sym.lambdify((tuple(x), tuple(u)), w, 'numpy')
        self.c_z = sym.lambdify((tuple(x), tuple(u)), derive_by_array(w, z), 'numpy')
        self.c_zz = sym.lambdify((tuple(x), tuple(u)),hessian(w, z), 'numpy')

        J = continious_cost.sym_c(x0=None, t0=None, xF=xc, tF=0)
        self.J = sym.lambdify(tuple(x), J, 'numpy')
        self.J_x = sym.lambdify(tuple(x), derive_by_array(J, x), 'numpy')
        self.J_xx = sym.lambdify(tuple(x), hessian(J, x), 'numpy')

    def g(self, x, u, i=None):
        c = self.c(x, u)
        c_z = self.c_z(x, u)
        n = self.env.state_size
        c_zz = self.c_zz(x, u)
        c_ux, c_uu = c_zz[n:n:], c_zz[n:, n:]
        return c, c_z[:n], c_z[n:], c_zz[:n,:n], c_ux, c_uu

    def gN(self, x):
        J = self.J(x)
        J_x = self.J_x(x)
        J_xx = self.J_xx(x)
        return J, J_x, J_xx

def nz(X,a,b=None):
    return np.zeros((a,) if b is None else (a,b)) if X is None else X

class DiscreteQRCost(DiscreteCost):
    """
    The cost function has the form

    > J = 1/2 x^T Q x + 1/2 u^T R u + u^T H x  + q^T x + r^T u + qc

    Plus a terminal term of the form:

    > J = 1/2 x^T QN x + qN^T x + qcN

    """
    def __init__(self, env=None, state_size=-1, action_size=-1, Q=None,R=None,H=None,q=None,r=None,qc=0, QN=None, qN=None,qcN=0):
        if env is not None:
            n, m = env.state_size, env.action_size
        else:
            n, m = state_size, action_size
        self.env = env
        self.QN, self.qN = nz(QN,n,n), nz(qN,n)
        self.Q, self.q = nz(Q, n, n), nz(q, n)
        self.R, self.H, self.r = nz(R, m, m), nz(H, m, n), nz(r, m)
        self.qc, self.qcN = qc, qcN
        self.flds = {'QN', 'qN', 'Q', 'q', 'R', 'H', 'r', 'qcN', 'qc'}

    def c(self, x, u, i=None, compute_derivatives=True):
        c = 1/2 * (x @ self.Q @ x) + 1/2 * (u @ self.R @ u) + u @ self.H @ x + self.q @ x + self.r @ u + self.qc
        c_x = 1/2 * (self.Q + self.Q.T) @ x + self.q
        c_u = 1 / 2 * (self.R + self.R.T) @ u + self.r
        c_ux = self.H
        c_xx = self.Q
        c_uu = self.R
        if compute_derivatives:
            # this is useful for MPC when we apply an optimizer rather than LQR (iLQR)
            return c, c_x, c_u, c_xx, c_ux, c_uu
        else:
            return c

    def cN(self, x):
        J = 1/2 * (x @ self.QN @ x) + self.qN @ x + self.qcN
        J_x = 1 / 2 * (self.QN + self.QN.T) @ x + self.qN
        return J, J_x, self.QN

    def __add__(self, c):
        return DiscreteQRCost(env=self.env, **{k: self.__dict__[k] + c.__dict__[k] for k in self.flds})

    def __mul__(self, c):
        return DiscreteQRCost(env=self.env, **{k: self.__dict__[k] * c for k in self.flds})

def targ2matrices(t, Q=None):
    """
    1/2 * (x - t)**2 = 1/2 * x' * x + 1/2 * t' * t - x * t
    """
    n = t.size
    if Q is None:
        Q = np.eye(n)
    return Q, -1/2 * (Q @ t + t @ Q), 1/2 * t @ Q @ t

def goal_seeking_qr_cost(env, Q=None, x_target=None, QN=None, xN_target=None):
    cost = DiscreteQRCost(env)
    if x_target is not None:
        Q,q,qc = targ2matrices(x_target,Q=Q)
        cost = cost + DiscreteQRCost(env, Q = Q, q=q, qc=qc)

    if xN_target is not None:
        QN, qN, qcN = targ2matrices(xN_target, Q=QN)
        cost += DiscreteQRCost(env, QN=QN, qN=qN, qcN=qcN)
    return cost
