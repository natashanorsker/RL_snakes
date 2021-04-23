"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.

  [Kel17] Matthew Kelly. An introduction to trajectory optimization: how to do your own direct collocation. SIAM Review, 59(4):849–904, 2017. (See kelly2017.pdf). 
"""
import sympy as sym
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from scipy.optimize import Bounds

class ContiniousTimeSymbolicModel: 
    '''
    Continious time symbolic model. See (Her21, Section 8.3) for a top-level description.
    '''
    action_space = None
    observation_space = None
    state_labels = None
    action_labels = None
    def __init__(self, cost=None, simple_bounds=None): 
        self.cost = cost
        t = sym.symbols("t")
        x = symv("x", self.state_size)
        u = symv("u", self.action_size)
        self.f = sym.lambdify( (x, u, t), self.sym_f(x, u, t)) 
        def sp2bounds(space):
            return Bounds(list(space.low), list(space.high))
        sb = dict(t0=Bounds([0], [0]), tF=Bounds([0], [np.inf]),
                  u=sp2bounds(self.action_space),
                  x=sp2bounds(self.observation_space),
                  x0=sp2bounds(self.observation_space),
                  xF=sp2bounds(self.observation_space) )
        if simple_bounds is None:
            simple_bounds = {}
        self.simple_bounds_ = {**sb, **simple_bounds}
        if self.state_labels is None:
            self.state_labels = [f'x{i}' for i in range(self.state_size)]
        if self.action_labels is None:
            self.action_labels = [f'u{i}' for i in range(self.action_size)]


    def reset(self):
        # Returns a start position for the system.
        raise NotImplementedError()

    def simulate(self, x0, u_fun, t0, tF, N_steps=1000, method='rk4'):
        """
        Defaults to RK4 simulation of the trajectory from x0, u0, t0 to tf, see (Her21, Algorithm 18)
        Method can be either 'rk4' or 'euler'

        u_fun has to be a function which returns a list/tuple with same dimension as action_space
        x0 is initial position; it too must be a list of size state_space
        """
        u_fun = ensure_policy(u_fun)
        tt = np.linspace(t0, tF, N_steps+1)   # Time grid t_k = tt[k] between t0 and tF.
        xs = [ np.asarray(x0) ]
        u = [ u_fun(x0, t0 )]
        for k in range(N_steps):
            Delta = tt[k+1] - tt[k]
            tn = tt[k]
            xn = xs[k]
            un = u[k]
            unp = u_fun(xn, tn + Delta)
            if method == 'rk4':
                """ Implement the RK4 method here. This is a rather important question, so please get in touch with me if you are stuck.
                """
                # TODO: 5 lines missing.
                raise NotImplementedError("")
            elif method == 'euler':
                xnp = xn + Delta * np.asarray(self.f(xn, un, tn))
            else:
                raise Exception("Bad integration method", method)
            xs.append(xnp)
            u.append(unp)

        xs = np.stack(xs, axis=0)
        u = np.stack(u, axis=0)
        return xs, u, tt

    def sym_f(self, x, u, t=None):
        raise NotImplementedError()

    @property
    def state_size(self):
        return np.prod(self.observation_space.shape)

    @property
    def action_size(self):
        return np.prod(self.action_space.low.shape)

    def render(self, x, mode="human"):
        raise NotImplementedError()

    """ Below are less important helper-functions etc. """
    def animate_rollout(self, x0, u_fun, t0, tF, N_steps = 1000, fps=10):
        if sys.gettrace() is not None:
            print("Not animating stuff in debugger as it crashes.")
            return
        y, _, tt = self.simulate(x0, u_fun, t0, tF, N_steps=N_steps)
        secs = tF-t0
        frames = int( np.ceil( secs * fps ) )
        I = np.round( np.linspace(0, N_steps-1, frames)).astype(int)
        y = y[I,:]

        for i in range(frames):
            self.render(x=y[i] )
            time.sleep(1/fps)

    def sym_cf(self, t0, tF, x0, xF):
        """ Compute Mayer term in cost function """
        return self.cost.sym_cf(t0=t0, tF=tF, x0=x0, xF=xF)

    def sym_c(self, x, u, t):
        ''' Compute Lagrange term in cost function '''
        return self.cost.sym_c(x=x, u=u, t=t)

    def sym_h(self, x, u, t):
        '''
        Dynamical path constraint of the form: (See (Kel17, Eq.(1.3)))

        h(x, u, t) <= 0.

        These are very rarely used by the methods we consider (so far only in brachiostochrone old with direct methods).
        '''
        return []

    def simple_bounds(self):
        ''' Simple inequality constraints (i.e. z_lb <= z <= z_ub)
        Returned as a dict with keys representing the variables they constrain. For instance:

        >>> sb = env.simple_bounds()
        >>> b = sb['x0']

        b will now be a scipy Bounds object and constraints can be found as:

        >>> b.lb <= x0 <= b.ub

        returned as a dict; see implementations. '''
        return self.simple_bounds_

    def set_simple_bounds(self, bounds):
        self.simple_bounds_ = {**self.simple_bounds_, **bounds}

    def sym_g(self, t0,tF,x0,xF):
        '''
        Boundary constraints

        g(t0,tF,x0,xF) <= 0.

        Note: We will not use this function for the course.
        '''
        return []

    def close(self):
        pass

    def guess(self):
        bnds = self.simple_bounds()
        def mfin(z):
            return [z_ if np.isfinite(z_) else 0 for z_ in z]

        xL = mfin(bnds['x0'].lb)
        xU = mfin(bnds['xF'].ub)

        gs = {'t0': 0,
                 'tF': 4,
                 'x': [xL, xU],
                 'u': [mfin(bnds['u'].lb), mfin(bnds['u'].lb)]}
        return gs

def symv(s, n):
    """
    Returns a vector of symbolic functions. For instance if s='x' and n=3 then it will return
    [x0,x1,x2]
    where x0,..,x2 are symbolic variables.
    """
    return sym.symbols(" ".join(["%s%i," % (s, i) for i in range(n)]))

def ensure_policy(u):
    """
    Ensure u corresponds to a policy function with input arguments u(x, t)
    """
    if callable(u):
        return u
    else:
        return lambda x, t: np.asarray(u).reshape((-1,))

def plot_trajectory(x_res, tt, lt='k-', ax=None, labels=None, legend=None):
    M = x_res.shape[1]
    if labels is None:
        labels = [f"x_{i}" for i in range(M)]

    if ax is None:
        if M == 2:
            a = 234
        if M == 3:
            r = 1
            c = 3
        else:
            r = 2 if M > 1 else 1
            c = (M + 1) // 2

        H = 2*r if r > 1 else 3
        W = 6*c
        # if M == 2:
        #     W = 12
        f, ax = plt.subplots(r,c, figsize=(W,H))
        if M == 1:
            ax = np.asarray([ax])
        print(M,r,c)

    for i in range(M):
        if len(ax) <= i:
            print("issue!")

        a = ax.flat[i]
        a.plot(tt, x_res[:, i], lt, label=legend)

        a.set_xlabel("Time/seconds")
        a.set_ylabel(labels[i])
        # a.set_title(labels[i])
        a.grid(True)
        if legend is not None and i == 0:
            a.legend()
        # if i == M:
    plt.tight_layout()
    return ax

def make_space_above(axes, topmargin=1.0):
    """ increase figure size to make topmargin (in inches) space for
        titles, without changing the axes sizes"""
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)
