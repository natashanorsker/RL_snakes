"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
import numpy as np
from irlc.ex04.model_linear_quadratic import LinearQuadraticModel
"""
   Simulate a Harmonic oscillator governed by equations:

   d^2 x1 / dt^2 = -k/m x1 + u(x1, t)

   where x1 is the position and u is our externally applied force (the control)
   k is the spring constant and m is the mass. See:

   https://en.wikipedia.org/wiki/Simple_harmonic_motion#Dynamics

   for more details.
   In the code, we will re-write the equations as:

   dx/dt = f(x, u),   u = u_fun(x, t)

   where x = [x1,x2] is now a vector and f is a function of x and the current control.
   here, x1 is the position (same as x in the first equation) and x2 is the velocity.

   The function should return ts, xs, C

   where ts is the N time points t_0, ..., t_{N-1}, xs is a corresponding list [ ..., [x_1(t_k),x_2(t_k)], ...] and C is the cost.
   """

class HarmonicOscilatorModel(LinearQuadraticModel):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }
    """
    See: https://books.google.dk/books?id=tXZDAAAAQBAJ&pg=PA147&lpg=PA147&dq=boeing+747+flight+0.322+model+longitudinal+flight&source=bl&ots=L2RpjCAWiZ&sig=ACfU3U2m0JsiHmUorwyq5REcOj2nlxZkuA&hl=en&sa=X&ved=2ahUKEwir7L3i6o3qAhWpl4sKHQV6CdcQ6AEwAHoECAoQAQ#v=onepage&q=boeing%20747%20flight%200.322%20model%20longitudinal%20flight&f=false
    """
    def __init__(self, k=1., m=1., drag=0.0, Q=None, R=None):
        self.k = k
        self.m = m
        A = [[0, 1],
             [-k/m, 0]]

        B = [[0], [1/m]]
        C = [[0], [drag/m]]

        A, B, C = np.asarray(A), np.asarray(B), np.asarray(C)
        if Q is None:
            Q = np.eye(2)
        if R is None:
            R = np.eye(1)
        self.viewer = None
        super().__init__(A=A, B=B, Q=Q, R=R, d=C)

    def reset(self):
        return [1, 0]

    def render(self, x, mode="human"):
        """ This is a bunch of messy code responsible for rendering the environment.
        you do not have to understand it. """
        from irlc.pacman.graphicsUtils_gym_new import GraphicsUtilGym, formatColor
        if self.viewer is None:
            self.gd = GraphicsUtilGym()
            self.viewer = self.gd.begin_graphics(color="#ffffff")
        self.gd.clear_screen(draw_background=True)

        # we want something to map local coordinates to global.
        xlims = [-2, 2]
        ylims = [-2, 2]
        def l2g(xl,yl):
            x_ = (xl - xlims[0]) / (xlims[1]- xlims[0])
            y_ = (yl - ylims[0]) / (ylims[1] - ylims[0])
            x_ = x_ * self.gd._canvas_xs
            y_ = y_ * self.gd._canvas_ys
            return x_, y_

        def l2g_scale(x):
            return l2g(x,0)[0]-l2g(0,0)[0]

        pos = l2g(x[0],0)
        xx = np.linspace(0,1)
        coil = np.sin(xx*2*np.pi*5)
        self.gd.square( pos=l2g(0,0), r =l2g_scale(0.1), color=formatColor(0,0,0) )
        self.gd.circle(pos=pos, r=l2g_scale(0.1), fillColor=formatColor(.7, .7, .7), outlineColor="#000000")
        xx,coil = l2g(xx*(x[0]-0.2)+0.1, coil*0.1)
        self.gd.plot(xx, coil, width=1.0)
        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def close(self):
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()

class DiscreteHarmonicOscilatorModel(DiscretizedModel): 
    def __init__(self, dt=0.1, discretization_method=None, **kwargs):
        model = HarmonicOscilatorModel(**kwargs)
        super().__init__(model=model, dt=dt, discretization_method=discretization_method)
        self.cost = model.cost.discretize(self, dt=dt) 

class HarmonicOscilatorEnvironment(ContiniousTimeEnvironment): 
    def __init__(self, Tmax=80, supersample_trajectory=False, **kwargs):
        model = DiscreteHarmonicOscilatorModel(**kwargs)
        self.dt = model.dt
        super().__init__(discrete_model=model, Tmax=Tmax, supersample_trajectory=supersample_trajectory) 
