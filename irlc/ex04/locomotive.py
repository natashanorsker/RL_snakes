"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
from irlc.ex04.model_harmonic import HarmonicOscilatorModel

class LocomotiveModel(HarmonicOscilatorModel):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }
    def __init__(self, m=1., slope=0.0, target=0):
        """
        Slope is the uphill slope of the train (in degrees). E.g. slope=15 makes it harder for the engine.

        :param m:
        :param slope:
        """
        self.target = target
        drag = -np.sin(slope/360*2*np.pi) * m * 9.82
        self.slope = slope
        super().__init__(m=m, k=0., drag=drag)
        from gym.spaces import Box
        self.action_space = Box(low=np.asarray([-100.]), high=np.asarray([100.]), dtype=np.float)

    def reset(self):
        return [-1, 0]

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
            theta = np.radians(-self.slope)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            xl = R[0,0] * xl + R[0, 1] * yl
            yl = R[1, 0] * xl + R[1, 1] * yl

            x_ = (xl - xlims[0]) / (xlims[1]- xlims[0])
            y_ = (yl - ylims[0]) / (ylims[1] - ylims[0])
            x_ = x_ * self.gd._canvas_xs
            y_ = y_ * self.gd._canvas_ys
            return x_, y_

        def l2g_scale(x):
            return l2g(x,0)[0]-l2g(0,0)[0]

        pos = l2g(x[0],0)
        self.gd.square( pos=l2g(self.target,0), r =l2g_scale(0.05), color=formatColor(0,0,0) )
        self.gd.circle(pos=pos, r=l2g_scale(0.1), fillColor=formatColor(.7, .7, .7), outlineColor="#000000")
        dw = 0.03
        pb = [ (-2,dw), (2, dw), (2, -dw), (-2, -dw)]
        pb = [l2g(*p) for p in pb]
        self.gd.polygon( pb, outlineColor=formatColor(0, 0, 0), filled=True, fillColor=formatColor(.7, .7, .7) )
        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def close(self):
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()

class DiscreteLocomotiveModel(DiscretizedModel):
    def __init__(self, *args, dt=0.1, **kwargs):
        model = LocomotiveModel(*args, **kwargs)
        super().__init__(model=model, dt=dt)
        self.cost = model.cost.discretize(self, dt=dt)

class LocomotiveEnvironment(ContiniousTimeEnvironment):
    def __init__(self, *args, dt=0.1, Tmax=5, **kwargs):
        model = DiscreteLocomotiveModel(*args, dt=dt, **kwargs)
        self.dt = model.dt
        super().__init__(discrete_model=model, Tmax=Tmax)
