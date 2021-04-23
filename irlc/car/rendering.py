"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
# import matplotlib.pyplot as plt

def make_matplotlib_viewer(model, matplotlib_renderfun):
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    matplotlib_renderfun(plt)
    if not hasattr(model, 'viewer'):
        model.viewer = None
    # plotClosedLoopLMPC(LMPController=None, map=self.map)
    fig = plt.gcf()

    fig.canvas.draw()

    # from matplotlib.backends.backend_
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    width, height = fig.get_size_inches() * fig.get_dpi()

    # buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # buf.shape = (h, w, 3)
    # plotdata = buf[:,:,1:]
    # plt.title("Actual plot")
    # plt.show()
    # plt.figure()
    # plt.imshow(plotdata)
    # plt.title("Image plot")
    # plt.show()
    # arr = buf[:, :, 1:]
    arr = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
    import pyglet
    from pyglet import gl
    image = pyglet.image.ImageData(arr.shape[1], arr.shape[0],
                                   'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)
    gl.glTexParameteri(gl.GL_TEXTURE_2D,
                       gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

    if model.viewer is None:
        from gym.envs.classic_control import rendering
        model.viewer = rendering.Viewer(w, h)
        # self.viewer = rendering.SimpleImageViewer()
        # format_size =3
        # bytes_per_channel = 1
        # dimensions = (160,160)
        # # populate our array with some random data
        # data = np.random.random_integers(
        #     low=0,
        #     high=1,
        #     size=(dimensions[0], dimensions[1], format_size)
        # )
        # convert any 1's to 255
        # data *= 200
        # data[50:,50:,:] = 100
        # arr = data
        # height, width, _channels = arr.shape

        # a = 234
        # texture = image.get_texture()
        # self.width = 3
        # self.height = 3
        #
        # texture.width = self.width
        # texture.height = self.height

        # pixels = [
        #     255, 0, 0, 0, 255, 0, 0, 0, 255,  # RGB values range from
        #     255, 0, 0, 255, 0, 0, 255, 0, 0,  # 0 to 255 for each color
        #     255, 0, 0, 255, 0, 0, 255, 0, 0,  # component.
        # ]
        # rawData = (gl.GLubyte * len(pixels))(*pixels)
        # imageData = pyglet.image.ImageData(3, 3, 'RGB', rawData)

        # self.window.clear()
        # self.window.switch_to()
        # self.window.dispatch_events()
        # texture.blit(0, 0)  # draw
        # texture.blit(-self.width / 2, -self.height / 2, width=self.width, height=self.height)

        # self.viewer.imshow( data )
        # self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
        # car = rendering.make_capsule(1, .2)
        # car.set_color(.8, .3, .3)
        # self.pole_transform = rendering.Transform()
        # car.add_attr(self.pole_transform)

        # self.viewer.add_geom(car)

        # axle = rendering.make_circle(.05)
        # axle.set_color(0, 0, 0)
        # self.viewer.add_geom(axle)
        # fname = path.join(path.dirname(__file__), "assets/clockwise.png")

        # a = 234
        # if False:
        #     from irlc.ex16.rendering import MatplotlibImage
        #
        #     self.img = MatplotlibImage(im=None, width=4., height=4.)
        #
        #     self.imgtrans = rendering.Transform()
        #     self.img.add_attr(self.imgtrans)


    model.viewer.add_onetime(MI(image, H=h, W=w))

class MI():
    def __init__(self, image, H, W):
        self.image = image
        self.width = W
        self.height = H

    def render(self):
        # self.image.blit(-self.width / 2, -self.height / 2, width=self.width, height=self.height)
        self.image.blit(0, 0, width=self.width, height=self.height)

from PIL import Image


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.fromstring("RGBA", (w, h), buf.tostring())

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=numpy.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


from gym.envs.classic_control.rendering import Geom


class MatplotlibImage(Geom):
    def __init__(self, im, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height

        # img = pyglet.image.load(fname)
        # self.img = img
        self.img = None  # make pyglet image here
        self.img = im2pyglet(im)
        print(self.img)

        self.flip = False

    def render1(self):
        self.img.blit(-self.width / 2, -self.height / 2, width=self.width, height=self.height)


from pyglet.gl import *
import pyglet


def im2pyglet(data):
    import numpy


    # the size of our texture
    dimensions = (16, 16)

    # we need RGBA textures
    # which has 4 channels
    format_size = 4
    bytes_per_channel = 1

    # populate our array with some random data
    data = numpy.random.random_integers(
        low = 0,
        high = 1,
        size = (dimensions[ 0 ] * dimensions[ 1 ], format_size)
        )

    # convert any 1's to 255
    data *= 255

    # set the GB channels (from RGBA) to 0
    data[ :, 1:-1 ] = 0


    # ensure alpha is always 255
    data[ :, 3] = 255

    data[16 * 16 // 2:, 0] = 100

    # we need to flatten the array

    # convert to GLubytes
    # tex_data = (GLubyte * data.size)(*data.astype('uint8'))


    pixels = data[:,:3].reshape((-1,)).tolist()

    # pixels[:40] = 100
    for k in range(100):
        pixels[k] = k

    # data.shape = -1

    # pixels = data[:,:3].tolist()

    print(pixels)

    rawData = (GLubyte * len(pixels))(*pixels)


    imageData = pyglet.image.ImageData(dimensions[0],
        dimensions[1], 'RGB', rawData)
    return imageData


    # print(tex_data)
    return pyglet.image.ImageData(
        dimensions[0],
        dimensions[1],
        "RGBA",
        tex_data,
        pitch=dimensions[1] * format_size * bytes_per_channel
    )

    # return data
