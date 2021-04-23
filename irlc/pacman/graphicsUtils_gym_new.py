"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
# graphicsUtils.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
# 547 lines
# 556 lines

import sys
import time
import numpy as np
import pyglet
from gym.envs.classic_control import rendering
from gym.envs.classic_control.rendering import Viewer

ghost_shape = [
    (0, - 0.5),
    (0.25, - 0.75),
    (0.5, - 0.5),
    (0.75, - 0.75),
    (0.75, 0.5),
    (0.5, 0.75),
    (- 0.5, 0.75),
    (- 0.75, 0.5),
    (- 0.75, - 0.75),
    (- 0.5, - 0.5),
    (- 0.25, - 0.75)
]

def _adjust_coords(coord_list, x, y):
    for i in range(0, len(coord_list), 2):
        coord_list[i] = coord_list[i] + x
        coord_list[i + 1] = coord_list[i + 1] + y
    return coord_list

def formatColor(r, g, b):
    return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))

def colorToVector(color):
    return list(map(lambda x: int(x, 16) / 256.0, [color[1:3], color[3:5], color[5:7]]))

def h2rgb(color):
    if color is None:
        return None
    if color.startswith("#"):
        color = color[1:]
    return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))

def sleep(secs):
    time.sleep(secs)
    # global _root_window
    # if _root_window == None:
    #     time.sleep(secs)
    # else:
    #     _root_window.update_idletasks()
    #     _root_window.after(int(1000 * secs), _root_window.quit)
    #     _root_window.mainloop()

# def render():
#     pass
# return viewer.render()

class GraphicsUtilGym:
    viewer = None
    _Windows = sys.platform == 'win32'  # True if on Win95/98/NT

    _root_window = None      # The root window for graphics output
    _canvas = None      # The canvas which holds graphics
    _canvas_xs = None      # Size of canvas object
    _canvas_ys = None
    _canvas_x = None      # Current position on canvas
    _canvas_y = None
    _canvas_col = None      # Current colour (set to black below)
    _canvas_tsize = 12
    _canvas_tserifs = 0




    def begin_graphics(self, width=640, height=480, color=formatColor(0, 0, 0), title=None):


        self.viewer = Viewer(width=int(width), height=int(height))
        # viewer.render()
        from gym.envs.classic_control.cartpole import CartPoleEnv
        # Check for duplicate call
        # if _root_window is not None:
            # Lose the window.
            # _root_window.destroy()

        # Save the canvas size parameters
        self._canvas_xs, self._canvas_ys = width - 1, height - 1
        self._canvas_x, self._canvas_y = 0, self._canvas_ys
        self._bg_color = color
        return self.viewer

    def draw_background(self):
        corners = [(0,0), (0, self._canvas_ys), (self._canvas_xs, self._canvas_ys), (self._canvas_xs, 0)]
        self.polygon(corners, self._bg_color, fillColor=self._bg_color, filled=True, smoothed=False)

    def clear_screen(self, draw_background=True):
        # global _canvas_x, _canvas_y
        # _canvas.delete('all')
        self.viewer.geoms = []
        if draw_background:
            self.draw_background()
        # _canvas_x, _canvas_y = 0, _canvas_ys

    def fixxy(self, xy):
        return (xy[0], self.fixy(xy[1]))

    def plot(self, x, y, color=None, width=1.0):
        coords = [(x_,y_) for (x_, y_) in zip(x,y)]
        if color is None:
            color = "#000000"
        # from gym.envs.classic_control.rendering import make_polyline
        return self.polygon(coords, outlineColor=color, filled=False, width=width)

    def polygon(self, coords, outlineColor, fillColor=None, filled=True, smoothed=1, behind=0, width=1.0, closed=False):
        # print("polygon")
        c = []
        for coord in coords:
            c.append(coord[0])
            c.append(coord[1])

        if fillColor == None: fillColor = outlineColor
        if not filled: fillColor = ""
        from gym.envs.classic_control import rendering
        c = [self.fixxy(tuple(c[i:i+2])) for i in range(0, len(c), 2)]
        poly = None

        if not filled:
            poly = rendering.PolyLine(c, close=closed)
            poly.set_linewidth(width)
            poly.set_color(*h2rgb(outlineColor))
        else:
            poly = rendering.FilledPolygon(c)
            poly.set_color(*h2rgb(fillColor))
            poly.add_attr(rendering.LineWidth(10))

        if len(outlineColor) > 0 and filled: # Not sure why this cannot be merged with the filled case...
            outl = rendering.PolyLine(c, close=True)
            outl.set_linewidth(width)
            outl.set_color(*h2rgb(outlineColor))
            # poly = outl
        if poly is not None:
            self.viewer.add_geom(poly)
        else:
            raise Exception("Bad polyline")

        return poly

    def square(self, pos, r, color, filled=1, behind=0):
        x, y = pos
        coords = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
        return self.polygon(coords, color, color, filled, 0, behind=behind)

    def circle(self, pos, r, outlineColor, fillColor, endpoints=None, style='pieslice', width=2):
        x, y = pos
        x0, x1 = x - r - 1, x + r
        y0, y1 = y - r - 1, y + r
        if endpoints == None:
            e = [0, 359]
        else:
            e = list(endpoints)
        while e[0] > e[1]: e[1] = e[1] + 360
        # viewer.draw_circle()
        # r *= 3
        if endpoints is not None and len(endpoints) > 0:
            tt = np.linspace(e[0]/360 * 2*np.pi, e[-1]/360 * 2*np.pi, int(r*20) )
            px = np.cos(tt) * r
            py = -np.sin(tt) * r
            pp = list(zip(px.tolist(), py.tolist()))
            if style == 'pieslice':
                pp = [(0,0),] + pp + [(0,0),]
            pp = [( (x+a, y+b)) for (a,b) in pp  ]
            if style == 'arc':
                pp = [self.fixxy(p_) for p_ in pp]
                outl = rendering.PolyLine(pp, close=False)
                outl.set_linewidth(width)
                outl.set_color(*h2rgb(outlineColor))
                self.viewer.add_geom(outl)
            elif style == 'pieslice':
                circ = self.polygon(pp, fillColor=fillColor, outlineColor=outlineColor, width=width)
            else:
                raise Exception("bad style", style)
        else:
            circ = rendering.make_circle(r)
            circ.set_color(*h2rgb(fillColor))
            tf = rendering.Transform(translation = self.fixxy(pos))
            circ.add_attr(tf)
            self.viewer.add_geom(circ)
        return None
        # return _canvas.create_arc(x0, y0, x1, y1, outline=outlineColor, fill=fillColor,
        #                           extent=e[1] - e[0], start=e[0], style=style, width=width)

    # def image(pos, file="../../blueghost.gif"):
    #     x, y = pos
    #     # img = PhotoImage(file=file)
    #     return _canvas.create_image(x, y, image = Tkinter.PhotoImage(file=file), anchor = Tkinter.NW)

    #
    # def refresh():
    #     pass
        # _canvas.update_idletasks()

    def moveCircle(self, id, pos, r, endpoints=None):
        global _canvas_x, _canvas_y
        x, y = pos
        x0, x1 = x - r - 1, x + r
        y0, y1 = y - r - 1, y + r
        if endpoints == None:
            e = [0, 359]
        else:
            e = list(endpoints)
        while e[0] > e[1]: e[1] = e[1] + 360
        self.edit(id, ('start', e[0]), ('extent', e[1] - e[0]))
        # move_to(id, x0, y0)

    def edit(id, *args):
        pass
        # _canvas.itemconfigure(id, **dict(args))

    # def get_pil_im():
    #     return None, None, viewer

    def fixy(self, y):
        return self.viewer.height-y

    def text(self, pos, color, contents, font='Helvetica', size=12, style='normal', anchor="w"):
        # global _canvas_x, _canvas_y

        x, y = pos
        # font0 = font
        font = (font, str(size), style)
        from PIL import ImageFont
        # font = ImageFont.truetype("Helvetica", 15)
        from io import BytesIO
        # font = ImageFont.truetype(font0+".tff", size)
        # draw.text((x, y),"Sample Text",(r,g,b))

        # file = open("fonts/AbyssinicaSIL-R.ttf", "rb")
        if size < 0:
            sz = -size
        else:
            sz = size
        # pfont = ImageFont.truetype('gulim', size=sz)
        # w, h = font.getsize("contents")
        # W, H = pilim.size
        # font.getsize(contents)
        # file = open("Fonts/courB12.pil", "rb")
        # bytes_font = BytesIO(file.read())
        # font = ImageFont.truetype(bytes_font, 15)
        # w, h = pdraw.textsize(contents, font=pfont)
        # print(style, font)
        # if anchor == 'c':
        #     xx = x - w//2
        #     yy = y - h//2
        # elif anchor == "n":
        #     xx = x - w//2
        #     yy = y
        # elif anchor == "s":
        #     xx = x - w//2
        #     yy = y - h
        # elif anchor == "w":
        #     xx = x
        #     yy = y - h//2
        # elif anchor == "e":
        #     xx = x - w
        #     yy = y - h//2
        # else:
        #     xx = x
        #     yy = y

        # pyglet fix
        ax = "center"
        ax = "left" if anchor == "w" else ax
        ax = "right" if anchor == "e" else ax

        ay = "center"
        ay = "baseline" if anchor == "s" else ay
        ay = "top" if anchor == "n" else ay

        psz = int(-size * 0.75) if size < 0 else size

        cl = tuple(int(c*255) for c in h2rgb(color) )+(255,)
        label = pyglet.text.Label(contents, x=int(x), y = self.fixy(int(y)),  font_name='Arial', font_size=psz, bold=style=="bold",
                                  color=cl,
                                  anchor_x=ax, anchor_y=ay)

        self.viewer.add_geom(TextGeom(label))
        return None
        # pdraw.text( (xx, yy), contents, fill=color, font=pfont)
        # return _canvas.create_text(x, y, fill=color, text=contents, font=font, anchor=anchor)


    def changeText(self, id, newText, font=None, size=12, style='normal'):
        # print("Changing text")
        # _canvas.itemconfigure(id, text=newText)
        if font != None:
            pass
            # _canvas.itemconfigure(id, font=(font, '-%d' % size, style))

    # def changeColor(id, newColor):
    #     _canvas.itemconfigure(id, fill=newColor)



    def line(self, here, there, color=formatColor(0, 0, 0), width=2):
        x0, y0 = here[0], here[1]
        x1, y1 = there[0], there[1]

        # pil
        # pdraw.line( ( (x0, y0), (x1, y1) ), fill=color, width=width)
        # pyglet
        from gym.envs.classic_control import rendering

        # c = [tuple(c[i:i + 2]) for i in range(0, len(c), 2)]
        # if filled == 0:
        #     poly = rendering.PolyLine(c, close=True)
        # else:
        poly = MyLine(self.fixxy(here), self.fixxy(there), width=width)
        # poly.add_attr()
        # poly.set_linewidth(width)
        # rendering._add_attrs(poly, {'linewidth':0})
        # poly.set_linewidth(0)
        poly.set_color(*h2rgb(color))
        poly.add_attr(rendering.LineWidth(width))

        # if len(outlineColor) > 0:
        #     outl = rendering.PolyLine(c, close=True)
        #
        #     outl.set_color(*h2rgb(outlineColor))
        #     viewer.add_geom(outl)

        self.viewer.add_geom(poly)
        return None
        # return _canvas.create_line(x0, y0, x1, y1, fill=color, width=width)

    ##############################################################################
    ### Keypress handling ########################################################
    ##############################################################################

    # We bind to key-down and key-up events.

    # _keysdown = {}
    # _keyswaiting = {}
    # # This holds an unprocessed key release.  We delay key releases by up to
    # # one call to keys_pressed() to get round a problem with auto repeat.
    # _got_release = None
    #
    # def _keypress(event):
    #     global _got_release
    #     #remap_arrows(event)
    #     _keysdown[event.keysym] = 1
    #     _keyswaiting[event.keysym] = 1
    # #    print event.char, event.keycode
    #     _got_release = None
    #
    # def _keyrelease(event):
    #     global _got_release
    #     #remap_arrows(event)
    #     try:
    #         del _keysdown[event.keysym]
    #     except:
    #         pass
    #     _got_release = 1
    #
    # def remap_arrows(event):
    #     # TURN ARROW PRESSES INTO LETTERS (SHOULD BE IN KEYBOARD AGENT)
    #     if event.char in ['a', 's', 'd', 'w']:
    #         return
    #     if event.keycode in [37, 101]: # LEFT ARROW (win / x)
    #         event.char = 'a'
    #     if event.keycode in [38, 99]: # UP ARROW
    #         event.char = 'w'
    #     if event.keycode in [39, 102]: # RIGHT ARROW
    #         event.char = 'd'
    #     if event.keycode in [40, 104]: # DOWN ARROW
    #         event.char = 's'

    # def _clear_keys(event=None):
    #     global _keysdown, _got_release, _keyswaiting
    #     _keysdown = {}
    #     _keyswaiting = {}
    #     _got_release = None

    # def keys_pressed(d_o_e=None,
    #                  d_w=tkinter._tkinter.DONT_WAIT):
    #     if d_o_e is None:
    #         d_o_e = _root_window.dooneevent
    #
    #     d_o_e(d_w)
    #     if _got_release:
    #         d_o_e(d_w)
    #     return _keysdown.keys()

    # def keys_waiting():
    #     global _keyswaiting
    #     keys = _keyswaiting.keys()
    #     _keyswaiting = {}
    #     return keys

    # Block for a list of keys...
    # def wait_for_keys():
    #     keys = []
    #     while keys == []:
    #         keys = keys_pressed()
    #         sleep(0.05)
    #     return keys
    # def remove_from_screen(x,
    #                        d_o_e=None,
    #                        d_w=tkinter._tkinter.DONT_WAIT):
    #     if d_o_e is None:
    #         d_o_e = _root_window.dooneevent
    #     _canvas.delete(x)
    #     d_o_e(d_w)


    # def writePostscript(filename):
    #     "Writes the current canvas to a postscript file."
    #     with open(filename, 'w') as psfile:
    #         psfile.write(_canvas.postscript(pageanchor='sw',
    #                          y='0.c',
    #                          x='0.c'))
    #     # psfile.close()

from gym.envs.classic_control.rendering import glBegin, GL_LINES, glVertex2f, glEnd
class MyLine(rendering.Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0), width=1):
        rendering.Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = rendering.LineWidth(width)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()

class TextGeom(rendering.Geom):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def render(self):
        self.label.draw()
