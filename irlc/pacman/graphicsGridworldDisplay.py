"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
# graphicsGridworldDisplay.py
# ---------------------------
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
from irlc.pacman import util
from irlc.pacman.graphicsUtils_gym_new import GraphicsUtilGym, sleep, formatColor

BACKGROUND_COLOR = formatColor(0,0,0)
EDGE_COLOR = formatColor(1,1,1)
OBSTACLE_COLOR = formatColor(0.5,0.5,0.5)
TEXT_COLOR = formatColor(1,1,1)
MUTED_TEXT_COLOR = formatColor(0.7,0.7,0.7)
LOCATION_COLOR = formatColor(0,0,1)

WINDOW_SIZE = -1
GRID_SIZE = -1
GRID_HEIGHT = -1
MARGIN = -1



from irlc.pacman.gym_graphicsDisplay import PACMAN_CAPTURE_OUTLINE_WIDTH, PACMAN_COLOR, PACMAN_OUTLINE_WIDTH, PACMAN_SCALE, GHOST_COLORS, TEAM_COLORS
import math



def drawPacman(ga, pacman, index, position=(1,2), gridSize=1):
    # position = self.getPosition(pacman)
    # d = pacman.draw_extra['delta_xy']
    # position = (position[0] + d[0], position[1]+d[1])
    # screen_point = to_screen(position)
    screen_point = position
    # if 'endpoints' in pacman.draw_extra:
    #     endpoints = pacman.draw_extra['endpoints']
    # else:
    endpoints = getEndpoints(0)

    width = PACMAN_OUTLINE_WIDTH
    outlineColor = PACMAN_COLOR
    fillColor = PACMAN_COLOR
    outlineColor = fillColor = LOCATION_COLOR
    # outlineColor = LOCATION_COLOR

    # if self.capture:
    #     outlineColor = TEAM_COLORS[index % 2]
    #     fillColor = GHOST_COLORS[index]
    #     width = PACMAN_CAPTURE_OUTLINE_WIDTH

    return [ga.circle(screen_point, PACMAN_SCALE * gridSize*2,
                   fillColor = fillColor, outlineColor = outlineColor,
                   endpoints = endpoints,
                   width = width)]

def getEndpoints(direction, position=(0,0)):
    x, y = position
    pos = x - int(x) + y - int(y)
    width = 30 + 80 * math.sin(math.pi* pos)

    delta = width / 2
    if (direction == 'West'):
        endpoints = (180+delta, 180-delta)
    elif (direction == 'North'):
        endpoints = (90+delta, 90-delta)
    elif (direction == 'South'):
        endpoints = (270+delta, 270-delta)
    else:
        endpoints = (0+delta, 0-delta)
    return endpoints

class GraphicsGridworldDisplay:
    def __init__(self, gridworld, size=120, speed=1.0, adaptor='tk'):
        self.gridworld = gridworld
        self.size = size
        self.speed = speed
        self.ga = GraphicsUtilGym()
        # if sys.adaptor == 'gym':
        #     self.viewer = viewer

    def setup(self, gridworld, title="Gridworld Display", size=120):
        global GRID_SIZE, MARGIN, SCREEN_WIDTH, SCREEN_HEIGHT, GRID_HEIGHT
        grid = gridworld.grid
        WINDOW_SIZE = size
        GRID_SIZE = size
        GRID_HEIGHT = grid.height
        MARGIN = GRID_SIZE * 0.75
        screen_width = (grid.width - 1) * GRID_SIZE + MARGIN * 2
        screen_height = (grid.height - 0.5) * GRID_SIZE + MARGIN * 2

        self.viewer = self.ga.begin_graphics(screen_width,
                       screen_height,
                       BACKGROUND_COLOR, title=title)

    def start(self):
        self.setup(self.gridworld, size=self.size)

    # def pause(self):
    #     wait_for_keys()

    def displayValues(self, agent, currentState = None, message = 'Agent Values'):
        values = util.Counter()
        policy = {}
        states = self.gridworld.getStates()
        for state in states:
            values[state] = agent.getValue(state)
            policy[state] = agent.getPolicy(state)
        pilim = self.drawValues(self.gridworld, values, policy, currentState, message)
        # sleep(0.05 / self.speed)
        return pilim

    def displayNullValues(self, currentState = None, message = ''):
        values = util.Counter()
        #policy = {}
        states = self.gridworld.getStates()
        for state in states:
            values[state] = 0.0
            #policy[state] = agent.getPolicy(state)
        self.drawNullValues(self.gridworld, currentState,'')
        # drawValues(self.gridworld, values, policy, currentState, message)
        # sleep(0.05 / self.speed)
        # return pilim

    def displayQValues(self, agent, currentState = None, message = 'Agent Q-Values'):
        qValues = util.Counter()
        states = self.gridworld.getStates()
        for state in states:
            for action in self.gridworld.getPossibleActions(state):
                qValues[(state, action)] = agent.getQValue(state, action)
        pilim = self.drawQValues(self.gridworld, qValues, currentState, message)
        # sleep(0.05 / self.speed)
        # return pilim

    def drawNullValues(self, gridworld, currentState=None, message=''):
        grid = gridworld.grid
        self.blank()
        for x in range(grid.width):
            for y in range(grid.height):
                state = (x, y)
                gridType = grid[x][y]
                isExit = (str(gridType) != gridType)
                isCurrent = (currentState == state)
                if gridType == '#':
                    self.drawSquare(x, y, 0, 0, 0, None, None, True, False, isCurrent)
                else:
                    self.drawNullSquare(gridworld.grid, x, y, False, isExit, isCurrent)
        pos = to_screen(((grid.width - 1.0) / 2.0, - 0.8))
        self.ga.text(pos, TEXT_COLOR, message, "Courier", -32, "bold", "c")
        # return get_pil_im()

    # def get_viewer(self):
    #     return get_pil_im()[-1]




    def drawValues(self, gridworld, values, policy, currentState = None, message = 'State Values'):
        grid = gridworld.grid
        self.blank()
        valueList = [values[state] for state in gridworld.getStates()] + [0.0]
        minValue = min(valueList)
        maxValue = max(valueList)
        for x in range(grid.width):
            for y in range(grid.height):
                state = (x, y)
                gridType = grid[x][y]
                isExit = (str(gridType) != gridType)
                isCurrent = (currentState == state)
                if gridType == '#':
                    self.drawSquare(x, y, 0, 0, 0, None, None, True, False, isCurrent)
                else:
                    value = values[state]
                    all_action = []
                    if policy != None and state in policy:
                        all_action = policy[state]
                        actions = gridworld.getPossibleActions(state)
                    # if all_action is not None:
                    # if len(all_action) == 0:
                    #     all_action = []
                    if gridworld.isTerminal(state) or all_action is None:
                        all_action = []
                    if len(all_action) == 1 and all_action[0] == 'exit':
                        all_action = []
                    # print(all_action)

                    all_action = [ ('exit' if  a_ not in actions and 'exit' in actions else a_) for a_ in all_action ]
                    # if action not in actions and 'exit' in actions:
                    #     action = 'exit'

                    valString = '%.2f' % value
                    self.drawSquare(x, y, value, minValue, maxValue, valString, all_action, False, isExit, isCurrent)
        pos = to_screen(((grid.width - 1.0) / 2.0, - 0.8))
        self.ga.text( pos, TEXT_COLOR, message, "Courier", -32, "bold", "c")
        # import reinforcement.graphicsUtils
        # return get_pil_im()


    def drawQValues(self, gridworld, qValues, currentState = None, message = 'State-Action Q-Values'):
        grid = gridworld.grid
        self.blank()
        stateCrossActions = [[(state, action) for action in gridworld.getPossibleActions(state)] for state in gridworld.getStates()]
        import functools
        qStates = functools.reduce(lambda x,y: x+y, stateCrossActions, [])
        qValueList = [qValues[(state, action)] for state, action in qStates] + [0.0]
        minValue = min(qValueList)
        maxValue = max(qValueList)
        for x in range(grid.width):
            for y in range(grid.height):
                state = (x, y)
                gridType = grid[x][y]
                isExit = (str(gridType) != gridType)
                isCurrent = (currentState == state)
                actions = gridworld.getPossibleActions(state)
                if actions == None or len(actions) == 0:
                    actions = [None]
                bestQ = max([qValues[(state, action)] for action in actions])
                bestActions = [action for action in actions if qValues[(state, action)] == bestQ]

                q = util.Counter()
                valStrings = {}
                for action in actions:
                    v = qValues[(state, action)]
                    q[action] += v
                    valStrings[action] = '%.2f' % v
                if gridType == '#':
                    self.drawSquare(x, y, 0, 0, 0, None, None, True, False, isCurrent)
                elif isExit:
                    action = 'exit'
                    action = next(iter(q.keys()))
                    value = q[action] # q[action]
                    valString = '%.2f' % value
                    # print(value)
                    self.drawSquare(x, y, value, minValue, maxValue, valString, [action], False, isExit, isCurrent)
                else:
                    self.drawSquareQ(x, y, q, minValue, maxValue, valStrings, actions, isCurrent)
        pos = to_screen(((grid.width - 1.0) / 2.0, - 0.8))
        self.ga.text( pos, TEXT_COLOR, message, "Courier", -32, "bold", "c")
        # return get_pil_im()
        # return im, viewer

    def blank(self):
        self.ga.clear_screen()

    def drawNullSquare(self, grid,x, y, isObstacle, isTerminal, isCurrent):

        square_color = getColor(0, -1, 1)

        if isObstacle:
            square_color = OBSTACLE_COLOR

        (screen_x, screen_y) = to_screen((x, y))
        self.square( (screen_x, screen_y),
                       0.5* GRID_SIZE,
                       color = square_color,
                       filled = 1,
                       width = 1)

        self.square( (screen_x, screen_y),
                       0.5* GRID_SIZE,
                       color = EDGE_COLOR,
                       filled = 0,
                       width = 3)

        if isTerminal and not isObstacle:
            self.square( (screen_x, screen_y),
                         0.4* GRID_SIZE,
                         color = EDGE_COLOR,
                         filled = 0,
                         width = 2)
            self.ga.text( (screen_x, screen_y),
                   TEXT_COLOR,
                   str(grid[x][y]),
                   "Courier", -24, "bold", "c")


        text_color = TEXT_COLOR

        if not isObstacle and isCurrent:
            # self.ga.circle( (screen_x, screen_y), 0.1*GRID_SIZE, LOCATION_COLOR, fillColor=LOCATION_COLOR )
            self.draw_player((screen_x, screen_y), 0.1 * GRID_SIZE, outlineColor=LOCATION_COLOR, fillColor=LOCATION_COLOR)

    def drawSquare(self, x, y, val, min, max, valStr, all_action, isObstacle, isTerminal, isCurrent):
        square_color = getColor(val, min, max)
        if isObstacle:
            square_color = OBSTACLE_COLOR

        (screen_x, screen_y) = to_screen((x, y))
        self.square( (screen_x, screen_y),
                       0.5* GRID_SIZE,
                       color = square_color,
                       filled = 1,
                       width = 1)
        self.square( (screen_x, screen_y),
                       0.5* GRID_SIZE,
                       color = EDGE_COLOR,
                       filled = 0,
                       width = 3)
        if isTerminal and not isObstacle:
            self.square( (screen_x, screen_y),
                         0.4* GRID_SIZE,
                         color = EDGE_COLOR,
                         filled = 0,
                         width = 2)


        from irlc.gridworld.utils import NORTH, EAST, SOUTH, WEST
        if all_action is None:
            all_action = []
        for action in all_action:
            if action == NORTH:
                self.ga.polygon( [(screen_x, screen_y - 0.45*GRID_SIZE), (screen_x+0.05*GRID_SIZE, screen_y-0.40*GRID_SIZE), (screen_x-0.05*GRID_SIZE, screen_y-0.40*GRID_SIZE)], EDGE_COLOR, filled = 1, smoothed = False)
            if action == SOUTH:
                self.ga.polygon( [(screen_x, screen_y + 0.45*GRID_SIZE), (screen_x+0.05*GRID_SIZE, screen_y+0.40*GRID_SIZE), (screen_x-0.05*GRID_SIZE, screen_y+0.40*GRID_SIZE)], EDGE_COLOR, filled = 1, smoothed = False)
            if action == WEST:
                self.ga.polygon( [(screen_x-0.45*GRID_SIZE, screen_y), (screen_x-0.4*GRID_SIZE, screen_y+0.05*GRID_SIZE), (screen_x-0.4*GRID_SIZE, screen_y-0.05*GRID_SIZE)], EDGE_COLOR, filled = 1, smoothed = False)
            if action == EAST:
                self.ga.polygon( [(screen_x+0.45*GRID_SIZE, screen_y), (screen_x+0.4*GRID_SIZE, screen_y+0.05*GRID_SIZE), (screen_x+0.4*GRID_SIZE, screen_y-0.05*GRID_SIZE)], EDGE_COLOR, filled = 1, smoothed = False)


        text_color = TEXT_COLOR

        if not isObstacle and isCurrent:
            # self.ga.circle( (screen_x, screen_y), 0.1*GRID_SIZE, outlineColor=LOCATION_COLOR, fillColor=LOCATION_COLOR )
            self.draw_player( (screen_x, screen_y), 0.1*GRID_SIZE, outlineColor=LOCATION_COLOR, fillColor=LOCATION_COLOR )

        if not isObstacle:
            self.ga.text( (screen_x, screen_y), text_color, valStr, "Courier", -30, "bold", "c")

    def drawSquareQ(self, x, y, qVals, minVal, maxVal, valStrs, bestActions, isCurrent):
        from irlc.gridworld.utils import NORTH, EAST, SOUTH, WEST
        (screen_x, screen_y) = to_screen((x, y))

        center = (screen_x, screen_y)
        nw = (screen_x-0.5*GRID_SIZE, screen_y-0.5*GRID_SIZE)
        ne = (screen_x+0.5*GRID_SIZE, screen_y-0.5*GRID_SIZE)
        se = (screen_x+0.5*GRID_SIZE, screen_y+0.5*GRID_SIZE)
        sw = (screen_x-0.5*GRID_SIZE, screen_y+0.5*GRID_SIZE)
        n = (screen_x, screen_y-0.5*GRID_SIZE+5)
        s = (screen_x, screen_y+0.5*GRID_SIZE-5)
        w = (screen_x-0.5*GRID_SIZE+5, screen_y)
        e = (screen_x+0.5*GRID_SIZE-5, screen_y)

        actions = qVals.keys()
        for action in actions:
            wedge_color = getColor(qVals[action], minVal, maxVal)
            if action == NORTH:
                self.ga.polygon( (center, nw, ne), wedge_color, filled = 1, smoothed = False)
                #text(n, text_color, valStr, "Courier", 8, "bold", "n")
            if action == SOUTH:
                self.ga.polygon( (center, sw, se), wedge_color, filled = 1, smoothed = False)
                #text(s, text_color, valStr, "Courier", 8, "bold", "s")
            if action == EAST:
                self.ga.polygon( (center, ne, se), wedge_color, filled = 1, smoothed = False)
                #text(e, text_color, valStr, "Courier", 8, "bold", "e")
            if action == WEST:
                self.ga.polygon( (center, nw, sw), wedge_color, filled = 1, smoothed = False)
                #text(w, text_color, valStr, "Courier", 8, "bold", "w")

        self.square( (screen_x, screen_y),
                       0.5* GRID_SIZE,
                       color = EDGE_COLOR,
                       filled = 0,
                       width = 3)
        self.ga.line(ne, sw, color = EDGE_COLOR)
        self.ga.line(nw, se, color = EDGE_COLOR)

        if isCurrent:
            # self.ga.circle( (screen_x, screen_y), 0.1*GRID_SIZE, LOCATION_COLOR, fillColor=LOCATION_COLOR )
            self.draw_player((screen_x, screen_y), 0.1 * GRID_SIZE, outlineColor=LOCATION_COLOR, fillColor=LOCATION_COLOR)


        for action in actions:
            text_color = TEXT_COLOR
            if qVals[action] < max(qVals.values()): text_color = MUTED_TEXT_COLOR
            valStr = ""
            if action in valStrs:
                valStr = valStrs[action]
            h = -20
            if action == NORTH:
                #polygon( (center, nw, ne), wedge_color, filled = 1, smooth = 0)
                self.ga.text(n, text_color, valStr, "Courier", h, "bold", "n")
            if action == SOUTH:
                #polygon( (center, sw, se), wedge_color, filled = 1, smooth = 0)
                self.ga.text(s, text_color, valStr, "Courier", h, "bold", "s")
            if action == EAST:
                #polygon( (center, ne, se), wedge_color, filled = 1, smooth = 0)
                self.ga.text(e, text_color, valStr, "Courier", h, "bold", "e")
            if action == WEST:
                #polygon( (center, nw, sw), wedge_color, filled = 1, smooth = 0)
                self.ga.text(w, text_color, valStr, "Courier", h, "bold", "w")

    def square(self, pos, size, color, filled, width):
        x, y = pos
        dx, dy = size, size
        return self.ga.polygon([(x - dx, y - dy), (x - dx, y + dy), (x + dx, y + dy), (x + dx, y - dy)], outlineColor=color,
                       fillColor=color, filled=filled, width=width, smoothed=False,closed=True)


    def draw_player(self, position, grid_size, outlineColor=LOCATION_COLOR, fillColor=LOCATION_COLOR):
        return drawPacman(self.ga, pacman=None, index=0, position=position, gridSize=grid_size*1.2)
        # draw_pacman(self.ga)


def getColor(val, minVal, max):
    r, g = 0.0, 0.0
    if val < 0 and minVal < 0:
        r = val * 0.65 / minVal
    if val > 0 and max > 0:
        g = val * 0.65 / max
    return formatColor(r,g,0.0)

def to_screen(point):
    ( gamex, gamey ) = point
    x = gamex*GRID_SIZE + MARGIN
    y = (GRID_HEIGHT - gamey - 1)*GRID_SIZE + MARGIN
    return ( x, y )

def to_grid(point):
    (x, y) = point
    x = int ((y - MARGIN + GRID_SIZE * 0.5) / GRID_SIZE)
    y = int ((x - MARGIN + GRID_SIZE * 0.5) / GRID_SIZE)
    print(point, "-->", (x, y))
    return (x, y)
