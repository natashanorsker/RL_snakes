"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import sys
from collections import defaultdict
from gym.spaces.discrete import Discrete
from irlc.ex09.mdp import MDP
from irlc.ex09.mdp import MDP2GymEnv
from irlc.gridworld.utils import Gridworld

grid_cliff_grid = [[' ',' ',' ',' ',' '],
                   ['S',' ',' ',' ',' '],
                   [-100,-100, -100, -100, 10]]

grid_cliff_grid2 = [[' ',' ',' ',' ',' '],
        [8,'S',' ',' ',10],
        [-100,-100, -100, -100, -100]]

grid_discount_grid = [[' ',' ',' ',' ',' '],
        [' ','#',' ',' ',' '],
        [' ','#', 1,'#', 10],
        ['S',' ',' ',' ',' '],
        [-10,-10, -10, -10, -10]]

grid_bridge_grid = [[ '#',-100, -100, -100, -100, -100, '#'],
        [   1, 'S',  ' ',  ' ',  ' ',  ' ',  10],
        [ '#',-100, -100, -100, -100, -100, '#']]

grid_book_grid = [[' ',' ',' ',+1],
        [' ','#',' ',-1],
        ['S',' ',' ',' ']]

grid_maze_grid = [[' ',' ',' ',+1],
        ['#','#',' ','#'],
        [' ','#',' ',' '],
        [' ','#','#',' '],
        ['S',' ',' ',' ']]

sutton_corner_maze = [[1,   ' ', ' ', ' '], 
                      [' ', ' ', ' ', ' '],
                      [' ', 'S', ' ', ' '],
                      [' ', ' ', ' ', 1]] 

# A big old open maze.
grid_open_grid = [[' ']*8 for _ in range(5)]
grid_open_grid[0][0] = 'S'
grid_open_grid[-1][-1] = 1

class BerkleyGridMDP(MDP):
    def __init__(self, grid, living_reward=0):
        bmdp = Gridworld(grid)
        super().__init__(initial_state=bmdp.getStartState())
        self.step_reward = living_reward
        self.bmdp = bmdp
        self.is_terminal = self.bmdp.isTerminal
        self.A = self.bmdp.getPossibleActions
        # self.epsilon = epsilon

    def Psr(self, state, action):
        # P = {}
        P = defaultdict(float)
        # eps = self.epsilon
        # nA = len(self.A(state))
        # for a in self.A(state):
        #     pr = (1-self.epsilon) + self.epsilon/nA if a == action else self.epsilon/nA

        for sp, pr in self.bmdp.getTransitionStatesAndProbs(state, action):
            r = self.bmdp.getReward(state, action, sp) + self.step_reward
            P[(sp, r)] += pr

        return P

class FrozenGridMDP(BerkleyGridMDP):
    def __init__(self, grid, is_slippery=True, living_reward=0):
        self.is_slippery= is_slippery
        super().__init__(grid, living_reward=living_reward)

    def Psr(self, state, action):
        if not self.is_slippery or self.is_terminal(state):
            return super().Psr(state, action)
        else:
            a = action
            acts = [(a - 1) % 4, a, (a + 1) % 4]
            acts = [a_ for a_ in acts if a_ in self.A(state)]
            P = defaultdict(float)
            for a_ in acts:
                tp = self.bmdp.getTransitionStatesAndProbs(state, a_)
                for sp, pr in tp:
                    r = self.bmdp.getReward(state, a_, sp) + self.step_reward
                    P[(sp, r)] += pr * 1/len(acts)
            return P

class BerkleyGridEnvironment(MDP2GymEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def get_keys_to_action(self):
        # from reinforcement.gridworld import NORTH, SOUTH, EAST, WEST
        from irlc.gridworld.utils import NORTH, SOUTH, EAST, WEST
        from pyglet.window import key
        return {(key.LEFT,): WEST, (key.RIGHT,): EAST, (key.UP,): NORTH, (key.DOWN,): SOUTH}

    def _get_mdp(self, grid, uniform_initial_state=False):
        return BerkleyGridMDP(grid, living_reward=self.living_reward)

    def __init__(self, grid=None, adaptor='gym', uniform_initial_state=True, living_reward=0,**kwargs):
        self.living_reward = living_reward
        mdp = self._get_mdp(grid)
        super().__init__(mdp)
        self.display = None
        self.adaptor = adaptor
        self.viewer = None
        self.action_space = Discrete(4)
        self.render_episodes = 0
        self.render_steps = 0


        def _step(*args, **kwargs):
            o = type(self).step(self, *args, **kwargs)
            done = o[2]
            self.render_steps += 1
            self.render_episodes += done
            return o
        self.step = _step

    def render(self, mode='human', s=None, v=None, Q=None, pi=None, policy=None, v2Q=None, gamma=0, method_label="", label=None):
        sys.adaptor = self.adaptor
        if label is None:
            label = f"{method_label} AFTER {self.render_steps} STEPS"
        from irlc.pacman import graphicsGridworldDisplay  # must be in this order bc. of hacky trick.
        speed = 1
        gridSize = 150
        if self.display is None:
            self.display = graphicsGridworldDisplay.GraphicsGridworldDisplay(self.mdp.bmdp, gridSize, speed)
            self.display.start()

        if s is None:
            s = self.state
        state = s
        class MAQ:
            def __init__(self, Q, states, env):
                self.states = states
                self.Q = Q
                self.env = env
                self.lup = {s: i for i, s in enumerate(self.states)}

            def getQValue(self, state, action):
                return self.Q[state, action]

        class MAG2:
            def __init__(self, env, v, states, pi=None, policy=None, v2Q=None):
                """ Policy is a more general policy which outputs multiple actions for each state. Generally in the policy[s][a] = prob. format. """
                self.states = states
                self.v = v
                self.pi = pi
                self.lup = {s: i for i, s in enumerate(self.states)}
                self.env = env
                self.v2Q = v2Q
                self.policy = policy

            def getPolicy(self, state):
                policy = self.policy
                max_actions = None
                if policy is None and pi is not None:
                    max_actions = [pi[state]]
                if policy is not None:
                    if state not in policy:
                        return None
                    max_Q = max(policy[state].values())
                    max_actions = [a for a, q in policy[state].items() if q > max_Q - 0.0001]
                if v2Q is not None:
                    qv = v2Q(state)
                    if len(qv) == 0:
                        return None
                    else:
                        max_Q = max(qv.values())
                        max_actions = [a for a, q in qv.items() if q > max_Q - 0.0001]
                return max_actions
            def getValue(self, state):
                if state not in self.v:
                    return 0
                return self.v[state]

        viewer = self.display.viewer
        viewer.geoms = viewer.geoms[:2]
        if Q is not None:
            self.display.displayQValues(MAQ(Q, self.mdp.states, self), currentState=state, message=label)
        elif v is not None:
            # ustate = state if state in self.mdp.nonterminal_states else None
            self.display.displayValues(MAG2(env=self, v=v, states=self.mdp.nonterminal_states, pi=pi, policy=policy, v2Q=v2Q), currentState=state, message=label)
        else:
            self.display.displayNullValues(currentState=state)
        self.viewer = self.display.viewer
        if self.adaptor == "gym":
            return viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class BookGridEnvironment(BerkleyGridEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_book_grid, *args, **kwargs)

class BridgeGridEnvironment(BerkleyGridEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_bridge_grid, *args, **kwargs)

class CliffGridEnvironment(BerkleyGridEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_cliff_grid, *args, **kwargs)

class OpenGridEnvironment(BerkleyGridEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_open_grid, *args, **kwargs)

class SuttonCornerGridEnvironment(BerkleyGridEnvironment): 
    def __init__(self, *args, living_reward=-1, **kwargs):
        super().__init__(sutton_corner_maze, *args, living_reward=living_reward, **kwargs) 

# class FrozenLakeEnvironment(BerkleyGridEnvironment):
#     def __init__(self, *args, living_reward=-1, **kwargs):
#         super().__init__(sutton_corner_maze, *args, living_reward=living_reward, **kwargs)


class FrozenLake(BerkleyGridEnvironment):
    def _get_mdp(self, grid):
        return FrozenGridMDP(grid, is_slippery=self.is_slippery, living_reward=self.step_reward)

    def __init__(self, is_slippery=True, step_reward=0, *args, **kwargs):
        from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
        self.is_slippery=is_slippery
        menv = FrozenLakeEnv(is_slippery=is_slippery)
        self.step_reward = step_reward
        self.menv = menv
        map =  menv.desc.tolist()
        def dc(s):
            s = s.decode("ascii")
            if s == 'F':
                s = ' '
            if s == 'G':
                s = 1
            if s == 'H':
                s = 0
            return s
        grid = [[dc(s) for s in l] for l in map]
        super().__init__(grid=grid, *args, **kwargs)

    def step(self, action):
        return super().step(action)
