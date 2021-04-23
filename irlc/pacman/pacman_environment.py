"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import gym
from gym import RewardWrapper
from irlc.pacman.gpacman import Directions
from irlc.pacman.gym_game import Agent
from irlc.pacman.layout import getLayout
from irlc.pacman.ghostAgents import RandomGhost
from irlc.pacman.gpacman import ClassicGameRules
from irlc.pacman import textDisplay
from irlc.utils.common import ExplicitActionSpace


class GymPacmanEnvironment(gym.Env):
    """
    Really messy pacman environment class. I do not recommend reading this code.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }

    def __init__(self, animate_movement=False, layout='mediumGrid', zoom=2.0, num_ghosts=4, frame_time=0.05, ghost_agent=None,
                 layout_str=None):
        self.metadata['video_frames_per_second'] = 1/frame_time
        GAgent = RandomGhost
        if ghost_agent is not None:
            GAgent = ghost_agent
        timeout = 30
        self.ghosts = []
        for i in range(num_ghosts):
            self.ghosts.append(GAgent(i+1))
        if layout_str is not None:
            from irlc.pacman import layout
            self.layout = layout.Layout(layout_str.strip().splitlines())
        else:
            self.layout = getLayout( layout )
            if self.layout is None:
                raise Exception("Layout file not found", layout)
        self.pacman = Agent(index=0)
        self.rules = ClassicGameRules(timeout)
        self.options_frametime = frame_time
        self.options_showGhosts = True
        self.first_person_graphics = False

        self.animate_movement = animate_movement
        ### Setup displays.
        self.null_display = textDisplay.NullGraphics()
        # self.options_frametime = frame_time
        self.options_zoom = zoom
        textDisplay.SLEEP_TIME = frame_time # self.options.frameTime
        self.text_display = textDisplay.PacmanGraphics()
        self.game = None #self._new_game()

        self.SOUTH = Directions.SOUTH

        self.pac_actions = [Directions.STOP, Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]
        self._actions_gym2pac = {v:v for i,v in enumerate(self.pac_actions)}
        self._actions_pac2gym = {v:v for i,v in enumerate(self.pac_actions)}
        self.action_space = ExplicitActionSpace(self) # Wrapper environments copy the action space.

        # Helper class to set varying action spaces.
        class PP:
            def __init__(self, pac2gym):
                self.pac2gym = pac2gym
            def __getitem__(self, state):
                return {self.pac2gym[pm_action]: "new_state" for pm_action in state.getLegalActions()}
        self.P = PP(self._actions_pac2gym)
        self.visitedlist = None
        self.ghostbeliefs = None # for display purposes
        self.path = None


    def _setup_graphics_display(self):
        if not hasattr(self, 'graphics_display'):
            from irlc.pacman import gym_graphicsDisplay
            # import graphicsDisplay
            if self.first_person_graphics:
                self.graphics_display = gym_graphicsDisplay.FirstPersonPacmanGraphics(self.options_zoom, self.options_showGhosts, frameTime=self.options_frametime)
                self.graphics_display.ghostbeliefs = self.ghostbeliefs
            else:
                self.graphics_display = gym_graphicsDisplay.PacmanGraphics(self.options_zoom, frameTime=self.options_frametime)

        self.game.display = self.graphics_display
        self.game.display.visitedlist = self.visitedlist
        if not hasattr(self.game.display, 'viewer'):
            self.game.display.initialize(self.game.state.data)

    def _new_game(self):
        catchExceptions = False
        beQuiet = True
        self.rules.quiet = True
        game = self.rules.newGame(self.layout, self.pacman, self.ghosts, beQuiet, catchExceptions)
        return game

    def reset(self):
        self.game = self._new_game()
        if self.animate_movement:
            self._setup_graphics_display()

        if self.animate_movement:
            self.game.display.master_render(self.game.state.data)
        self.game.numMoves = 0
        return self.state

    def close(self):
        if hasattr(self, 'env'):
            self.env.close()
        if hasattr(self.game, 'display') and hasattr(self.game.display, 'viewer'):
            if self.game.display.viewer is not None:
                self.game.display.viewer.close()
            self.game.display.viewer = None
            # print("Shutting down viewer...")

    @property
    def state(self):
        if self.game is None:
            return None
        return self.game.state.deepCopy()

    def process_action(self, agent_index_, action_, animate=True):
        if action_ not in self.P[self.game.state] and agent_index_ == 0 and animate:
            print("Agent tried action", action_)
            print("available actions ", self.P[self.game.state])
            action_ = 0
            raise Exception()
        action_ = self._actions_gym2pac[action_]
        self.game.state = self.game.state.generateSuccessor(agentIndex=agent_index_, action=action_)

        if hasattr(self.game, 'display') and self.animate_movement:
            self.game.display.update(self.game.state.data, animate=self.animate_movement, ghostbeliefs=self.ghostbeliefs, path=self.path, visitedlist=self.visitedlist)  # Change the display
        self.game.rules.process(self.game.state, self.game)  # Allow for game specific conditions (winning, losing, etc.)

    def step(self, action):
        if self.animate_movement:
            self._setup_graphics_display()

        r_ = self.game.state.getScore()
        self.process_action(agent_index_=0, action_=action) # Calculate effect of this action.

        for agentIndex in range(1, len(self.game.agents)):
            done = self.game.gameOver or self.game.state.isWin() or self.game.state.isLose()
            if done: break
            a = self.game.agents[agentIndex].getAction(self.state)
            self.process_action(agent_index_=agentIndex, action_=self._actions_pac2gym[a])

        if self.game.gameOver and not (self.game.state.isWin() or self.game.state.isLose() ):
            raise Exception("Game ended in a draw? strange")
        done = self.game.gameOver or self.game.state.isWin() or self.game.state.isLose()
        reward = self.game.state.getScore() - r_
        observation = self.state
        # if done:
        #     self.game = None
        return observation, reward, done, {}

    def get_keys_to_action(self):
        from pyglet.window import key
        dd = {(key.LEFT,): Directions.WEST,
                (key.RIGHT,): Directions.EAST,
                (key.UP,): Directions.NORTH,
                (key.DOWN,): Directions.SOUTH}
        return {k: self._actions_pac2gym[v] for k,v in dd.items()}

    def render(self, mode='human', visitedlist=None, ghostbeliefs=None, path=None):
        if mode in ["human", 'rgb_array']:
            self.visitedlist = visitedlist
            self._setup_graphics_display()
            self.path = path
            self.ghostbeliefs = ghostbeliefs
            self.game.display.master_render(self.game.state.data, ghostbeliefs=ghostbeliefs, path=path, visitedlist=visitedlist)
            return self.graphics_display.viewer.render(return_rgb_array=mode=="rgb_array")
        elif mode in ['ascii']:
            self.game.display = self.text_display
            return self.game.display.draw(self.game.state)
        else:
            raise Exception("Bad video mode", mode)

    @property
    def viewer(self):
        if hasattr(self, 'graphics_display') and hasattr(self.graphics_display, 'viewer'):
            return self.graphics_display.viewer
        else:
            return None


class PacmanWinWrapper(RewardWrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.env.game.state.isWin():
            assert(observation.isWin())
            reward = 1
        else:
            reward = 0
        return observation, reward, done, info

if __name__ == "__main__":
    """
    Example usage:
    """
    layout = 'mediumClassic'
    env = GymPacmanEnvironment(layout=layout)
    import time

    experiment = "experiments/pacman_q"
    if True:
        from irlc.utils.player_wrapper import PlayWrapper
        from irlc.ex01.agent import Agent, train

        agent = Agent(env)
        agent = PlayWrapper(agent, env)
        train(env, agent, num_episodes=1)

    env.unwrapped.close()
    time.sleep(0.1)
    env.close()
