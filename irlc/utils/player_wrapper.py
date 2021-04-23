"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import matplotlib
from gym import logger
from irlc.ex01.agent import Agent

try:
    # Using this backend apparently clash with scientific mode. Not sure why it was there in the first place so
    # disabling it for now.

    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warn('failed to set matplotlib backend, plotting will not work: %s' % str(e))
    plt = None

import matplotlib
from gym import logger

try:
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.warn('failed to set matplotlib backend, plotting will not work: %s' % str(e))
    plt = None

class AgentWrapper(Agent):
    """Wraps the environment to allow a modular transformation.

    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.

    .. note::

        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.

    """
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.agent, name)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def pi(self, state,k=None):
        return self.agent.pi(state, k=k)
        # return self.env.step(action)

    def train(self, *args, **kwargs):
        return self.agent.train(*args, **kwargs)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.agent)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.agent.unwrapped

PAUSE_KEY = ord('p')
SPACEBAR = "_SPACE_BAR_PRESSED_"
class PlayWrapper(AgentWrapper):
    def __init__(self, agent, env, keys_to_action=None):
        super().__init__(agent, env)
        if keys_to_action is None:
            if hasattr(env, 'get_keys_to_action'):
                keys_to_action = env.get_keys_to_action()
            elif hasattr(env.unwrapped, 'get_keys_to_action'):
                keys_to_action = env.unwrapped.get_keys_to_action()
            else:
                assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                              "please specify one manually"

        self.keys_to_action = keys_to_action
        relevant_keys = set(sum(map(list, keys_to_action.keys()), []))
        self.env = env
        self.human_wants_restart = False
        self.human_sets_pause = False
        self.human_agent_action = -1
        self.human_demand_autoplay = False

    # space bar: 0x0020
    def key_press(self,key, mod):
        if key == 0xff0d: self.human_wants_restart = True
        if key == PAUSE_KEY:
            self.human_demand_autoplay = not self.human_demand_autoplay
            a = -1
        else:
            a = self.keys_to_action.get((key,), -1)

        if key == 0x0020:
            a = SPACEBAR
        self.human_agent_action = a

    def key_release(self,key, mod):
        pass

    def _get_viewer(self):
        return self.env.viewer if hasattr(self.env, 'viewer') else self.env.unwrapped.viewer

    def setup(self):
        self._get_viewer().window.on_key_press = self.key_press
        self._get_viewer().window.on_key_release = self.key_release

    def pi(self,state, k=None):
        pi_action = super().pi(state, k=k) # make sure super class pi method is called in case it has side effects.
        self.setup()
        import time
        while True:
            time.sleep(0.02)
            self._get_viewer().window.dispatch_events()
            a = self.human_agent_action
            if a == SPACEBAR or self.human_demand_autoplay:
                # Just do what the agent wanted us to do
                action_okay = True
                a = pi_action
            elif hasattr(self.env, 'P'):
                if len(self.env.P[state]) == 1 and a != -1:
                    a = next(iter(self.env.P[state]))
                action_okay = a in self.env.P[state]
            elif self.env.action_space is not None:
                action_okay = self.env.action_space.contains(a)
            else:
                action_okay = a != -1
            if action_okay:
                self.human_agent_action = -1
                break
        return a

def main():
    from irlc.utils.berkley import BerkleyBookGridEnvironment
    from irlc.ex11.sarsa_agent import SarsaAgent
    from irlc.ex01.agent import train
    from irlc.utils.berkley import VideoMonitor
    env = BerkleyBookGridEnvironment(adaptor='gym')
    agent = SarsaAgent(env, gamma=0.95, alpha=0.5)
    # """
    agent = PlayWrapper(agent, env)

    env = VideoMonitor(env, agent=agent, video_file="videos/SarsaGridworld.mp4", fps=30, continious_recording=True,
                       label="SADSF",
                       monitor_keys=("Q",))
    # """
    # env.reset()
    # env.render()
    train(env, agent, num_episodes=3)
    env.close()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='MontezumaRevengeNoFrameskip-v4', help='Define Environment')
    # args = parser.parse_args()
    # env = gym.make(args.env)
    # play(env, zoom=4, fps=60)

if __name__ == "__main__":
    main()
