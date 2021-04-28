"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc.ex09.value_iteration import value_iteration
from irlc.ex09.mdp import GymEnv2MDP
from irlc import TabularAgent
import numpy as np

class ValueIterationAgent(TabularAgent):
    def __init__(self, env, mdp=None, gamma=1, epsilon=0, **kwargs):
        super().__init__(env)
        if mdp is None: # Try to see if MDP can easily be found from environment.
            if hasattr(env, 'mdp'):
                mdp = env.mdp
            elif hasattr(env, 'P'):
                mdp = GymEnv2MDP(env)
            else:
                raise Exception("Must supply a MDP so I can plan!")
        self.epsilon = epsilon
        self.policy, self.v = value_iteration(mdp, gamma)
        # raise NotImplementedError("")

    def pi(self, s, k=0):
        """ With probability (1-epsilon), the take optimal action as computed using value iteration
         With probability epsilon, take a random action. You can do this using return self.random_pi(s)
        """
        if np.random.rand() < self.epsilon:
            return self.random_pi(s)
        else: 
            return self.policy[s]
            # raise NotImplementedError("Implement function body")

if __name__ == "__main__":
    from irlc.gridworld.gridworld import BookGridEnvironment, SuttonCornerGridEnvironment
    env = SuttonCornerGridEnvironment(living_reward=-1)
    from irlc import VideoMonitor, train
    agent = ValueIterationAgent(env, mdp=env.mdp)                   # Make an agent
    # Let's have a cute little animation. Try to leave out the agent_monitor_keys line to see what happens.
    env = VideoMonitor(env, agent=agent, agent_monitor_keys=('v',)) # The monitor-keys stuff is a bit hacky, but we need a mechanism to tell the environments about properties in the agent
    train(env, agent, num_episodes=20)                             # Train for 100 episodes
    env.savepdf("smallgrid.pdf") # Take a snapshot of the final configuration
    env.close() # Whenever you use a VideoMonitor, call this to avoid a dumb openglwhatever error message on exit
