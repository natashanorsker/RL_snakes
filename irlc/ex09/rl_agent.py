"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
from irlc.utils.common import defaultdict2
from collections import OrderedDict
from irlc import Agent

class TabularAgent(Agent):
    """
    The self.Q variable is a custom datastructure to save the Q(s,a)-values.
    There are multiple ways to implement the Q-values, most of which will get us in troubles
    if we want to implement the examples in Sutton.

    I solve this using the TabularQ class above. What it amounts to is that you can access Q-values as

    >>> q_value = self.Q[s,a]

    and set them as

    >>> self.Q[s,a] = new_q_value

    It also provides helpful methods. For instance if Q[s,a1] = q1, Q[s,a2] = q2, ...
    then

    >>> actions, qs = self.Q.Qs(s)

    defines actions=[a1,a2,...] and qs=[q1,q2,...]
    """
    def __init__(self, env, gamma=0.99, epsilon=0):
        super().__init__(env)
        self.gamma, self.epsilon = gamma, epsilon
        self.Q = TabularQ(env)

    def pi(self, s, k=None):
        return self.random_pi(s)

    def pi_eps(self, s):
        """ Implement epsilon-greedy exploration. Return random action with probability self.epsilon,
        else be greedy wrt. the Q-values. """
        return self.random_pi(s) if np.random.rand() < self.epsilon else self.Q.get_optimal_action(s)

    def random_pi(self, s):
        """ Generates a random action given s.

        It might seem strange why this is useful, however many policies requires us to to random exploration, and it is
        possible to get the method wrong.
        We will implement the method depending on whether self.env defines an MDP or just contains an action space.
        """
        if hasattr(self.env, 'P'):
            return np.random.choice(list(self.env.P[s].keys()))
        else:
            return self.env.action_space.sample()

class ValueAgent(TabularAgent): 
    """
    This is a simple wrapper class around the Agent class above. It fixes the policy and is therefore useful for doing
    value estimation.
    """
    def __init__(self, env, gamma=0.95, policy=None, v_init_fun=None): 
        self.env = env
        self.policy = policy  # policy to evaluate
        """ Value estimates. 
        Initially v[s] = 0 unless v_init_fun is given in which case v[s] = v_init_fun(s). """
        self.v = defaultdict2(float if v_init_fun is None else v_init_fun) 
        super().__init__(env, gamma=gamma)

    def pi(self, s, k=None):  
        return self.random_pi(s) if self.policy is None else self.policy(s) 

    def value(self, s):
        return self.v[s]


class TabularQ:
    """
    Tabular Q-values. This is a helper class for the Q-agent to store Q-values without too much hassle with
    state-dependent action spaces and so on.
    """
    def __init__(self, env):
        # This may need to be changed (s in P)
        qfun = lambda s: OrderedDict({a: 0 for a in (env.P[s] if hasattr(env, 'P') else range(env.action_space.n))})
        self.q_ = defaultdict2(lambda s: qfun(s))
        self.env = env

    def get_Qs(self, state):
        (actions, Qa) = zip(*self.q_[state].items())
        return tuple(actions), tuple(Qa)

    def get_optimal_action(self, state):
        actions, Qa = self.get_Qs(state)
        a_ = np.argmax(np.asarray(Qa) + np.random.rand(len(Qa)) * 1e-8)
        return actions[a_]

    def __getitem__(self, state_comma_action):
        s, a = state_comma_action
        return self.q_[s][a]

    def __setitem__(self, state_comma_action, q_value):
        s, a = state_comma_action
        self.q_[s][a] = q_value

    def items(self):  # not sure this is used
        raise Exception()
        return self.q_.items()

    def to_dict(self):
        # Convert to a regular dictionary
        d = {s: {a: Q for a, Q in Qs.items() } for s,Qs in self.q_.items()}
        return d
