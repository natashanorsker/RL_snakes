import numpy as np
from utils import defaultdict2
from collections import OrderedDict


# Base Agent Class:
class Agent:
    """ Main agent class. See (Her21, Subsection 1.4.3) for additional details.  """

    def __init__(self, env):
        self.env = env

    def pi(self, s, k=None):
        """ Compute the policy pi_k(s).
        For discrete application (dynamical programming/search and reinforcement learning), k is discrete k=0, 1, 2, ...
        For control applications, k is continious and denote simulation time t.

        :param s: Current state
        :param k: Current time index.
        :return: action
        """
        return self.env.action_space.sample()

    def train(self, s, a, r, sp, done=False):
        """
        Called at each step of the simulation after a = pi(s,k) and environment transition to sp.
        Allows the agent to learn from experience

        :param s: Current state x_k
        :param a: Action taken
        :param r: Reward obtained by taking action a_k in x_k
        :param sp: State environment transitioned to x_{k+1}
        :param done: Whether environment terminated when transitioning to sp
        :return: None
        """
        pass

    def __str__(self):
        """ Optional: A unique name for this agent. Used for labels when plotting, but can be kept like this. """
        return super().__str__()

    def extra_stats(self):
        """ Optional: Can be used to record extra information from the Agent while training.
        You can safely ignore this method. """
        return {}


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
        d = {s: {a: Q for a, Q in Qs.items()} for s, Qs in self.q_.items()}
        return d


# class QAgent(TabularAgent):
#     """
#     Implement the Q-learning agent here. Note that the Q-datastructure already exist
#     (see agent class for more information)
#     """
#     def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1):
#         self.alpha = alpha
#         super().__init__(env, gamma, epsilon)
#
#     def pi(self, s, k=None):
#         """
#         Return current action using epsilon-greedy exploration. Look at the TabularAgent class
#         for ideas.
#         """
#         return self.pi_eps()
#         # raise NotImplementedError("Implement function body")
#
#     def train(self, s, a, r, sp, done=False):
#         """
#         Implement the Q-learning update rule, i.e. compute a* from the Q-values.
#         As a hint, note that self.Q[sp][a] corresponds to q(s_{t+1}, a) and
#         that what you need to update is self.Q[s][a] = ...
#         """
#
#         #raise NotImplementedError("Implement function body")
#
#     def __str__(self):
#         return f"QLearner_{self.gamma}_{self.epsilon}_{self.alpha}"




#################
class QAgent(TabularAgent):
    """
    Implement the Q-learning agent (SB18, Section 6.5)
    Note that the Q-datastructure already exist, as do helper functions useful to compute an epsilon-greedy policy
    (see TabularAgent class for more information)
    """
    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1):
        self.alpha = alpha
        super().__init__(env, gamma, epsilon)

    def pi(self, s, k=None):
        """
        Return current action using epsilon-greedy exploration. Look at the TabularAgent class
        for ideas.
        """
        return self.pi_eps()
        raise NotImplementedError("Implement function body")

    def train(self, s, a, r, sp, done=False):
        """
        Implement the Q-learning update rule, i.e. compute a* from the Q-values.
        As a hint, note that self.Q[sp,a] corresponds to q(s_{t+1}, a) and
        that what you need to update is self.Q[s, a] = ...

        You may want to look at self.Q.get_optimal_action(state) to compute a = argmax_a Q[s,a].
        """
        astar = self.Q.get_optimal_action(sp)
        maxQ = self.Q[sp][astar]
        self.Q[s][a] = self.Q[s][a] + self.alpha * (r + self.gamma * maxQ - self.Q[s][a])
        # raise NotImplementedError("Implement function body")

    def __str__(self):
        return f"QLearner_{self.gamma}_{self.epsilon}_{self.alpha}"

