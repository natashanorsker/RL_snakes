"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.

"""
from irlc.ex01.graph_traversal import G222, SmallGraphDP

class SearchProblem: 
    """
    An abstract search problem. Provides the functionality defined in (Her21, Definition 4.2.1).
    """
    def __init__(self, initial_state=None): 
        if initial_state is not None:
            self.set_initial_state(initial_state)

    def set_initial_state(self, state):
        """ Re-set the initial (start) state of the search problem. """
        self.initial_state = state

    def is_terminal(self, state):
        """ Return true if and only if state is the terminal state. """
        raise NotImplementedError("Implement a goal test")

    def available_transitions(self, state):
        """ return the available set of transitions in this state
        as a dictionary {a: (s1, c), a2: (s2,c), ...}
        where a is the action, s1 is the state we transition to when we take action 'a' in state 'state', and
        'c' is the cost we will obtain by that transition.
        """
        raise NotImplementedError("Transition function not impelmented") 

class EnsureTerminalSelfTransitionsWrapper(SearchProblem):
    def __init__(self, search_problem):
        self._sp = search_problem
        super().__init__(search_problem.__dict__.get('initial_state', None))

    def set_initial_state(self, state):
        self._sp.set_initial_state(state)

    @property
    def initial_state(self):
        return self._sp.initial_state

    def is_terminal(self, state):
        return self._sp.is_terminal(state)

    def available_transitions(self, state):
        return {0: (state, 0)} if self.is_terminal(state) else self._sp.available_transitions(state)

class DP2SP(SearchProblem):
    """ This class converts a Deterministic DP environment to a shortest path problem matching the description
    in (Her21, eq. (2.16)).
    """
    def __init__(self, env, initial_state):
        self.env = env
        self.terminal_state = "terminal_state"
        super(DP2SP, self).__init__(initial_state=(initial_state, 0))

    def is_terminal(self, state):
        return state == self.terminal_state

    def available_transitions(self, state):
        """ Implement the dp-to-search-problem conversion described in (Her21, Theorem 4.2.2). Keep in mind the time index is
        absorbed into the state; this means that state = (x, k) where x and k are intended to be used as
        env.f(x, <action>, <noise w>, k).
        As usual, you can set w=None since the problem is assumed to be deterministic.

        The output format should match SearchProblem, i.e. a dictionary with keys as u and values as (next_state, cost).
        """
        if state == self.terminal_state:
            return {0: (self.terminal_state, 0)}
        s, k = state
        # TODO: 3 lines missing.
        raise NotImplementedError("return transtitions as dictionary")


class SmallGraphSP(SearchProblem):
    G = G222


class GraphSP(SearchProblem): 
    """ Implement the small graph graph problem in (Her21, Subsection 2.1.1) """
    G = G222

    def __init__(self, start=2, goal=5):
        self.goal = goal
        super().__init__(initial_state=start)

    def is_terminal(self, state):
        return state == self.goal

    def available_transitions(self, i,k=None):
        # In vertex i, return available transitions i -> j and their cost.
        return {j: (j, cost) for (i_,j), cost in self.G.items() if i_ == i} 

    @property
    def vertices(self):
        # Helper function: Return number of vertices in the graph.
        return len(set([i for edge in self.G for i in edge]))
