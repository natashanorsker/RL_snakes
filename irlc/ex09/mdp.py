"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
from gym import Env
from collections import defaultdict
from tqdm import tqdm
import sys

class MDP: 
    """
    The basic MDP class. It defines three main components:

    1) In a state s, the available actions as
        > mdp.A(s)
    2) In a state s, taking action a, the availble transitions
        > mdp.Psr(s, a)
        This function returns a dictionary of the form:
        > { (sp1, r1): p1, (sp2, r2): p2, ...}
        such that
            P(S' = sp, R = r | S=s, A=a) = mdp.Psr[ (sp,r) ]

    3) A terminal test which is true if state is terminal
        > mdp.is_terminal(state)

    4) An initial state distribution of the form of a dictionary:
        {s: p(S_0=s), ...}

    In addition to this, it also defines a list of all states and all non-terminal states:
        >>> mdp.states
        >>> mdp.nonterminal_states

    The set of states is computed on-the-fly by finding all states that can be reached from the initial state distribution.
    In other words, when you first call mdp.states, it may take some time for the call to finish.
    The advantage of this approach is you could implement and MDP with an infinite amount of states and as long as you never
    call mdp.states you wont run out of memory.

    """
    def __init__(self, initial_state=None, verbose=False): 
        self.verbose=verbose
        self.initial_state = initial_state  # Optional: In this case we start in this state with probability 1. 
        # The following variables that begin with _ are used to cache computations. The reason why we don't compute them
        # up-front is because their computation may be time-consuming and they might not be needed.
        self._states = None
        self._nonterminal_states = None
        self._terminal_states = None

    def is_terminal(self, state): 
        return False # Return true if the given state is terminal.

    def Psr(self, state, action):
        raise NotImplementedError("Return state distribution as a dictionary (see class documentation)")

    def A(self, state):
        """ Return the set of available actions in state state """
        raise NotImplementedError("Return set/list of actions in given state A(s) = {a1, a2, ...}") 

    def initial_state_distribution(self):
        """ return a dictionary of the form {s: ps, ...} such that P(S_0=s) = ps """
        if self.initial_state is not None:
            return {self.initial_state: 1}
        else:
            raise Exception("Either specify the initial state, or implement this method.")

    @property
    def nonterminal_states(self):
        if self._nonterminal_states is None:
            self._nonterminal_states = [s for s in self.states if not self.is_terminal(s)]
        return self._nonterminal_states

    @property
    def states(self):
        if self._states is None:
            next_chunk = set(self.initial_state_distribution().keys())
            all_states = list(next_chunk)
            while True:
                new_states = set()
                for s in tqdm(next_chunk, file=sys.stdout) if self.verbose else next_chunk:
                    if self.is_terminal(s):
                        continue
                    for a in self.A(s):
                        new_states = new_states  | {sp for sp, r in self.Psr(s, a)}

                new_states  = [s for s in new_states if s not in all_states]
                if len(new_states) == 0:
                    break
                all_states += new_states
                next_chunk = new_states
            self._states = list(set(all_states))

        return self._states

def get_connected_states(mdp, initial):

    pass

def rng_from_dict(d):
    """ Helper function. If d is a dictionary {x1: p1, x2: p2, ...} then this will sample an x_i with probability p_i """
    w, pw = zip(*d.items())             # seperate w and p(w)
    i = np.random.choice(len(w), p=pw)  # Required because numpy cast w to array (and w may contain tuples)
    return w[i]

class MDP2GymEnv(Env):
    def __init__(self, mdp):
        self.mdp = mdp
        self.state = None
        self.P = {s: {a: 1 for a in self.mdp.A(s)} for s in self.mdp.nonterminal_states}

    def reset(self):
        ps = self.mdp.initial_state_distribution()
        self.state = rng_from_dict(ps) # np.random.choice(a=w, p=pw)
        return self.state

    def step(self, action):
        ps = self.mdp.Psr(self.state, action)
        self.state, reward = rng_from_dict(ps)
        done = self.mdp.is_terminal(self.state)
        return self.state, reward, done, {}

class GymEnv2MDP(MDP):
    def __init__(self, env):
        super().__init__()
        self._states = list(range(env.observation_space.n))
        if hasattr(env, 'env'):
            env = env.env
        self._terminal_states = []
        for s in env.P:
            for a in env.P[s]:
                for (pr, sp, reward, done) in env.P[s][a]:
                    if done:
                        self._terminal_states.append(sp)

        self._terminal_states = set(self._terminal_states)
        self.env = env

    def is_terminal(self, state):
        return state in self._terminal_states

    def A(self, state):
        return list(self.env.P[state].keys())

    def Psr(self, state, action):
        d = defaultdict(float)
        for (pr, sp, reward, done) in self.env.P[state][action]:
            d[ (sp, reward)] += pr
        return d

if __name__ == '__main__':
    import gym
    env = gym.make("FrozenLake-v0")
    mdp = GymEnv2MDP(env)
    from irlc.ex09.value_iteration import value_iteration
    value_iteration(mdp)

    mdp = GymEnv2MDP(gym.make("FrozenLake-v0")) 
    print("N = ", mdp.nonterminal_states)
    print("S = ", mdp.states)
    print("Is state 3 terminal?", mdp.is_terminal(3), "is state 11 terminal?", mdp.is_terminal(11)) 
    state = 0 
    print("A(S=0) =", mdp.A(state))
    action = 2
    mdp.Psr(state, action)  # Get transition probabilities
    for (next_state, reward), Pr in mdp.Psr(state, action).items():
        print(f"P(S'={next_state},R={reward} | S={state}, A={action} ) = {Pr:.2f}") 
