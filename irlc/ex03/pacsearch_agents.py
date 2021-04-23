"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import time
from collections import OrderedDict
from irlc.ex01.agent import Agent
from irlc.ex03.gsearch import depthFirstSearch, breadthFirstSearch, aStarSearch, uniformCostSearch
from irlc.pacman.pacman_environment import GymPacmanEnvironment


class GymSearchAgent(Agent):
    name = "SearchAgent"
    def __init__(self, env, problem):
        self.problem = problem
        print(f'[{self.__str__()}] using problem type', problem.__class__.__name__)
        self.visitedlist = None
        self.path = None
        self.actions = None
        super().__init__(env)

    def fix_state_(self, state):
        if isinstance(self.env, GymPacmanEnvironment):
            return state[0] if isinstance(state[0], tuple) else state
        else:
            return state

    def pi(self, state, k=None):
        if self.actions is None:
            starttime = time.time()
            (self.actions, self.path), self.totalCost, self.visitedlist, self.num_expanded = self.setup_actions(state)
            self.path = [self.fix_state_(s) for s in self.path]
            dt = time.time() - starttime
            print(f"[{self.__str__()}] Number of search nodes visited {self.num_expanded}")
            print(f'[{self.__str__()}] Path found with total cost of {self.totalCost} in {dt:2.3f} seconds')

            vl_ = self.visitedlist
            self.visitedlist = OrderedDict()
            for k, v in vl_.items():
                if not isinstance(k, str):
                    self.visitedlist[self.fix_state_(k)] = v
            self.actionIndex = 0

        """ self.actions contains all the actions we have planned, 
        and self.actionIndex is the current time step. 
        Return the right action from self.actions and update the actionIndex
        """
        # TODO: 2 lines missing.
        raise NotImplementedError("Return the action the search agent should take in this time step; update action index.")

    def setup_actions(self, state):
        raise NotImplementedError()

    def train(self, s, a, r, sp, done=False):
        if done:
            self.actions = None

    def __str__(self):
        return self.name


class BFSAgent(GymSearchAgent):
    name = "BFS Agent"
    def setup_actions(self, state):
        self.problem.set_initial_state(state)
        return breadthFirstSearch(self.problem)

class DFSAgent(GymSearchAgent):
    name = "DFS Agent"
    def setup_actions(self, state): 
        """ Setup actions using Uniform Cost Search. See BFS agent for the (very simple) idea """
        # TODO: 2 lines missing.
        raise NotImplementedError("Implement function body")

class UniformCostAgent(GymSearchAgent):
    name = "Uniform Cost Search Agent"
    def setup_actions(self, state): 
        """ Setup actions using Uniform Cost Search. See BFS agent for the (very simple) idea """
        # TODO: 2 lines missing.
        raise NotImplementedError("Implement function body")

class AStarAgent(GymSearchAgent):  # Note this is optional
    name = "A* Search Agent"
    def __init__(self, env, problem, heuristic=None):
        self.heuristic=heuristic
        super().__init__(env, problem)
        print(f"[{self.__str__()}] using heuristic {heuristic}")

    def setup_actions(self, state): 
        """ Return the search solution (see the BFS agent, but remember to also pass the self.heuristic!) """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")
        return aStarSearch(self.problem, heuristic=self.heuristic)
