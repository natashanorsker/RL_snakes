"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.

"""
import numpy as np

from irlc.ex03.multisearch_agents import gminmax, MultiAgentSearchAgent

class GymAlphaBetaAgent(MultiAgentSearchAgent):
    """ Implement alpha-beta search as defined in (Her21, Algorithm 17)"""

    def __init__(self, env, depth=1):
        super().__init__(env, depth)

    def multisearch_evaluate(self, state):
        return self.alpha_beta(state, 0, self.depth, alpha=-np.inf, beta=np.inf)

    def alpha_beta(self, x, q: int, d: int, alpha: float, beta: float):
        if d == 0 or self.model.terminal_test(x):
            return self.model.utility(x), None
        if q == 0:
            return self.MaxValue(x, q, d, alpha, beta)
        else:
            return self.MinValue(x, q, d, alpha, beta)

    def MaxValue(self, x, q, d, alpha, beta):
        maxScore = -np.inf
        maxAction = None

        for u in self.model.actions(x, q):
            successor = self.model.get_successor(x, u, q)
            score, _ = self.alpha_beta(successor, 1, d, alpha, beta)
            # TODO: 6 lines missing.
            raise NotImplementedError("Update maxAction, maxScore, alpha here")
        return maxScore, maxAction

    def MinValue(self, x, q, d, alpha, beta):
        q_ = (q + 1) % self.model.get_players(x)  # next player q'
        d_ = d - 1 if q == x.getNumAgents() - 1 else d  # next depth d'
        minScore = np.inf
        minAction = None

        for u in self.model.actions(x, q):
            successor = self.model.get_successor(x, u, q)
            score, _ = self.alpha_beta(successor, q_, d_, alpha, beta)
            # TODO: 7 lines missing.
            raise NotImplementedError("If you got the _searchMax updates correct, the changes here should be obvious (see pseudo code)")
        return minScore, minAction

    def __str__(self):
        return "AlphaBetaAgent"

def question_alphabeta(depth=3):
    gminmax(layout='minimaxClassic', Agent=GymAlphaBetaAgent, depth=depth, render=True)
    gminmax(layout='minimaxClassic', Agent=GymAlphaBetaAgent, depth=depth, episodes=100)

if __name__ == "__main__":
    depth = 3
    from irlc.ex03.multisearch_agents import question_minimax, question_expectimax
    question_minimax(depth)  
    question_expectimax(depth)  
    question_alphabeta(depth)  
