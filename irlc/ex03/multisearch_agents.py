"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.

"""
import numpy as np
from irlc.ex01.agent import Agent, train
from irlc.pacman.pacman_environment import GymPacmanEnvironment
from irlc.utils.video_monitor import VideoMonitor


class MultiSearchProblem: 
    """ Implement the needed functionality of the multiagent search problem described in (Her21, Definition 6.1.1).
    Remember there are Q players, q=0,1,...,Q-1, and player q=0 is you. """
    def utility(self, state):
        return 0  # The utility (reward) of the state

    def terminal_test(self, state):
        return False # True if the state is a terminal state

    def get_players(self, state):
        """ Return the number of players in the game. Player 0 is always the Agent, and 1, 2, ... are the opponents """
        return 2  # a two-player game like chess

    def actions(self, state, player):
        """ Returns a dictionary of actions/probabilities of the form actions = {a1: p1, a2: p2, ..} such that
        actions[k] is the chance 'player' will choose action k in the given state.
        Obviously this only has to be defined for player >= 1, since player=0 is you.
        """
        return {0: 0.8, 1: 0.2}

    def get_successor(self, state, action, agentIndex):
        """
        Return the state that occurs when player `agentIndex` takes action `action` in state `state`
        """
        return state 

class PacmanMultisearchModel(MultiSearchProblem):
    def __init__(self):
        self.num_expanded = 0

    def utility(self, state):
        return state.getScore()

    def terminal_test(self, state):
        return state.isWin() or state.isLose()

    def get_players(self, state):
        return state.getNumAgents()

    def actions(self, state, player):
        actions = state.getLegalActions(player)
        return {a: 1/len(actions) for a in actions}

    def get_successor(self, state, action, agentIndex):
        self.num_expanded += 1
        return state.generateSuccessor(agentIndex, action)

class MultiAgentSearchAgent(Agent): 
    def __init__(self, env, depth=2):
        self.depth = depth
        self.minimax_scores = []
        self.model = PacmanMultisearchModel()
        self.num_expanded = []
        super().__init__(env)

    def multisearch_evaluate(self, state):
        # Implement this function. Compute best (score, action)
        best_score, best_action = None, None
        return best_score, best_action

    def pi(self, gameState, k=None):
        self.model.num_expanded = 0
        self_score, action = self.multisearch_evaluate(gameState)
        self.minimax_scores.append(self_score)
        self.num_expanded.append(self.model.num_expanded)
        return action 


class GymMinimaxAgent(MultiAgentSearchAgent):
    """
    Implement minimax search as defined in (Her21, Algorithm 16)
    """
    def __init__(self, env, depth=1):
        super().__init__(env, depth)

    def multisearch_evaluate(self, gameState):
        return self.minimax(gameState, 0, self.depth) # return (optimal score, optimal action)

    def minimax(self, x, q: int, d: int): # q is current player, d is depth (same as in code)
        if d == 0 or self.model.terminal_test(x):
            return self.model.utility(x), None

        q_ = (q + 1) % x.getNumAgents()             # q' = q_: Next agent index (q=0 is you, q >= 1 other agents)
        d_ = d-1 if q == x.getNumAgents()-1 else d  # d' = d_: Decrease depth by one if q==Q-1 (i.e. all ghosts have played a round)

        # TODO: 1 lines missing.
        raise NotImplementedError("Compute the V here (I recommend using a dictionary)")
        best_action = max(V, key=V.get) if q == 0 else min(V, key=V.get) # I use a dictionary V[a] = expected utility.
        return V[best_action], best_action

    def __str__(self):
        return "MiniMaxAgent"

class GymExpectimaxAgent(MultiAgentSearchAgent):
    """
    Implement expectimax search as defined in (Her21, Algorithm 15)

    """
    def __init__(self, env, depth=1):
        super().__init__(env, depth)

    def multisearch_evaluate(self, state):
        return self.expectimax(state, 0, self.depth)

    def expectimax(self, x, q: int, d: int):
        if d == 0 or self.model.terminal_test(x):
            return self.model.utility(x), None

        q_ = (q + 1) % x.getNumAgents()                 # q' = q_: Next agent index (q=0 is you, q >= 1 other agents)
        d_ = d - 1 if q == x.getNumAgents() - 1 else d  # d' = d_: Decrease depth by one if q==Q-1 (i.e. all ghosts have played a round)

        """ Hints: 
         * The following is nearly equivalent to minimax. To obtain the probability of an action, recall that
            self.model.actions(x,q) = {action: probability, ...}  (i.e. a dict with keys: actions and values: probabilities)
        
         * You can loop over them as:
            for u, p_u in self.model.actions(x,q).items(): ...
         * Defining a dict V as in minimax may be of help. """
        # TODO: 1 lines missing.
        raise NotImplementedError("")
        if q == 0: 
            # TODO: 2 lines missing.
            raise NotImplementedError("Return (best score, best action)")
        else: 
            # TODO: 1 lines missing.
            raise NotImplementedError("Return (averaged score, None)")

    def __str__(self):
        return "ExpectiMaxAgent"

def gminmax(layout='minimaxClassic', Agent=None, depth=None, render=False, episodes=1, **kwargs):
    # Code for running an experiment with an agent.
    env = GymPacmanEnvironment(layout=layout, zoom=2, animate_movement=render, **kwargs)
    agent = Agent(env, depth=depth)
    if render:
        env = VideoMonitor(env, agent=agent, agent_monitor_keys=tuple())
    stats, _ = train(env, agent, num_episodes=episodes, verbose=False)
    if episodes > 1:
        winp = np.mean( [s['Accumulated Reward']> 0 for s in stats] )
        print("Ran agent", Agent, "at depth", depth, "on problem", layout)
        print("Avg. win probability:", winp, "path length", np.mean([s['Length'] for s in stats]))
        print("Avg. search nodes expanded:", np.mean(agent.num_expanded), "\n")
    env.close()

def question_minimax(depth=3):
    gminmax(layout='minimaxClassic', Agent=GymMinimaxAgent, depth=depth, render=True)
    gminmax(layout='minimaxClassic', Agent=GymMinimaxAgent, depth=depth, episodes=100)

def question_expectimax(depth=3):
    gminmax(layout='minimaxClassic', Agent=GymExpectimaxAgent, depth=depth, render=True)
    gminmax(layout='minimaxClassic', Agent=GymExpectimaxAgent, depth=depth, episodes=100)

if __name__ == "__main__":
    depth = 2
    question_minimax(depth) 
    question_expectimax(depth) 
