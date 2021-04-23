"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
from irlc.ex09.mdp import MDP
import seaborn as sns

# action space available to the agent
UP,RIGHT, DOWN, LEFT = 0, 1, 2, 3 
class SmallGridworldMDP(MDP):
    def __init__(self, rows=4, cols=4):
        self.rows = rows # Number of rows, columns.
        self.cols = cols
        super().__init__(initial_state=(rows//2, cols//2) ) # Initial state is in the middle of the board.

    def A(self, state):
        return [UP, DOWN, RIGHT, LEFT] # All four directions available.

    def Psr(self, state, action):
        row, col = state # state is in the format  state = (row, col)

        if action == UP:
            row -= 1
        if action == DOWN:
            row += 1
        if action == LEFT:
            col += 1
        if action == RIGHT:
            col -= 1

        col = min(self.cols-1, max(col, 0))
        row = min(self.rows-1, max(row, 0))
        reward = -1 # Always get a reward of -1
        next_state = (row, col)
        # Note that P(next_state, reward | state, action) = 1 because environment is deterministic
        return {( next_state, reward): 1}

    def is_terminal(self, state):
        row, col = state
        return (row == 0 and col == 0) or (row == self.rows-1 and col == self.cols-1) 


def plot_value_function(env, v):
    A = np.zeros((env.rows, env.cols))
    for (row, col) in env.nonterminal_states:
        A[row, col] = v[(row,col)]
    sns.heatmap(A, cmap="YlGnBu", annot=True, cbar=False, square=True, fmt='g')
