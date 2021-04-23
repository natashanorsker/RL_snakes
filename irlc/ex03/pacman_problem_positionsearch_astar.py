"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
from irlc.ex03.pacsearch_agents import AStarAgent
from irlc.ex03.pacman_problem_positionsearch import GymPositionSearchProblem, maze_search

def manhattanHeuristic(state, problem):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1, _ = state # Get the XY location of the current state and goal. Remember that state = ((x,y), None).
    xy2 = problem.goal
    # TODO: 1 lines missing.
    raise NotImplementedError("return the distance to goal according to the Manhattan distance.")

def euclideanHeuristic(state, problem):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1, _ = state
    xy2 = problem.goal
    return np.sqrt( ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5 )

if __name__ == "__main__":
    # A^* search and a heuristics function
    render = False
    maze_search(layout='bigMaze', SAgent=AStarAgent, heuristic=manhattanHeuristic, problem=GymPositionSearchProblem(), render=render, zoom=.5)  
