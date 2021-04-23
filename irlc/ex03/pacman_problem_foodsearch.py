"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from itertools import combinations
from irlc.ex02.search_problem import SearchProblem
from irlc.ex03.gsearch import breadthFirstSearch
from irlc.ex03.pacman_problem_positionsearch import GymPositionSearchProblem
from irlc.ex03.pacman_problem_positionsearch import maze_search
from irlc.ex03.pacsearch_agents import DFSAgent, BFSAgent, AStarAgent
from irlc.pacman.gpacman import Directions
from irlc.pacman.gym_game import Actions


class GymFoodSearchProblem(SearchProblem):
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( (x,y), foodGrid ) where
      position: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see irlc/pacman/gym_game.py) of either True or False, specifying remaining food
    """
    def __init__(self):
        self.heuristicInfo = {}  # A dictionary for the heuristic to store information
        super().__init__(initial_state=None)

    def set_initial_state(self, gamestate):
        self.walls = gamestate.getWalls()
        self.startingGameState = gamestate # for the food heuristic
        # search problem state is of the form state = ((x,y), food) 
        # The food-variable is an instance of the Grid class, see Grid class in irlc/pacman/gym_game.py
        self.initial_state = (gamestate.getPacmanPosition(), gamestate.getFood()) 

    def is_terminal(self, state):
        position, food = state
        # returns True if the problem has terminated, i.e. all food has been eaten. Look at the Grid class for ideas.
        return True if problem is solved
        # TODO: 1 lines missing.
        raise NotImplementedError("")

    def available_transitions(self, state):
        transitions = {}
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            (x, y), food = state
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = food.copy()
                # TODO: 1 lines missing.
                raise NotImplementedError("Update the nextFood variable. Pacman eats the food at nextx, nexty. Grid can be index as nextFood[a][b]")
                transitions[direction] = ( ((nextx, nexty), nextFood), 1)
        return transitions


if __name__ == "__main__":
    render = False  # I recommend looking at the DFS agent.

    ### Part A: BFS/DFS
    maze_search(layout='trickySearch', SAgent=BFSAgent, problem=GymFoodSearchProblem(), zoom=2, render=render)  
    maze_search(layout='trickySearch', SAgent=DFSAgent, problem=GymFoodSearchProblem(), zoom=2, render=render)  
