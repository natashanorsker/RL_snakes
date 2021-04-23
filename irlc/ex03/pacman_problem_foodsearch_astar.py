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

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( position, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For old, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For old, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    pacman, foodGrid = state
    food = foodGrid.asList()
    def dist(a, b): # This might come in handy
        return cachedMazeDistance(a, b, cache=problem.heuristicInfo, gameState=problem.startingGameState)

    if len(food) == 0:
        return 0 # No food remaining; the problem is solved

    return your heuristic
    # TODO: 5 lines missing.
    raise NotImplementedError("")

# Helper functions
def cachedMazeDistance(a, b, cache, gameState):
    if a > b:
        a, b = b, a
    try:
        return cache[a, b]
    except KeyError:
        d = mazeDistance(a, b, gameState)
        cache[a, b] = d
        return d

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)
    """
    prob = GymPositionSearchProblem(gameState, start=point1, goal=point2)
    (actions, states), _, _, _ = breadthFirstSearch(prob)
    return len(  actions )


if __name__ == "__main__":
    render = False
    from irlc.ex03.pacman_problem_foodsearch import maze_search, GymFoodSearchProblem
    ### A^* search and a heuristics function
    maze_search(layout='trickySearch', SAgent=AStarAgent, heuristic=foodHeuristic, problem=GymFoodSearchProblem(), render=render, zoom=2)  
