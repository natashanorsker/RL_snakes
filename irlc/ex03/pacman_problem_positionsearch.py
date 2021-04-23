"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.

"""
import numpy as np
from irlc.ex01.agent import train
from irlc.ex02.search_problem import SearchProblem
from irlc.ex03.pacsearch_agents import DFSAgent, BFSAgent, AStarAgent
from irlc.pacman.gpacman import Directions
from irlc.pacman.gym_game import Actions
from irlc.pacman.pacman_environment import GymPacmanEnvironment
from irlc.utils.video_monitor import VideoMonitor


class GymPositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function. See `SearchProblem` class for details or (Her21, Definition 4.2.1).

    This particular search problem will be used to let Pacman navigate to a particular grid position goal=(x,y) in the
    Pacman level.

    For all pacman search problems, a state x_k in the search model will be of the general form:

        state = ((x, y), extra_information) 

    where (x, y) is the current (integer) location in the maze. The extra_information allows us to specify more
    complex behaviors, such as eating all the food pellets (in which case extra_information would tell us which food
    pellets remain). For a position search problem, all we need to know is the (x, y) location, and so the game state has the form

        state = ((x, y), None) 

    Having defined the states, we also need functionality to figure out what the legal transitions are and so on, i.e.
    a way to know what the rules are in Pacman game. These are all supplied in the `gameState` object, which
    represents the Pacman level. For documentation, see irlc/pacman/gpacman.py and look at the GameState class.
    """
    def __init__(self, gameState=None, costFn=lambda x: 1, goal=(1, 1), start=None):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.costFn = costFn
        self.goal = goal
        super().__init__()
        if gameState is not None:
            self.set_initial_state(gamestate=gameState)
        if start is not None:
            self.initial_state = (start, None)

    def is_terminal(self, state):
        return state[0] == self.goal

    def set_initial_state(self, gamestate):
        if gamestate is not None:
            self.walls = gamestate.getWalls()
            self.initial_state = (gamestate.getPacmanPosition(), None)
        else:
            self.initial_state = None

    def available_transitions(self, state):
        """
           return the available set of transitions in this state
           as a dictionary transitions = {a: (s1, c), a2: (s2,c), ...}
           i.e. transitions[action] = (next_state, cost).
           To reproduce my results, *only* include the actions which are *not* blocked by a wall (There is no need to check those).
       """
        transitions = {}
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            (x, y),_ = state # Get the (x,y) position of the current state
            dx, dy = Actions.directionToVector(action) # The horizontal/vertical change of going in this direction
            nextx, nexty = int(x + dx), int(y + dy)
            # Make sure Pacman does not try to pass through walls. This can be done using self.walls[next_x][next_y]
            # If pacman arrives at a legal position, update the transitions dictionary.
            # The cost of the next state should be computed using the cost function self.costFn.
            # Update transitions[actions] = (next_state, cost) here. Look at self.walls and self.costFn for details.
            # TODO: 4 lines missing.
            raise NotImplementedError("")
        return transitions


def maze_search(layout='tinyMaze', SAgent=None, heuristic=None, problem=None, render=True, zoom=1.0):
    if problem is None:
        problem = GymPositionSearchProblem()
    env = GymPacmanEnvironment(layout=layout, zoom=zoom, animate_movement=render)
    if heuristic is None:
        agent = SAgent(env, problem=problem)
    else:
        agent = SAgent(env, problem=problem, heuristic=heuristic)

    if render:
        env = VideoMonitor(env, agent=agent, agent_monitor_keys=("visitedlist",'path'))
    stats, trajectory = train(env, agent, num_episodes=1,verbose=False, return_trajectory=True)
    reward =  stats[0]['Accumulated Reward']
    length = stats[0]['Length']
    print(f"Environment terminated in {length} steps with reward {reward}\n")
    env.close()


def bfs_search_tiny():
    # Example of a basic interaction with the search problem: Set up a pacman envirionment and a search problem (in this case, go to lower-left corner)
    problem = GymPositionSearchProblem(goal=(1, 1))  # Goal position =(1,1) 
    env = GymPacmanEnvironment(layout='tinyMaze', zoom=2.0)
    agent = BFSAgent(env, problem=problem)
    env = VideoMonitor(env, agent=agent, agent_monitor_keys=("visitedlist", 'path'))
    stats, _ = train(env, agent, num_episodes=1, verbose=False)  

    reward = stats[0]['Accumulated Reward']
    length = stats[0]['Length']
    print(f"Environment terminated in {length} steps with reward {reward}")
    env.close()  

    ### Part A continued: Take a selfie to visualize the search path
    env = GymPacmanEnvironment(layout='tinyMaze', zoom=2.0)
    agent = BFSAgent(env, problem=problem)
    env = VideoMonitor(env, agent=agent, agent_monitor_keys=("visitedlist", 'path'), frame_snapshot_callable=1,
                       snapshot_base="tinymaze_positionsearch.pdf")
    stats, _ = train(env, agent, num_episodes=1, verbose=False)
    env.close()

if __name__ == "__main__":
    ### Part A: Basic BFS on a small problem
    bfs_search_tiny()

    ### Part B: BFS/DFS on a larger problem
    render=False
    maze_search(layout='bigMaze', SAgent=BFSAgent, problem=GymPositionSearchProblem(), render=render, zoom=.5)  
    maze_search(layout='bigMaze', SAgent=DFSAgent, problem=GymPositionSearchProblem(), render=render, zoom=.5)  
