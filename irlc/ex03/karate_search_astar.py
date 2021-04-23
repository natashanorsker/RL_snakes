"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc.ex03.karate_search import KarateGraphSP, dist_nodes, run_all_search_methods, plot_solution
from irlc.ex03.gsearch import aStarSearch

def run_astar():
    def heuristic(state, problem=None): 
        """ Return distance from the state to proble.goal. Hint: Use dist_nodes(a,b,problem.pos) """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")
    (actions, path), cost, visited, num_expanded = aStarSearch(Gk, heuristic=heuristic)  
    plot_solution(Gk, path, visited, cost, num_expanded, method='AstarSearch')  


if __name__ == "__main__":
    # Plot the network
    Gk = KarateGraphSP(start=14, goal=16, weighted=True)
    # run_all_search_methods()
    run_astar()
