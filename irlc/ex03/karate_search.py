"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from irlc import savepdf
from irlc.ex02.search_problem import GraphSP
from irlc.ex03.gsearch import breadthFirstSearch, uniformCostSearch, depthFirstSearch, aStarSearch


class KarateGraphSP(GraphSP):
    def __init__(self, start, goal, weighted=True):
        super().__init__(start, goal)
        self.Gnx = nx.karate_club_graph() # We use the networkx package for layout
        self.pos = nx.spring_layout(self.Gnx, seed=14) # Lay out nodes and store (x, y) position of all nodes in pos.
        G = dict()  # Turn the networkx representation into our familiar dictionary representation.
        for (a,b) in self.Gnx.edges:
            G[(a,b)] = G[(b,a)] = (dist_nodes(a, b, self.pos) if weighted else 1.0)
        self.G = G # the superclass will define the transitions correctly.

def dist_nodes(a, b, pos):  # Euclidian distance.
    return np.sqrt(np.sum((pos[a] - pos[b])**2))

def draw_network(Gk, path=None):
    pos = Gk.pos
    Gnx = Gk.Gnx
    nx.draw(Gnx, pos, node_color='w', edgecolors='k')
    nx.draw_networkx_labels(Gnx, pos=pos)
    if path is not None:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(Gnx,pos,nodelist=path,node_color='r')
        nx.draw_networkx_edges(Gnx,pos,edgelist=path_edges,edge_color='r',width=3)
    plt.axis('equal')

def plot_solution(Gk, path, visited, cost, num_expanded, method):
    print("Solved network using method:", method)
    print("Shortest path found! cost was:", cost)
    print(f"{num_expanded} search nodes were expanded. Order in which nodes are visisted:")
    print("> ", [v for v in visited])
    draw_network(Gk, path)
    plt.title("Karate network and shortest as found using {method}")
    savepdf(f"karate_{method}")
    plt.show()


def run_all_search_methods():
    Gk_unweighted = KarateGraphSP(start=14, goal=16, weighted=False)
    (actions,path), cost, visited, num_expanded = breadthFirstSearch(Gk_unweighted) 
    plot_solution(Gk_unweighted, path, visited, cost, num_expanded, method='BFS') 

    (actions,path), cost, visited, num_expanded = depthFirstSearch(Gk_unweighted) 
    plot_solution(Gk_unweighted, path, visited, cost, num_expanded, method='DFS') 

    Gk = KarateGraphSP(start=14, goal=16, weighted=True)
    (actions, path), cost, visited, num_expanded = uniformCostSearch(Gk) 
    plot_solution(Gk, path, visited, cost, num_expanded, method='UniformCostSearch') 


if __name__ == "__main__":
    # Plot the network
    Gk = KarateGraphSP(start=14, goal=16)
    draw_network(Gk)
    savepdf("karate")
    plt.show()
    run_all_search_methods()
