"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.

"""
from collections import OrderedDict
from collections import deque
from typing import List
from irlc.ex03.queues import PriorityQueue

class SearchNode:
    """
    Implement the search-node data structure. It is a simple wrapper around each state in the search problem. In other words,
    for a state s the search node n = Node(s) will contain:

    n.state = s
    n.previous = The (previous) node in the path from the start state to n.state
    n.action = The action we had to take in the previous node to go from n.previous.state -> n.state
    n.cost = The cost of this path.

    See also (Her21, Subsection 5.1.2)
    """
    def __init__(self, state, previous=None, action=None, cost=0):
        self.state = state
        self.previous = previous
        self.action = action # the action that bring us to this node from the previous node
        self.cost = cost

    def plan(self):
        """ Returns the plan to get from start node to this state. I.e.
        >>> actions, plan = node.plan()
        are such that if we start in k=0, and state x[k] = x0, we take action
        actions[k] we will traverse to state[k+1]
        and so on for k=0,1,2,... until we get to node.state.
        """
        n = self # Current node
        actions = []
        path = []
        while n.action is not None:
            actions = [n.action] + actions
            path = [n.state] + path
            n = n.previous
        path = [n.state] + path
        return actions, path


class Frontier:
    """
    A generic frontier queue, which contains the pop and insert methods used in (Her21, Algorithm 8)
    """
    def popNext(self) -> SearchNode:
        """ Pop (return) a search node from the queue """
        pass

    def push(self, node: SearchNode):
        """ Add the search node to the queue """
        pass

    def isEmpty(self) -> bool:
        """ True if the frontier queue is empty. """
        pass

class FifoFrontier(Frontier):
    """
    Implements a First-in First-out (FIFO) frontier queu. Use the build in deque() module for fast computations.
    see (Her21, Algorithm 6)
    """
    def __init__(self):
        self._queue = deque()

    def popNext(self) -> SearchNode:
        """
        Read the difference between popleft() and pop()
        """
        return self._queue.popleft()

    def push(self, node: SearchNode):
        return self._queue.append(node)

    def isEmpty(self) -> bool:
        return len(self._queue) == 0

class LifoFrontier(FifoFrontier):
    """
    Implements a First-in last-out (LIFO) frontier queue (see (Her21, Algorithm 11)). We can re-use most of the functionality from the fifo queue.
    """
    def popNext(self) -> SearchNode: 
        """ Implement the correct pop-method. Since it is a last-in first-out frontier, we need to 
        pop the node which was added the most recently. Note we inherit from the FifoFrontier
        which defines a self._queue field, which is an instance of pythons efficient deque data structure.
        In the FIFO frontier (see above) we did:
        
        >>> return self._queue.popLeft() 
        
        which pop the left-most element. You need to pop the right-most. Either guess, or look at the documentation.
        :return: right-most element pop'ed from the frontier queue.
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

class PriorityFrontier(Frontier):
    """
    Implements a uniform cost frontier. Since we already have defined a priority queue, it is simply a matter of using the priority
    queue correctly. See (Her21, Algorithm 10) for further details.
    """
    def __init__(self):
        """ Instantiates a priority queue with appropriate pop/insert methods. It implements (Her21, Algorithm 9) """
        self.priority_queue = PriorityQueue()

    def popNext(self) -> SearchNode: 
        """ use the pop() method for the self.priority_queue """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def push(self, node: SearchNode): 
        """
        use the self.priority_queue.push method. Remember to supply it with the cost of the node.
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def isEmpty(self):
        return self.priority_queue.isEmpty()

class AStarFrontier(PriorityFrontier):
    """
    Implements the A* frontier queue, see (Her21, Algorithm 12). Since we already have a priority frontier, it is simply about
    updating the method for inserting nodes.
    """
    def __init__(self, heuristic, problem):
        self.heuristic = heuristic  # h(x) = self.heuristic(x, self.problem)
        self.problem = problem
        super().__init__()

    def push(self, node):
        """ Insert node into priority queue with the right priority.
        You should use the node and self.heuristic
        """
        # Compute the priority
        # TODO: 1 lines missing.
        raise NotImplementedError("")
        self.priority_queue.push(node, priority)

def graphSearch(problem, frontier: Frontier):
    """
    Generic graph search algorithm. Implements: (Her21, Algorithm 8).
    The frontier determines which specific search
    is implemented (dfs, bsf, ucs or astar).
    Returns the goal node, which is a structure that can be used to obtain every
    useful piece of information about the problem solution (cost, path, actions).
    """
    visited = OrderedDict() # A dictionary which maintains order.
    num_expanded = 0
    frontier.push(SearchNode(problem.initial_state)) # Push initial state into frontier
    while not frontier.isEmpty():
        n = frontier.popNext()
        if n.state not in visited:
            visited[n.state] = True
            if problem.is_terminal(n.state):
                return n.plan(), n.cost, visited, num_expanded
            num_expanded += 1
            for action, (successor, stepCost) in problem.available_transitions(n.state).items(): 
                """ Perform check of whether the successor is in the visited nodes. If not, 
                create a new search node and push it into the frontier """
                # TODO: 3 lines missing.
                raise NotImplementedError("Implement function body")
    raise Exception("No path found to goal")

def depthFirstSearch(problem): 
    """
    Search the deepest nodes in the search tree first.
    Admitable, this is a pretty silly question as you only need to specify a different frontier.
    Look at the implementation of BFS, and note we have implemented a class called LifoFrontier, which to no huge surprice implements
    a Lifo Queue. Yes, you are being asked to change the F to an L.
    """
    # TODO: 1 lines missing.
    raise NotImplementedError("Implement function body")

def breadthFirstSearch(problem): 
    """Search the shallowest nodes in the search tree first."""
    return graphSearch(problem, FifoFrontier()) 

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    return graphSearch(problem, PriorityFrontier())

def aStarSearch(problem, heuristic = None):
    """Search the node that has the lowest combined cost and heuristic first."""
    if heuristic is None:
        heuristic = lambda state, problem: 0
    return graphSearch(problem, AStarFrontier(heuristic, problem))
