"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.

"""
import numpy as np
from irlc.ex01.dp_model import DPModel

"""
Graph of shortest path problem of (Her21, Subsection 2.1.1)
"""
G222 = {(1, 2): 6,  (1, 3): 5, (1, 4): 2, (1, 5): 2,  
        (2, 3): .5, (2, 4): 5, (2, 5): 7,
        (3, 4): 1,  (3, 5): 5, (4, 5): 3}  

def symG(G):
    """ make a graph symmetric. I.e. if it contains edge (a,b) with cost z add edge (b,a) with cost c """
    G.update({(b, a): l for (a, b), l in G.items()})
symG(G222)

class SmallGraphDP(DPModel):
    """ Implement the small-graph example in (Her21, Subsection 2.1.1). t is the terminal node. """
    def __init__(self, t, G=None):  
        self.G = G if G is not None else G222
        self.G[(t,t)] = 0  # make target position absorbing
        self.t = t
        self.nodes = {i for k in self.G for i in k}
        super(SmallGraphDP, self).__init__(N=len(self.nodes)-1)  

    def f(self, x, u, w, k):
        if (x,u) in self.G:  
            # TODO: 1 lines missing.
            raise NotImplementedError("Implement function body")
        else:
            raise Exception("Nodes are not connected")

    def g(self, x, u, w, k): 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def gN(self, x):  
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def S(self, k):   
        return self.nodes

    def A(self, x, k):
        return {j for (i,j) in self.G if i == x} 

def pi_silly(x, k): 
    if x == 1:
        return 2
    else:
        return 1 

def pi_inc(x, k): 
    # TODO: 1 lines missing.
    raise NotImplementedError("Implement function body")

def pi_smart(x,k): 
    # TODO: 1 lines missing.
    raise NotImplementedError("Should go from 1 to 5 along a low-cost route")

def policy_rollout(model, pi, x0):
    """
    Given an environment and policy, should compute one rollout of the policy and compute
    cost of the obtained states and actions. In the deterministic case this corresponds to

    J_pi(x_0)

    in the stochastic case this would be an estimate of the above quantity.
    """
    J, x, trajectory = 0, x0, [x0]
    for k in range(model.N):
        # TODO: 1 lines missing.
        raise NotImplementedError("Generate the action u = ... here using the policy")
        w = model.w_rnd(x, u, k) # This is required; just pass them to the transition function
        # TODO: 2 lines missing.
        raise NotImplementedError("Update J and generate the next value of x.")
        trajectory.append(x) # update the trajectory
    # TODO: 1 lines missing.
    raise NotImplementedError("Add last cost term env.gN(x) to J.")
    return J, trajectory

def main():
    t = 5  # target node
    model = SmallGraphDP(t=t)
    x0 = 1  # starting node
    print("Cost of pi_silly", policy_rollout(model, pi_silly, x0)[0]) 
    print("Cost of pi_inc", policy_rollout(model, pi_inc, x0)[0])
    print("Cost of pi_smart", policy_rollout(model, pi_smart, x0)[0])  

if __name__ == '__main__':
    main()
