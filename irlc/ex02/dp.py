"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.

"""
from irlc.ex01.graph_traversal import SmallGraphDP
from irlc.ex01.graph_traversal import policy_rollout

def DP_stochastic(model):
    """
    Implement the stochastic DP algorithm. The implementation follows (Her21, Algorithm 1).
    In case you run into problems, I recommend following the hints in (Her21, Subsection 3.2.1) and focus on the
    case without a noise term; once it works, you can add the w-terms. When you don't loop over noise terms, just specify
    them as w = None in env.f and env.g.
    """
    N = model.N
    J = [{} for _ in range(N + 1)]
    pi = [{} for _ in range(N)]
    J[N] = {x: model.gN(x) for x in model.S(model.N)}
    for k in range(N-1, -1, -1):
        for x in model.S(k):
            """
            Update pi[k][x] and Jstar[k][x] using the general DP algorithm given in (Her21, Algorithm 1).
            If you implement it using the pseudo-code, I recommend you define Q as a dictionary like the J-function such that
                        
            > Q[u] = Q_u (for all u in model.A(x,k))
            Then you find the u where Q_u is lowest, i.e. 
            > umin = arg_min_u Q[u]
            Then you can use this to update J[k][x] = Q_umin and pi[k][x] = umin.
            """
            # TODO: 4 lines missing.
            raise NotImplementedError("")
            """
            After the above update it should be the case that:

            J[k][x] = J_k(x)
            pi[k][x] = pi_k(x)
            """
    return J, pi


if __name__ == "__main__": # Test dp on small graph old given in (Her21, Subsection 3.2.1)
    print("Testing the deterministic DP algorithm on the small graph old")
    model = SmallGraphDP(t=5) # Instantiate the small graph old with target node 5 
    J, pi = DP_stochastic(model)
    # Print all optimal cost functions J_k(x_k) 
    for k in range(len(J)):
        print(", ".join([f"J_{k}({i}) = {v:.1f}" for i, v in J[k].items()]))
    s = 2  # start node
    J,xp = policy_rollout(model, pi=lambda x, k: pi[k][x], x0=s)
    print(f"Actual cost of rollout was {J} which should obviously be similar to J_0[{s}]")
    print(f"Path was", xp) 
    # Remember to check optimal path agrees with the the (self-evident) answer from the figure.
