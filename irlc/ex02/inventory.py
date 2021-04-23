"""
Implements the inventory-control problem from (Her21, Subsection 2.1.2). See todays slides if you are stuck!

References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.

"""
from irlc.ex01.dp_model import DPModel
from irlc.ex02.dp import DP_stochastic

class InventoryDPModel(DPModel): 
    def __init__(self, N=3):
        super().__init__(N=N)

    def A(self, x, k): # Action space A_k(x) 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def S(self, k): # State space S_k 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def g(self, x, u, w, k): # Cost function g_k(x,u,w) 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def f(self, x, u, w, k): # Dynamics f_k(x,u,w) 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def Pw(self, x, u, k): # Distribution over random disturbances 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def gN(self, x): 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

def main():
    inv = InventoryDPModel() 
    J,pi = DP_stochastic(inv)
    print(f"Inventory control optimal policy/value functions")
    for k in range(inv.N):
        print(", ".join([f" J_{k}(x_{k}={i}) = {J[k][i]:.2f}" for i in inv.S(k)] ) )
    for k in range(inv.N):
        print(", ".join([f"pi_{k}(x_{k}={i}) = {pi[k][i]}" for i in inv.S(k)] ) ) 

if __name__ == "__main__":
    main()
