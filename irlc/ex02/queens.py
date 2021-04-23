"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from timeit import timeit
import matplotlib.pyplot as plt
import numpy as np
from irlc import savepdf
from irlc.ex01.dp_model import DPModel
from irlc.ex02.search_problem import DP2SP
from irlc.ex02.dp_forward import dp_forward
import time

class QueensDP(DPModel):
    """
    The N-queen problem: Place N queens on a checkboard so no two queens attack each other

    We will consider this as a decision problem where
     * in each stage x_k consists of a configuration with k queens,
     * the actions are the coordinates (i,j) to place a new queen,
     * the reward is infinite if we end up with a bad configuration and otherwise 0.
    """
    def __init__(self, NQ=4):
        super(QueensDP, self).__init__(NQ)

    def valid_pos_(self, x, dx):
        """
        Check if adding dx to x will lead to a valid board configuration
        """
        a,b = dx
        for (i,j) in x:
            if a == i or b == j or i-j == a-b or a+b == i+j:
                return False
        return True

    def A(self, x, k): 
        # TODO: 2 lines missing.
        raise NotImplementedError("Implement function body")

    def g(self, x, u, w, k): 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def gN(self, x): 
        """
        This function is technically redundant when U is selected as
        those actions that will lead to a valid board position. I have implemented it anyway
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def f(self, x, u, w, k): 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

def chessboard_plot(q_sp, path):
    xy  = path[-2][0]
    N = len(xy)
    A = np.zeros( (N,N) )
    for (i,j) in xy:
        A[i,j] = 1

    plt.pcolor(1-A, edgecolors='k', linewidths=1, cmap='gray')
    plt.gca().set_aspect('equal', 'box')
    savepdf(f"queens{N}x{N}")
    plt.show()

def queens(N=4):
    start = time.time()
    q = QueensDP(N)
    s = ()  # first state is the empty chessboard
    q_sp = DP2SP(q, initial_state=s)
    J, actions, path = dp_forward(q_sp, N)
    print("Final configuration x_N from an optimal policy is:", path[-1]) 
    if N == 8:
        print(f"Elapsed time for solving 8x8 NQueens problem is {time.time() - start:4f} seconds (on my computer: ~0.04 seconds)")
    chessboard_plot(q_sp, path)

if __name__ == "__main__":
    queens(4)
    """
    If you are feeling ambitious you can try the 8-queen problem.    
    Note this will not work with an entirely naive implementation of the N-queens problem; you have to think about how 
    QueensDP.A can be implemented in a smarter way to reduce the number of states you have to search. 

    I have thrown in a benchmark from my ancient laptop to give an indication of whether you are on the right track.    
    """
    queens(8)
