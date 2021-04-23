"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
import scipy.linalg as linalg

def vec(M):
    return M.flatten('F')

def solve_linear_problem_simple(Y, X, U, lamb=0): 
    """ Implements the solver for the basic linear regression problem.
    The method solves a problem of the form:

    > x_{k+1} = A x_k + B u_k + c

    using the more familiar naming convention
    > Y = A X + B U + c.

    Assuming there are N observations, and x_k is n-dimensional, the conventions are that
    A is n x n dimensional, Y is n x N dimensional, X is n x N dimensional, and so on (i.e. observations are in the horizontal dimension). 
    """
    n,d = X.shape[0], U.shape[0]
    P_list = [np.eye(n), np.eye(n), np.eye(n)]
    Z_list = [X, U, np.ones( (1,X.shape[1]))]
    W = solve_linear_problem(Y, Z_list=Z_list, P_list=P_list, lamb=lamb)
    A, B, C = W[0], W[1], vec(W[2])
    return A, B, C

def solve_linear_problem(Y, Z_list, P_list=None, lamb=0, weights=None):
    if P_list is None:
        P_list = [np.eye(Z_list[0].shape[0])]

    if weights is None:
        weights = np.ones( (Y.shape[1],) )

    Sigma = linalg.kron( np.diag(weights), np.eye( Y.shape[0] ) )
    S = np.concatenate( [linalg.kron(Z.T, P) for Z,P in zip(Z_list, P_list) ], axis=1)
    Wvec = np.linalg.solve( (S.T @ Sigma @ S + lamb * np.eye(S.shape[1])), S.T @ Sigma.T @ vec(Y))
    # unstack
    W = []
    d0 = 0
    for Z,P in zip(Z_list, P_list):
        dims = (P.shape[1], Z.shape[0])
        Wj = np.reshape( Wvec[d0:d0+np.prod(dims)], newshape=dims, order='F')
        d0 += np.prod(dims)
        W.append(Wj)
    return W
