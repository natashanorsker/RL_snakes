"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.

"""
import numpy as np
import matplotlib.pyplot as plt
from irlc import bmatrix
from irlc import savepdf


def fz(X, a, b=None, N=None):
    """
    Helper function. Check if X is None, and if so return a list
    X = [A,A,....]
    which is N long and where each A is a (a x b) zero-matrix.
    """
    if X is not None:
        return X
    X = np.zeros((a,) if b is None else (a,b))
    return [X] * N

def LQR(A, B, d=None, Q=None, R=None, H=None, q=None, r=None, qc=None, QN=None, qN=None, qcN=None, mu=0):
    """
    Implement LQR as defined in (Her21, Algorithm 22). All terms retain their meaning/order of computation from the code; please be careful with the linear algebra!

    Input:
        We follow the convention A, B, etc. are lists of matrices/vectors/scalars, such that
        A_k = A[k] is the dynamical matrix, etc.

        A slight annoyance is we have both a Q-matrix, q-vector and q-constant.
        we will therefore use q to refer to the vector and qc to refer to the scalar.
    Return:
        We will return the (list of) control matrices/vectors L, l such that u = l + L x
    """
    N = len(A)
    n,m = B[0].shape
    # Initialize control matrices and cost terms
    L, l = [None]*N, [None]*N
    V, v, vc = [None]*(N+1), [None]*(N+1), [None]*(N+1)
    # Initialize constant cost-function terms to zero if not specified
    QN = np.zeros((n,n)) if QN is None else QN
    qN = np.zeros((n,)) if qN is None else qN
    qcN = 0 if qcN is None else qcN
    H,q,qc,r = fz(H,m,n,N=N), fz(q,n,N=N), fz(qc,1,N=N), fz(r,m,N=N)
    d = fz(d,n, N=N)

    # TODO: 1 lines missing.
    raise NotImplementedError("Initialize V[N], v[N], vc[N] here")

    In = np.eye(n)
    for k in range(N-1,-1,-1):
        # When you update S_uu and S_ux remember to add regularization as the terms ... (V[k+1] + mu * In) ...
        # Note that that to find x such that
        # >>> x = A^{-1} y this
        # in a numerically stable manner this should be done as
        # >>> x = np.linalg.solve(A, y)
        # TODO: 6 lines missing.
        raise NotImplementedError("")
        V[k] = 0.5 * (V[k] + V[k].T)  # I recommend putting this here to keep V positive semidefinite
        # TODO: 2 lines missing.
        raise NotImplementedError("")


    return (L,l), (V,v,vc)


def dlqr_J(x,V,v,vc, QN=None, qN=None, qcN=None):
    """
    Compute cost terms. Currently not used
    """
    Jk, xN = [1/2 * V_.T @ x_ @ V_ + v_.T @ x_ + vc_ for (x_, V_,v_, vc_) in zip(x[:-1],V,v,vc)] + x[-1]
    JN = (1/2*xN.T@QN@xN if QN is not None else 0) + (qN.T@xN if qN is not None else 0)
    return sum(Jk)+ JN + (qcN if QN is not None else 0)

def lqr_rollout(x0,A,B,d,L,l):
    """
    Compute a rollout (states and actions) given solution from LQR controller function.
    """
    x, states,actions = x0, [x0], []
    n,m = B[0].shape
    N = len(L)
    d = fz(d,n,1,N)
    l = fz(l,m,1,N)
    for k in range(N):
        u = L[k] @ x + l[k]
        x = A[k] @ x + B[k] @ u + d[k]
        actions.append(u)
        states.append(x)
    return states,actions

if __name__ ==  "__main__":
    """
    Solve this problem (see also lecture notes for the same example)
    http://cse.lab.imtlucca.it/~bemporad/teaching/ac/pdf/AC2-04-LQR-Kalman.pdf
    """
    N = 20
    A = np.ones((2,2))
    A[1,0] = 0
    B = np.asarray([[0], [1]])
    Q = np.zeros((2,2))
    R = np.ones((1,1))

    print("System matrices A, B, Q, R")
    print(bmatrix(A))  
    print(bmatrix(B))  
    print(bmatrix(Q))  
    print(bmatrix(R))  

    for rho in [0.1, 10, 100]:
        Q[0,0] = 1/rho
        (L,l), (V,v,vc) = LQR(A=[A]*N, B=[B]*N, d=None, Q=[Q]*N, R=[R]*N, QN=Q)

        x0 = np.asarray( [[1],[0]])
        trajectory, actions = lqr_rollout(x0,A=[A]*N, B=[B]*N, d=None,L=L,l=l)

        xs = np.concatenate(trajectory, axis=1)[0,:]

        plt.plot(xs, 'o-', label=f'rho={rho}')

        k = 10
        print(f"Control matrix in u_k = L_k x_k + l_k at k={k}:", L[k])
    for k in [N-1,N-2,0]:
        print(f"L[{k}] is:", L[k].round(4))
    plt.title("Double integrator")
    plt.xlabel('Steps $k$')
    plt.ylabel('$x_1 = $ x[0]')
    plt.legend()
    plt.grid()
    savepdf("dlqr_double_integrator")
    plt.show()
