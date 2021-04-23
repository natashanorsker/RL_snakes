"""
This implements two methods: The basic ILQR method, described in (Her21, Algorithm 24), and the linesearch-based method
described in (Her21, Algorithm 25). You are also invitet to consult (TET12) (which contains generalization to DDP)
and (Har20, Alg 1).

References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.

  [TET12] Yuval Tassa, Tom Erez, and Emanuel Todorov. Synthesis and stabilization of complex behaviors through online trajectory optimization. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 4906–4913. IEEE, 2012. (See tassa2012.pdf). 
  [Har20] James Harrison. Optimal and learning-based control combined course notes. (See AA203combined.pdf), 2020. 
"""
import warnings
import numpy as np
from irlc.ex06.dlqr import LQR

def ilqr_basic(env, N, x0, us_init=None, n_iterations=500,verbose=True):
    '''
    Basic ilqr. I.e. (Her21, Algorithm 24). Our notation (x_bar, etc.) will be consistent with the lecture slides
    '''
    mu, alpha = 1, 1 # We will get back to these. For now, just let them have defaults and don't change them
    n, m = env.state_size, env.action_size
    u_bar = [np.random.uniform(-1, 1,(env.action_size,)) for _ in range(N)] if us_init is None else us_init
    x_bar = [x0] + [np.zeros(n, )] * (N)
    """
    Initialize nominal trajectory xs, us using us and x0 (i.e. simulate system from x0 using action sequence us). 
    The simplest way to do this is to call forward_pass with all-zero sequence of control vector/matrix l, L.
    """
    # TODO: 2 lines missing.
    raise NotImplementedError("Initialize x_bar, u_bar here")
    J_hist = []
    for i in range(n_iterations):
        """
        Compute derivatives around trajectory and cost estimate J of trajectory. To do so, use the get_derivatives
        function        
        """
        # TODO: 2 lines missing.
        raise NotImplementedError("Compute J and derivatives f_x, f_u, ....")
        """  Backward pass: Obtain feedback law matrices l, L using the backward_pass function.
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Compute L, l = .... here")
        """ Forward pass: Given L, l matrices computed above, simulate new (optimal) action sequence. 
        In the lecture slides, this is similar to how we compute u^*_k and x_k
        Once they are computed, iterate the iLQR algorithm by setting x_bar, u_bar equal to these values
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Compute x_bar, u_bar = ...")
        if verbose:
            print(f"{i}> J={J:4g}, change in cost since last iteration {0 if i == 0 else J-J_hist[-1]:4g}")
        J_hist.append(J)
    return x_bar, u_bar, J_hist, L, l

def ilqr_linesearch(env, N, x0, n_iterations, us_init=None, tol=1e-6,verbose=True):
    """
    For linesearch implement method described in (Her21, Algorithm 25) (we will use regular iLQR, not DDP!)
    """
    # The range of alpha-values to try out in the linesearch
    # plus parameters relevant for regularization scheduling.
    alphas = 1.1 ** (-np.arange(10) ** 2)  # alphas = [1, 1.1^{-2}, ...]
    mu_min = 1e-6
    mu_max = 1e10
    Delta_0 = 2
    mu = 1.0
    Delta = Delta_0

    n, m = env.state_size, env.action_size
    u_bar = [np.random.uniform(-1, 1, (env.action_size,)) for _ in range(N)] if us_init is None else us_init
    x_bar = [x0] + [np.zeros(n, )] * (N)
    # Initialize nominal trajectory xs, us (same as in basic linesearch)
    # TODO: 2 lines missing.
    raise NotImplementedError("Copy-paste code from previous solution")
    J_hist = []

    converged = False
    for i in range(n_iterations):
        alpha_was_accepted = False
        """ Step 1: Compute derivatives around trajectory and cost estimate of trajectory.
        (copy-paste from basic implementation). In our implementation, J_bar = J_{u^star}(x_0) """
        # TODO: 2 lines missing.
        raise NotImplementedError("Obtain derivatives f_x, f_u, ... as well as cost of trajectory J_bar = ...")
        try:
            """
            Step 2: Backward pass to obtain control law (l, L). Same as before so more copy-paste
            """
            # TODO: 1 lines missing.
            raise NotImplementedError("Obtain l, L = ... in backward pass")
            """
            Step 3: Forward pass and alpha scheduling.
            Decrease alpha and check condition |J^new < J'|. Apply the regularization scheduling as needed. """
            for alpha in alphas:
                x_hat, u_hat = forward_pass(env, x_bar, u_bar, L=L, l=l, alpha=alpha) # Simulate trajectory using this alpha
                # TODO: 1 lines missing.
                raise NotImplementedError("Compute J_new = ... as the cost of trajectory x_hat, u_hat")

                if J_new < J_bar:
                    """ Linesearch proposed trajectory accepted! Set current trajectory equal to x_hat, u_hat. """
                    if np.abs((J_bar - J_new) / J_bar) < tol:
                        converged = True  # Method does not seem to decrease J; converged. Break and return.

                    J_bar = J_new
                    x_bar, u_bar = x_hat, u_hat
                    '''
                    The update was accepted and you should change the regularization term mu, 
                     and the related scheduling term Delta.                   
                    '''
                    # TODO: 1 lines missing.
                    raise NotImplementedError("Delta, mu = ...")
                    alpha_was_accepted = True # accept this alpha
                    break
        except np.linalg.LinAlgError as e:
            # Matrix in dlqr was not positive-definite and this diverged
            warnings.warn(str(e))

        if not alpha_was_accepted:
            ''' No alphas were accepted, which is not too hot. Regularization should change
            '''
            # TODO: 1 lines missing.
            raise NotImplementedError("Delta, mu = ...")

            if mu_max and mu >= mu_max:
                raise Exception("Exceeded max regularization term; we are stuffed.")

        dJ = 0 if i == 0 else J_bar-J_hist[-1]
        info = "converged" if converged else ("accepted" if alpha_was_accepted else "failed")
        if verbose:
            print(f"{i}> J={J_bar:4g}, decrease in cost {dJ:4g} ({info}).\nx[N]={x_bar[-1].round(2)}")
        J_hist.append(J_bar)
        if converged:
            break
    return x_bar, u_bar, J_hist, L, l

def backward_pass(f_x, f_u, c_x, c_u, c_xx, c_ux, c_uu, _mu=1):
    """
    Get L,l feedback law given linearization around nominal trajectory
    To do so, simply call LQR with appropriate inputs (i.e. the derivative terms).
    """
    # TODO: 6 lines missing.
    raise NotImplementedError("")
    (L, l), (V, v, vc) = LQR(A=A, B=B, R=R, Q=Q, QN=QN, H=H, q=q, qN=qN, r=r, mu=_mu)
    return L,l

def compute_J(env, xs, us):
    """
    Helper function which computes the cost of the trajectory. 
    
    Input: 
        xs: States (N+1) x [(state_size)]
        us: Actions N x [(state_size)]
        
    Returns:
        Trajectory's total cost.
    """
    N = len(us)
    JN = env.cN(xs[-1])
    return sum(map(lambda args: env.c(*args), zip(xs[:-1], us, range(N)))) + JN

def get_derivatives(model, x_bar, u_bar):
    """
    Compute derivatives for system dynamics around the given trajectory. should be handled using
    env.f and env.g+env.gN.

    f_x, f_u has the same meaning as in the notes, i.e. A_k, B_k. Note they are lists of the derivatives of the system dynamics wrt. x and u.

    Meanwhile the terms c, c_x, ... has the meaning described in (Her21, Subequation 12.8), i.e. the derivatives of the c (cost) terms.
    These derivatives will be returned as lists of matrices/vectors, i.e. one for each k-value. Note that in particular
    c will be a N+1 list of the cost terms, such that J = sum(c) is the total cost of the trajectory.
    """
    N = len(u_bar)
    """ Compute f_x = A_k, f_u = B_k (lists of matrices of length N)
    Recall env.f has output
        x, f_x[i], f_u[i], _, _, _ = env.f(x, u, i, compute_jacobian=True)
    """
    # TODO: 2 lines missing.
    raise NotImplementedError("")
    """ Compute derivatives of the cost function. For terms not including u these should be of length N+1 
    (because of gN!), for the other lists of length N
    recall env.g has output:
        c[i], c_x[i], c_u[i], c_xx[i], c_ux[i], c_uu[i] = env.c(x, u, i, compute_gradients=True)
    """
    # TODO: 2 lines missing.
    raise NotImplementedError("")
    # Concatenate the derivatives associated with the last time point N.
    cN, c_xN, c_xxN = model.cN(x_bar[N], compute_gradients=True)
    c = c + (cN,)
    c_x = c_x + (c_xN,)
    c_xx = c_xx + (c_xxN,)
    return (f_x, f_u), (c, c_x, c_u, c_xx, c_ux, c_uu)

def forward_pass(model, x_bar, u_bar, L, l, alpha=1.0):
    """Applies the controls for a given trajectory.

    Args:
        x_bar: Nominal state path [N+1, state_size].
        u_bar: Nominal control path [N, action_size].
        l: Feedforward gains [N, action_size].
        L: Feedback gains [N, action_size, state_size].
        alpha: Line search coefficient.

    Returns:
        Tuple of
            x: state path [N+1, state_size] simulated by the system
            us: control path [N, action_size] new control path
    """
    N = len(u_bar)
    x = [None] * (N+1)
    u_star = [None] * N
    x[0] = x_bar[0].copy()

    for i in range(N):
        """ Compute using (Her21, eq. (12.16))
        u_{i} = ...
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("u_star[i] = ....")
        """ Remember to compute 
        x_{i+1} = f_k(x_i, u_i^*)        
        here:
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("x[i+1] = ...")
    return x, u_star
