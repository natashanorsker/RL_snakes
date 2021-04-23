"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
from scipy import linalg
import scipy.sparse as sparse
from osqp import OSQP
import matplotlib.pyplot as plt
from irlc.car.utils import err
# from irlc.ex04.cost_agent import CostAgent
from irlc.ex07.lmpc_sample_safe_set import SampleSafeSet
from irlc.utils.timer import Timer
from irlc.ex07.fnc.Classes import ClosedLoopData, LMPCprediction
from irlc import Agent

class LMPCAgent(Agent):
    """Create the LMPC
    Attributes:
        addTrajectory: given a ClosedLoopData object adds the trajectory to SS, Qfun, uSS and updates the iteration index
        addPoint: this function allows to add the closed loop data at iteration j to the SS of iteration (j-1)
        update: this function can be used to set SS, Qfun, uSS and the iteration index.
    """
    def __init__(self, numSS_Points, numSS_it, N, shift, dt, Laps, TimeLMPC, env=None, cost_slack=None, cost_u_deriv=None,
                 save_spy_matrices=True, epochs=10,
                 debug=True):
        super().__init__(env)
        self.dt = env.dt
        self.N = N
        self.n = env.state_size
        self.d = env.action_size
        self.OldInput = np.zeros((1, self.d))
        self.cost_slack = cost_slack
        self.cost_u_deriv = cost_u_deriv
        self.SSset = SampleSafeSet(numSS_Points=numSS_Points, numSS_it=numSS_it, shift=shift, TimeLMPC=TimeLMPC, Laps=Laps, N=N, track_map=env.map, env=env)
        self.debug = debug
        if debug:
            '''
            Construct a standard MPC controller instance to test agreement between various values, etc. and generally to debug this controller.            
            '''
            import irlc.ex07.fnc.LMPC as LMPC2
            Q = env.cost.Q
            R = env.cost.R
            dR = cost_u_deriv.R  # [0,:]
            Qslack = cost_slack.Q
            self.lmpc_legacy = LMPC2.ControllerLMPC(numSS_Points, numSS_it, N, Qslack, Q, R, dR, self.n, self.d, shift, dt, env.map, Laps, TimeLMPC, Solver="OSQP")

        self.it = 0 # Number of laps completed
        self.save_spy_matrices = save_spy_matrices
        self.has_printed_spy = False

        # these are helper classes used to gather experience. Based on old class design so should be refactered
        self.LMPCOpenLoopData = LMPCprediction(self.N, self.n, self.d, TimeLMPC, self.SSset.numSS_Points, epochs)
        self.step_within_lap = 0 # Current step within this lap.

        self.timer = Timer()
        self.timer.start()

    def pi(self, x, t=None):
        x = x.copy()
        if self.step_within_lap == 0:
            self.env.seed = np.random.randint(0, 1000)
        i = self.step_within_lap
        self.env.seed = self.env.seed + i * 7 + np.random.randint(10) if self.debug else None

        (A, b), (G, h), (P, q) = self.getQP(x)

        self.timer.tic('solve')
        z, feasible = solve_quadratic_program(sparse.csr_matrix(P), q, sparse.csr_matrix(G), h, sparse.csr_matrix(A), b)
        if feasible == 0:
            raise Exception("Unfeasible at time ", i * self.env.dt, "Cur State: ", x, "Iteration ", self.it)
        self.timer.toc()

        # Extract solution and set linearization points
        xPred = np.squeeze(np.transpose(np.reshape((z[np.arange(self.n * (self.N + 1))]), (self.N + 1, self.n))))
        uPred = np.squeeze(np.transpose(np.reshape((z[self.n * (self.N + 1) + np.arange(self.d * self.N)]), (self.N, self.d))))
        lambd = z[self.n * (self.N + 1) + self.d * self.N:z.shape[0] - self.n]
        slack = z[z.shape[0] - self.n:]

        self.xPred = xPred.T
        if self.N == 1: # planning horizon length is 1.
            self.uPred = np.array([[uPred[0], uPred[1]]])
            self.LinInput = np.array([[uPred[0], uPred[1]]])
        else:
            self.uPred = uPred.T
            self.LinInput = np.vstack((uPred.T[1:, :], uPred.T[-1, :]))

        self.OldInput = uPred.T[0, :]
        self.LinPoints = np.vstack((xPred.T[1:, :], xPred.T[-1, :]))

        if self.debug:
            self.lmpc_legacy.LinPoints = self.LinPoints
            self.lmpc_legacy.LinInput = self.LinInput
            self.lmpc_legacy.OldInput = self.OldInput
        u = self.uPred[0, :]

        # save for plotting.
        self.LMPCOpenLoopData.PredictedStates[:, :, i, self.it] = self.xPred
        self.LMPCOpenLoopData.PredictedInputs[:, :, i, self.it] = self.uPred
        self.LMPCOpenLoopData.SSused[:, :, i, self.it] = self.SSset.SS_PointSelectedTot
        self.LMPCOpenLoopData.Qfunused[:, i, self.it] = self.SSset.Qfun_SelectedTot
        return u

    def train(self, x, u, cost, xp, done=False):
        lap_done = x[4] > xp[4]
        i = self.step_within_lap
        self.timer.tic('SS')
        if self.debug:
            self.lmpc_legacy.addPoint(x,u,i=self.step_within_lap)
        self.SSset.addPoint(x,u,i=self.step_within_lap) # Needed, but why?

        # self.SSset.new_add_point(x=self.xs_buffer[i], u=self.us_buffer[i], step_within_lap=self.step_within_lap, agent=self)
        self.timer.toc()
        self.step_within_lap += 1
        if lap_done:
            # We completed a lap; add last point of trajectory (with no corresponding action)
            xp_ = xp.copy()
            xp_[4] += self.env.map.TrackLength

            self.timer.tic("SS")
            self.SSset.addPoint(x=xp_, u=None, i=self.step_within_lap)
            x, u = self.SSset.get_latest_xu()
            self.add_trajectory( [x[i] for i in range(len(x))],  [u[i] for i in range(len(u))], loading_from_disk=False)
            self.timer.toc()
            self.step_within_lap = 0

        if lap_done:
            print("Lap", self.env.completed_laps, " completed", (i + 1) * self.env.dt, "seconds: " + self.timer.display())

    def get_closed_loop_lmpc(self):
        # assert(False)
        xx, uu = self.SSset.get_latest_xu()
        x0 = self.env.state
        ClosedLoopLMPC = ClosedLoopData(self.env.dt, self.SSset.TimeLMPC, v0=x0[0])
        ClosedLoopLMPC.SimTime = len(uu)
        ClosedLoopLMPC.x[:len(xx)] = xx
        ClosedLoopLMPC.u[:len(uu)] = uu
        return ClosedLoopLMPC

    def bound2Fh(self, bound):
        '''
        Transform a Bound object into matrices of the form Fx \leq h
        '''
        dF, dh = [], []
        n = len(bound.lb)
        for j, (lb, ub) in enumerate(zip(bound.lb, bound.ub)):
            dx = np.zeros((n,))
            dx[j] = 1
            if np.isfinite(ub):
                dF.append(dx), dh.append(ub)
            if np.isfinite(lb):
                dF.append(-dx), dh.append(-lb)

        F = np.stack(dF)
        h = np.stack(dh)
        return F, h

    def getQP(self, x0):
        """
        This is the main entry point to generate the QuadraticProgram which will later be solved.
        It will return 3 sets of matrices/solutions setting up the cost function, equality and inequality constraints
        """
        """ 
        Obtain sample-safe sets, i.e. the vectors v_i and their values Q(v_i) from the sample-safe set.             
        """
        self.timer.tic('SS.select_points')
        if not self.debug:
            # for some reason the LMPC implementation finds the sample set from x0. This is very, very strange.
            x0 = self.LinPoints[-1, :]
        SS_v, SS_Qv = self.SSset.select_points(x0) 
        self.timer.toc()
        self.timer.tic("SS.ABC")
        """
        SS_v contains the coordinates of the safe set points, while SS_Qv contains the assocated cost-to-go.
        The dimensions of SS_v are nL by n (n=6) and for SS_Qv they are nL by 1.
        """
        nL = len(SS_Qv) 
        N,n,d = self.N, self.env.state_size, self.env.action_size # N=12 (horizon), n=6,d=2 (state/action size)
        """ Approximation of system dynamics of the form:
        
        x_{k+1} = Atv[k] x_k + Btv[k] u_k + Ctv[k]
        
        That is, the constraint which should be implemented is:
        
        x_{k+1} - Atv[k] x_k - Btv[k] u_k = Ctv[k]
        
        where the left-hand side is absorbed into A and the right-hand side in b.         
        """
        Atv, Btv, Ctv = self._EstimateABC() 
        self.timer.toc()
        self.timer.tic("SS-rest")
        '''
        Equality constraints: 
        Ax = b
        '''
        A11 = -np.roll( linalg.block_diag(*Atv, np.zeros( (n,n) )), shift=n,axis=0)+np.eye(n*(N+1)) # Dynamic Model
        A12 = -np.roll(linalg.block_diag(*Btv, np.zeros((n, 2))), shift=n, axis=0) # Inputs

        """
        Compute the A-matrix and b-vector used for formulating the optimisation problem.
        The dynamic model and effect of inputs are given above by A11 and A12
        Have a look at the spy matrix in the lecture slides
        Remember that n=6 is the state dimension, d=2 the action_size, nL is the amount of selected safe set points and N is the horizon length
        This means that A11 is an n(N+1) by n(N+1) matrix, while A12 is an n(N+1) by nd(N+1) matrix 
        """
        # TODO: 5 lines missing.
        raise NotImplementedError("Implement the matrices A and b used for the equality constraints Az=b")

        '''
        Inequality constraints:
        Gx \leq h
        '''
        bounds = self.env.simple_bounds() 
        Fx, hx = self.bound2Fh(bounds['x'])
        Fu, hu = self.bound2Fh(bounds['u']) 

        # TODO: 5 lines missing.
        raise NotImplementedError("Implement the matrices G and h used for the inequality constraints Gz<=h")

        '''
        Build cost matrices: 0.5 x' P x + q' x (you do not need to read this code)
        '''
        Rd = self.cost_u_deriv.R
        Qslack = self.cost_slack.Q
        R = self.env.cost.R
        numSS_Points = self.SSset.numSS_Points

        bb = [self.env.cost.Q] * N
        Mx = linalg.block_diag(*bb)
        c = [R + 2 * np.diag(Rd)] * N

        Mu = linalg.block_diag(*c)
        # Need to consider that the last input appears just once in the difference
        for j in range(len(Rd)):
            Mu[Mu.shape[0] - 2+j, Mu.shape[1] - 2+j] = Mu[Mu.shape[0] - 2+j, Mu.shape[1] - 2+j] - Rd[j]

        # Derivative Input Cost
        OffDiaf = -np.tile(Rd, N - 1)
        np.fill_diagonal(Mu[2:], OffDiaf)
        np.fill_diagonal(Mu[:, 2:], OffDiaf)
        M00 = linalg.block_diag(Mx, self.env.cost.Q, Mu)
        M0 = linalg.block_diag(M00, np.zeros((numSS_Points, numSS_Points)), Qslack)
        # xtrack =
        q0 = -2*np.dot(np.append(np.tile(self.env.cost.q, N + 1), np.zeros(R.shape[0] * N)), M00)
        # Derivative Input
        # assert(sum(np.abs(q0)) == 0)
        q0[n * (N + 1):n * (N + 1) + d] = -2 * self.OldInput @ np.diag(Rd)
        q = np.append(np.append(q0, SS_Qv), np.zeros(self.env.cost.Q.shape[0]))
        P = 2 * M0  # Need to multiply by two because LSQP considers 1/2 in front of quadratic cost

        if self.debug and False:
            self.lmpc_legacy._set_selected_tot(x0)
            L2, G2, E2, M2, q2, F2, b2 = self.lmpc_legacy._getQP(x0)
            eq_b2 = E2 @ x0 + L2[:,0] # equality constraint target.

            if not self.has_printed_spy:
                plt.close()
                self.has_printed_spy = True
                from irlc import savepdf
                if self.save_spy_matrices:
                    for m,g in [ (A,"A"), (G,"G"), (np.transpose([(b)]), "b"), (np.transpose([(h)]), "h")]:
                        plt.figure()
                        plt.spy(m)
                        savepdf("spy_mat_of_%s"%g)
                        plt.close()

                print("Testing A matrix...")
                err(A - G2, message = "A not implemented correctly")
                print("Testing b vector...")
                err(b - eq_b2, message = "b not implemented correctly")

                print("Testing G matrix...")
                err(G - F2, message = "G not implemented correctly")
                print("Testing h vector...")
                err(h - b2, message = "h not implemented correctly")

                print("Testing P matrix...")
                err(P - M2, message = "P not implemented correctly")
                print("Testing q vector...")
                err(q - q2, message = "q not implemented correctly")

                
                
                    
                     
                
                                 
                
                
                
                    
                
        self.timer.toc()
        return (A,b), (G,h), (P,q)

    def _EstimateABC(self):
        Atv, Btv, Ctv = [], [], []
        for i in range(0, self.N):
            Ai, Bi, Ci = self.RegressionAndLinearization(x0=self.LinPoints[i, :], u0=self.LinInput[i, :])
            Atv.append(Ai)
            Btv.append(Bi)
            Ctv.append(Ci)

        return Atv, Btv, Ctv

    def RegressionAndLinearization(self, x0, u0):
        '''
        Regression and linearization around x0, u0. We obtain a system of the form

        x_{i+1} = A x_i + B u_i + C

        Code contains some magic values (size of neighbourhood, etc.)
        '''
        MaxNumPoint = 40
        h = 5
        lamb = 0.0
        stateFeatures = [0, 1, 2]
        scaling = np.diag([0.1, 1, 1, 1, 1])

        # see below for format: (y(output), input-feature)
        features_inputs_reg = [(0, [1]), (1, [0]), (2, [0])]

        dAi, dBi, dCi, _ = self.SSset.RegressionAndLinearization(x0, u0, MaxNumPoint, h, lamb, stateFeatures, scaling, features_inputs_reg)
        Iout = np.asarray( [i for (i,_) in features_inputs_reg] )
        fx ,Ai, _ = self.env.discrete_model.f(x0, u0, i=0, compute_jacobian=True) # Compute f(x0, u0) and Jacobian of df(x0,u0)/dx.

        Ai[Iout,:] = dAi[Iout,:]  # Use regression model for velocities and sympy jacobian for position and orientation.
        Bi = dBi

        Ci = fx - Ai @ x0
        Ci = np.reshape(Ci,newshape=(self.n,1))
        Ci[Iout,:] = dCi[Iout]
        return Ai, Bi, Ci

    def add_trajectory(self, xs, us, loading_from_disk=True):
        '''
        Add trajectory information to buffer; used when loading date or for stats maintainance.
        '''
        if self.debug:
            from irlc.ex07.fnc.Classes import ClosedLoopData
            xs_glob = [self.env.x_curv2x_XY(x_) for x_ in xs]
            # build this closed loop trajectory thingy for backwards compatibility
            ClosedLoopLMPC = ClosedLoopData(self.env.dt, len(us), v0=xs[0][0])
            ClosedLoopLMPC.SimTime = len(us)
            ClosedLoopLMPC.x[:len(xs)] = np.stack(xs)
            ClosedLoopLMPC.u[:len(us)] = np.stack(us)
            ClosedLoopLMPC.x_glob[:len(xs_glob)] = np.stack(xs_glob)
            self.lmpc_legacy.addTrajectory(ClosedLoopLMPC)
        if loading_from_disk:
            self.SSset.add_trajectory(xs, us)
        if self.it == 0:
            self.LinPoints = self.SSset.SS[1:self.N + 2, :, self.it]
            self.LinInput = self.SSset.uSS[1:self.N + 1, :, self.it]
        self.it += 1

def solve_quadratic_program(P, q, G=None, h=None, A=None, b=None, initvals=None):
    """
    Solve a Quadratic Program defined as:
        minimize
            (1/2) * z.T * P * z + q.T * z
        subject to
            G * z <= h
            A * z == b
    using OSQP <https://github.com/oxfordcontrol/osqp>.
    Parameters
    ----------
    P : scipy.sparse.csc_matrix Symmetric quadratic-cost matrix.
    q : numpy.array Quadratic cost vector.
    G : scipy.sparse.csc_matrix Linear inequality constraint matrix.
    h : numpy.array Linear inequality constraint vector.
    A : scipy.sparse.csc_matrix, optional Linear equality constraint matrix.
    b : numpy.array, optional Linear equality constraint vector.
    initvals : numpy.array, optional Warm-start guess vector.
    Returns
    -------
    z : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    Note
    ----
    OSQP requires `P` to be symmetric, and won't check for errors otherwise.
    Check out for this point if you e.g. `get nan values
    <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.
    """
    osqp = OSQP()
    if G is not None:
        l = -np.inf * np.ones(len(h))
        if A is not None:
            qp_A = sparse.vstack([G, A]).tocsc()
            qp_l = np.hstack([l, b])
            qp_u = np.hstack([h, b])
        else:  # no equality constraint
            qp_A = G
            qp_l = l
            qp_u = h
        osqp.setup(P=P.tocsc(), q=q, A=qp_A.tocsc(), l=qp_l, u=qp_u, verbose=False, polish=True)
    else:
        osqp.setup(P=P.tocsc(), q=q, A=None, l=None, u=None, verbose=False)
    if initvals is not None:
        osqp.warm_start(x=initvals)
    osqp.max_iter = 8000
    res = osqp.solve()
    import osqp._osqp as _osqp  # tue.
    if res.info.status_val != _osqp.constant('OSQP_SOLVED'):
        print("OSQP exited with status '%s'" % res.info.status)
    feasible = res.info.status_val in [_osqp.constant('OSQP_SOLVED'), _osqp.constant('OSQP_SOLVED_INACCURATE'), _osqp.constant('OSQP_MAX_ITER_REACHED')]
    z = res.x
    return z, feasible
# 501
