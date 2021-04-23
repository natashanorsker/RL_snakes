"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
from numpy import linalg as la
from irlc.car.utils import err

class LapData:
    lap_length = -1
    def __init__(self):
        self.t = []
        self.x = []
        self.u = []
        self.Q = []


class SampleSafeSet:
    def __init__(self, numSS_Points, numSS_it, TimeLMPC, shift, Laps, N, track_map, env=None):
        self.track_map = track_map
        self.env = env
        # self.cost_function = cost_function
        self.it = 0 # something something controller iteration
        '''
        What are these variables?
        '''
        self.numSS_Points = numSS_Points  # points to select in the SS set for barycentric approx
        self.numSS_it     = numSS_it
        self.shift = shift
        self.n = env.state_size
        self.d = env.action_size
        # Initialize the following quantities to avoi  bd dynamic allocation
        # TODO: is there a more graceful way to do this in python?
        NumPoints = int(TimeLMPC / env.dt) + 1

        self.NumPoints = NumPoints
        self.max_laps = Laps

        self.TimeSS  = 10000 * np.ones(Laps).astype(int)        # Time at which each j-th iteration is completed
        self.SS      = 10000 * np.ones((NumPoints, self.n, Laps))    # Sampled Safe SS
        self.uSS     = 10000 * np.ones((NumPoints, self.d, Laps))    # Input associated with the points in SS
        self.Qfun    =     0 * np.ones((NumPoints, Laps))       # Qfun: cost-to-go from each point in SS
        # This variable is not really used except for plotting. Should be computed post-hoc
        # Initialize the controller iteration
        self.it      = 0
        self.N = N

        ## New-style sample safe set.
        self.lap_data = []
        self.current_lap = None
        self.TimeLMPC = TimeLMPC

    def new_to_old(self):
        NumPoints = self.NumPoints
        Laps = self.max_laps
        n = self.env.state_size
        d = self.env.action_size
        TimeSS = 10000 * np.ones(Laps).astype(int)  # Time at which each j-th iteration is completed
        SS = 10000 * np.ones((NumPoints, n, Laps))  # Sampled Safe SS
        uSS = 10000 * np.ones((NumPoints, d, Laps))  # Input associated with the points in SS
        Qfun = 0 * np.ones((NumPoints, Laps))  # Qfun: cost-to-go from each point in SS
        a = 24234
        for it, l in enumerate(self.lap_data):
            xx = np.stack(l.x)
            uu = np.stack(l.u)
            Q = np.stack(l.Q)
            SS[:len(xx),:,it] = xx
            uSS[:len(uu),:,it] = uu

            # Q = self.cost_function(xx, uu, self.track_map.TrackLength)
            Qfun[:len(Q),it] = Q
            # self.Qfun[0:(end_it + 1), it] = self.cost_function(X, U, self.track_map.TrackLength)
            for i in np.arange(0, Qfun.shape[0]):
                if Qfun[i, it] == 0:
                    Qfun[i, it] = Qfun[i - 1, it] - 1
            TimeSS[it] = l.lap_length

        err(SS-self.SS)
        err(uSS-self.uSS)
        err(Qfun - self.Qfun)
        err(TimeSS - self.TimeSS)
        return SS, uSS, Qfun, TimeSS

    def new_add_point(self, x, u=None, Q = None, xp=None, do_check=True, step_within_lap=None,agent=None):
        """ New-style adding points to the sample-safe set. """

        if step_within_lap == 0:
            # if self.current_lap is not None:
            #     self.new_add_trajectory(self.current_lap.x, self.current_lap.u)
            #     self.lap_data.append(self.current_lap)
            if self.current_lap is None:
                self.current_lap = LapData()

        self.current_lap.x.append(x)

        if u is None:
            self.new_add_trajectory(self.current_lap.x, self.current_lap.u)
            self.current_lap = None
        else:
            # if len(self.lap_data) == 3:
            #     cdx = 168
            #     print(len(self.lap_data[2].u))
            #     print("worldy")

            self.current_lap.u.append(u.copy())
            l = self.lap_data[-1]
            cdx = l.lap_length + 1 + step_within_lap

            u0 = 10000 * np.ones((self.env.action_size,))
            x0 = 10000 * np.ones((self.env.state_size,))
            # if step_within_lap == 0:
            #
            while len(l.u) < cdx+1:
                l.u.append(u0.copy())
            while len(l.x) < cdx + 1:
                l.x.append(x0)
            while len(l.Q) < cdx + 1:
                l.Q.append(0)

            l.x[cdx] = x + np.array([0, 0, 0, 0, self.track_map.TrackLength, 0])
            l.u[cdx] = u.copy()
            l.Q[cdx] = l.Q[cdx-1] - 1

            if do_check:
                self.new_to_old()


    def new_add_trajectory(self, xs, us):
        l = LapData()
        l.x = xs
        l.u = us

        xx = np.stack(xs)
        uu = np.stack(us)

        Q = ComputeCost(xx, uu, self.track_map.TrackLength)
        l.lap_length = len(us)
        l.Q = Q.tolist()
        # print("new adding trajectory of length", len(xs), len(us))
        self.lap_data.append(l)

    def select_points(self, x0):
        # SS, uSS, Qfun, TimeSS = self.new_to_old()
        SS_PointSelectedTot      = np.empty((self.n, 0))
        Qfun_SelectedTot         = np.empty((0))
        for jj in range(0, self.numSS_it):
            SS_PointSelected, Qfun_Selected = SelectPoints(self.SS, self.Qfun, self.it - jj - 1, x0, self.numSS_Points / self.numSS_it, self.shift)
            SS_PointSelectedTot =  np.append(SS_PointSelectedTot, SS_PointSelected, axis=1)
            Qfun_SelectedTot    =  np.append(Qfun_SelectedTot, Qfun_Selected, axis=0)

        self.SS_PointSelectedTot = SS_PointSelectedTot
        self.Qfun_SelectedTot    = Qfun_SelectedTot
        return SS_PointSelectedTot, Qfun_SelectedTot

    def add_trajectory(self, xs, us):
        # print("old adding trajectory of length", len(xs), len(us))
        it = self.it
        end_it = len(us)
        self.TimeSS[it] = end_it
        X, U = np.stack(xs), np.stack(us)
        self.SS[0:(end_it + 1), :, it] = X
        # self.SS_glob[0:(end_it + 1), :, it] = ClosedLoopData.x_glob[0:(end_it + 1), :]
        self.uSS[0:end_it, :, it] = U
        self.Qfun[0:(end_it + 1), it] = ComputeCost(X, U, self.track_map.TrackLength)

        for i in np.arange(0, self.Qfun.shape[0]):
            if self.Qfun[i, it] == 0:
                self.Qfun[i, it] = self.Qfun[i - 1, it] - 1

        if self.it == 0:
            # TODO: made this more general
            self.LinPoints = self.SS[1:self.N + 2, :, it]
            self.LinInput = self.uSS[1:self.N + 1, :, it]

        self.it = self.it + 1

    def addPoint(self, x, u=None, i=None):
        """at iteration j add the current point to SS, uSS and Qfun of the previous iteration
        Arguments:
            x: current state
            u: current input
            i: at the j-th iteration i is the time at which (x,u) are recorded
        """
        step_within_lap = i
        if step_within_lap == 0:
            # if self.current_lap is not None:
            #     self.new_add_trajectory(self.current_lap.x, self.current_lap.u)
            #     self.lap_data.append(self.current_lap)
            if self.current_lap is None:
                self.current_lap = LapData()

        self.current_lap.x.append(x)
        if u is None:
            self.add_trajectory(self.current_lap.x, self.current_lap.u)
            self.current_lap = None
        else:
            self.current_lap.u.append(u.copy())

            Counter = self.TimeSS[self.it - 1]
            # print("Counter", Counter, 'dx', Counter + i + 1)
            self.SS[Counter + i + 1, :, self.it - 1] = x + np.array([0, 0, 0, 0, self.track_map.TrackLength, 0])
            self.uSS[Counter + i + 1, :, self.it - 1] = u
            if self.Qfun[Counter + i + 1, self.it - 1] == 0:
                self.Qfun[Counter + i + 1, self.it - 1] = self.Qfun[Counter + i, self.it - 1] - 1


    def ComputeIndex(self, h, SS, uSS, TimeSS, it, x0, stateFeatures, scaling, MaxNumPoint):
        # What to learn a model such that: x_{k+1} = A x_k  + B u_k + C
        oneVec = np.ones((SS[0:TimeSS[it], :, it].shape[0] - 1, 1))
        x0Vec = (np.dot(np.array([x0]).T, oneVec.T)).T

        DataMatrix = np.hstack((SS[0:TimeSS[it] - 1, stateFeatures, it], uSS[0:TimeSS[it] - 1, :, it]))

        diff = np.dot((DataMatrix - x0Vec), scaling)
        norm = la.norm(diff, 1, axis=1)
        indexTot = np.squeeze(np.where(norm < h))

        if (indexTot.shape[0] >= MaxNumPoint):
            index = np.argsort(norm)[0:MaxNumPoint]
        else:
            index = indexTot

        K = (1 - (norm[index] / h) ** 2) * 3 / 4
        return index, K

    def get_latest_xu(self):
        it = self.it - 1
        dT = self.TimeSS[it]
        xx = self.SS[:dT + 1, :, it]
        uu = self.uSS[:dT, :, it]
        return xx, uu

    def RegressionAndLinearization(self, x0, u0, MaxNumPoint, h, lamb, stateFeatures, scaling, features_inputs_reg):
        '''
        The functionality to perform the local linearization should probably be pushed onto the sample safe set
        which can maintain information about the use of multiple trials etc.

        magic values
        '''
        n, d = self.n, self.d
        it = self.it
        usedIt = range(it - 2, it)
        SS = self.SS
        uSS = self.uSS
        TimeSS = self.TimeSS

        Bi = np.zeros((n, d))
        Ai = np.zeros((n, n))
        Ci = np.zeros((n, 1))
        # Compute Index to use
        xLin = np.hstack((x0[stateFeatures], u0))
        indexSelected = []
        K = []
        for it in usedIt:
            DataMatrix = np.hstack((SS[0:TimeSS[it] - 1, stateFeatures, it], uSS[0:TimeSS[it] - 1, :, it]))
            diff = (DataMatrix - xLin ) @ scaling
            norm = la.norm(diff, 1, axis=1) # Find difference between previous state-action pairs and current state-action
            indexTot = np.where(norm < h)[0]
            index = np.argsort(norm)[0:MaxNumPoint] if len(indexTot) >= MaxNumPoint else indexTot
            dK = (1 - (norm[index] / h) ** 2) * 3 / 4 # Find kernel weights according to epinichikow kernel, see https://en.wikipedia.org/wiki/Kernel_(statistics)
            # Append these to collections
            K.append(dK)
            indexSelected.append(index)
        Ktot = np.diag(np.concatenate(K))

        ''' 
        Weighted ridge regression to produce one-step predictions 
        '''
        def llreg(infet, y_index):
            X0, y = [], []
            for idx, it in zip(indexSelected, usedIt):
                X0.append(np.hstack((np.squeeze(SS[np.ix_(idx, stateFeatures, [it])]),
                                     np.squeeze(uSS[np.ix_(idx, infet, [it])], axis=2))))
                y.append( np.squeeze(SS[np.ix_(idx + 1, [y_index], [it])]))

            X0 = np.concatenate(X0,axis=0)
            y = np.concatenate(y)

            M = np.hstack((X0, np.ones((X0.shape[0], 1))))
            Q =  (M.T @ Ktot) @ M + lamb * np.eye(M.shape[1])
            b =  M.T @ Ktot @ y
            Result = np.linalg.solve(Q,b) # Solves the normal equation x (M^T K M + lambda I) = M^T K y
            A = Result[0:len(stateFeatures)]
            B = Result[len(stateFeatures):(len(stateFeatures) + len(infet))]
            C = Result[-1]
            return A, B, C
        # local linear regression on some of the dynamics:
        for yIndex, inputFeatures in features_inputs_reg:
            Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex] = llreg(infet=inputFeatures,y_index=yIndex)
        return Ai, Bi, Ci, indexSelected


def SelectPoints(SS, Qfun, it, x0, numSS_Points, shift):
    # selects the closest point in the safe set to x0
    # returns a subset of the safe set which contains a range of points ahead of this point
    x = SS[:, :, it] # Extract all elements from safe set
    oneVec = np.ones((x.shape[0], 1))
    x0Vec = (np.dot(np.array([x0]).T, oneVec.T)).T # Make a lot of copies of current state
    diff = x - x0Vec # Compare each of the SS elements to current state
    norm = la.norm(diff, 1, axis=1) # Find distance between current sates and safe set points
    MinNorm = np.argmin(norm)

    # Use closest SS point and the next numSS_Points
    if (MinNorm + shift >= 0):
        SS_Points = x[int(shift + MinNorm):int(shift + MinNorm + numSS_Points), :].T
        Sel_Qfun = Qfun[int(shift + MinNorm):int(shift + MinNorm + numSS_Points), it]
    else:
        SS_Points = x[int(MinNorm):int(MinNorm + numSS_Points), :].T
        Sel_Qfun = Qfun[int(MinNorm):int(MinNorm + numSS_Points), it]

    return SS_Points, Sel_Qfun

def ComputeCost(x, u, TrackLength):
    Cost = 10000 * np.ones((x.shape[0]))  # The cost has the same elements of the vector x --> time +1
    # Now compute the cost moving backwards in a Dynamic Programming (DP) fashion.
    # We start from the last element of the vector x and we sum the running cost
    for i in range(0, x.shape[0]):
        if (i == 0):  # Note that for i = 0 --> pick the latest element of the vector x
            Cost[x.shape[0] - 1 - i] = 0
        elif x[x.shape[0] - 1 - i, 4] < TrackLength:
            Cost[x.shape[0] - 1 - i] = Cost[x.shape[0] - 1 - i + 1] + 1
        else:
            Cost[x.shape[0] - 1 - i] = 0
    return Cost
# 225
