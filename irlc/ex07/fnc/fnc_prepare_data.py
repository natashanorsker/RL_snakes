"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for
# education or reserach purposes provided that you provide clear attribution to UC Berkeley,
# including a reference to the papers describing the control framework:
# [1] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks. A Data-Driven
#     Control Framework." In IEEE Transactions on Automatic Control (2017).
#
# [2] Ugo Rosolia, Ashwin Carvalho, and Francesco Borrelli. "Autonomous racing using learning model predictive control."
#     In 2017 IEEE American Control Conference (ACC)
#
# [3] Maximilian Brunner, Ugo Rosolia, Jon Gonzales and Francesco Borrelli "Repetitive learning model predictive
#     control: An autonomous racing old" In 2017 IEEE Conference on Decision and Control (CDC)
#
# [4] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks: A Computationally
#     Efficient Approach for Linear System." IFAC-PapersOnLine 50.1 (2017).
#
# Attibution Information: Code developed by Ugo Rosolia
# (for clarifications and suggestions please write to ugo.rosolia@berkeley.edu).
#
# ----------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append("../../ex105/") # to find FNC module.
from irlc.ex04.cost_discrete import DiscreteQRCost
# from irlc.ex105.sympc.simulator import SymSimulator
from irlc.ex07.lmpc_agent import LMPCAgent
import numpy as np
import pickle
from irlc.car.sym_map import unityTestChangeOfCoordinates
from irlc.car.sym_car_model import CarEnvironment


def setup_lmpc_controller():
    np.random.seed(1)  # crash if seed = 0
    # freeze_support()
    # osqp = OSQP()
    # testCoordChangeFlag = 1
    # plotOneStepPredictionErrors = 1

    # ======================================================================================================================
    # ==================================== Initialize parameters for LMPC ==================================================
    # ======================================================================================================================
    dt = 1.0 / 10.0  # Controller discretization time
    N = 12  # Horizon length
    n = 6 # State dimentions
    d = 2  # State and Input dimension
    TimeLMPC = 400  # Simulation time. I got no idea what this one does... determines something about stats/max length in SS set.
    Laps = 10 + 1  # Total LMPC laps
    # v0 = 0.5  # Starting Velocity
    vt = 2.0  # Reference velocity

    # Safe Set Parameter
    # LMPC_Solver = "OSQP"  # Can pick CVX for cvxopt or OSQP. For OSQP uncomment line 14 in LMPC.py
    numSS_it = 2  # Number of trajectories used at each iteration to build the safe set
    numSS_Points = 32 + N  # Number of points to select from each trajectory to build the safe set
    shift = 0  # Given the closed point, x_t^j, to the x(t) select the SS points from x_{t+shift}^j

    # Tuning Parameters
    Qslack = 5 * np.diag([10, 1, 1, 1, 10, 1])  # Cost on the slack variable for the terminal constraint
    Q_LMPC = 0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # State cost x = [vx, vy, wz, epsi, s, ey]
    R_LMPC = 1 * np.diag([1.0, 1.0])  # Input cost u = [delta, a]
    Rd_LMPC = 5 * np.array([1.0, 1.0])  # Input rate cost u

    cost_slack = DiscreteQRCost(state_size=n, action_size=d, Q=Qslack)
    cost_u_deriv = DiscreteQRCost(state_size=n, action_size=d, R=Rd_LMPC)

    x_linear = np.array([vt, 0, 0, 0, 0, 0])
    cost = DiscreteQRCost(state_size=n, action_size=d, R=R_LMPC, Q=Q_LMPC, q=x_linear)  # self.env.cost.x_target
    # from irlc.ex05.sympc.sym_car_model import SymbolicBicycleModel
    # car = SymbolicBicycleModel(map_width=0.8, cost=cost)  # Initialize the bicycle model. Contains the map.


    car = CarEnvironment(map_width=0.8, cost=cost, max_laps=8, hot_start=True)  # Initialize the bicycle model. Contains the map.
    car.reset()

    print("Starting LMPC")

    LMPController = LMPCAgent(numSS_Points, numSS_it, N=N, shift=shift, dt=dt,
                              Laps=Laps, TimeLMPC=TimeLMPC,
                              env=car, cost_slack=cost_slack, cost_u_deriv=cost_u_deriv,
                              save_spy_matrices=False,
                              epochs=Laps)


    # Unpack these variables appropriately:
    def unpack_trajectory(CLD):
        j = (CLD.x[:, 4] > car.map.TrackLength).nonzero()[0][0]
        xs = [CLD.x[i] for i in range(j) ]
        us = [CLD.u[i] for i in range(j-1) ]
        LMPController.add_trajectory(xs, us, loading_from_disk=True)
        return xs, us
        # LMPController.SSset.add_trajectory(xs, us)
        # LMPController.SSset.new_add_trajectory(xs, us)
    import os

    df = [sys.path[0]+'/data/ClosedLoopDataLTV_MPC.obj', sys.path[0]+'/data/ClosedLoopDataPID.obj']
    rs = []
    for f in df:
        with open(f, 'rb') as file_data:
            dt = pickle.load(file_data, encoding='latin1')
        xs, us = unpack_trajectory(dt)

        rs.append(dict(x=xs, u=us, fn=os.path.basename(f)))
    ofile = os.path.join(os.path.dirname(__file__), "../lmpc_initial_lap_data.pkl")
    with open(ofile, 'bw') as f:
        pickle.dump(rs, f)

    # Test coordinate system transformation
    # Open simple solutions that can be used as a starting solutions
    with open(sys.path[0]+'/data/ClosedLoopDataPID.obj', 'rb') as file_data:
        ClosedLoopDataPID = pickle.load(file_data,encoding='latin1')

    with open(sys.path[0]+'/data/ClosedLoopDataLTV_MPC.obj', 'rb') as file_data:
        ClosedLoopDataLTV_MPC = pickle.load(file_data,encoding='latin1')

    # unpack_trajectory(ClosedLoopDataLTV_MPC)
    # unpack_trajectory(ClosedLoopDataPID)
    unityTestChangeOfCoordinates(LMPController.env.map, ClosedLoopDataPID)
    unityTestChangeOfCoordinates(LMPController.env.map, ClosedLoopDataLTV_MPC)


if __name__ == "__main__":
    setup_lmpc_controller()
