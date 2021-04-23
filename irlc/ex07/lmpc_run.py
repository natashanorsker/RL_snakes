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

from irlc.ex04.cost_discrete import DiscreteQRCost
from irlc.ex07.lmpc_agent import LMPCAgent
from irlc.ex07.lmpc_plot import plotClosedLoopLMPC
import numpy as np
import matplotlib.pyplot as plt
import pickle
from irlc.car.sym_car_model import CarEnvironment
from irlc import train
import os

def setup_lmpc_controller(max_laps=9):
    np.random.seed(3)  # crash if seed = 0
    # freeze_support() # Used for optimizer and multiprocessing support (deprecated)
    # ======================================================================================================================
    # ==================================== Initialize parameters for LMPC ==================================================
    # ======================================================================================================================
    dt = 1.0 / 10.0  # Controller discretization time
    N = 12  # Horizon length
    n = 6 # State dimentions
    d = 2  # State and Input dimension
    TimeLMPC = 400  # Simulation time. I got no idea what this one does... determines something about stats/max length in SS set.
    Laps = 10 + 1  # Total LMPC laps
    vt = 2.0  # Reference velocity

    # Safe Set Parameter
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

    x_linear = np.array([vt, 0, 0, 0, 0, 0]) # appears irrelevant for problem.
    cost = DiscreteQRCost(state_size=n, action_size=d, R=R_LMPC, Q=Q_LMPC, q=x_linear)  # self.env.cost.x_target

    car = CarEnvironment(map_width=0.8, cost=cost, max_laps=max_laps, hot_start=True)  # Initialize the bicycle model. Contains the map.
    car.reset()
    print("Starting LMPC")
    LMPController = LMPCAgent(numSS_Points, numSS_it, N=N, shift=shift, dt=dt,
                              Laps=Laps, TimeLMPC=TimeLMPC,
                              env=car, cost_slack=cost_slack, cost_u_deriv=cost_u_deriv,
                              save_spy_matrices=False,
                              epochs=Laps)

    print("Loading initial PID lap data")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + "/lmpc_initial_lap_data.pkl", 'rb') as f:
        rs = pickle.load(f)
    for lap in rs:
        print("> Loading initial lap data from", lap['fn'])
        LMPController.add_trajectory(lap['x'], lap['u'], loading_from_disk=True)

    return car, LMPController

def main(show_episode=False):
    car, LMPController = setup_lmpc_controller(max_laps=9)
    # ---- MAIN LMPC TRAINING LOOP ----
    from irlc import VideoMonitor
    if show_episode:
        car = VideoMonitor(car)
    stats_, traj_ = train(car, LMPController, num_episodes=1)
    car.close()

    # dt = car.dt
    ClosedLoopLMPC = LMPController.get_closed_loop_lmpc()
    LMPCOpenLoopData = LMPController.LMPCOpenLoopData

    '''
    Convert local to global replay buffer for plotting
    '''
    def replaybuffer2globals(SSset):
        SS_glob = np.copy(SSset.SS)
        for k in range(SSset.it):
            T = SSset.TimeSS[k]
            for i in range(T):
                ds = SSset.SS[i, :, k]
                s, ey, epsi = car.map.getGlobalPosition(ds[4], ds[5], epsi=ds[3])
                SS_glob[i, 3:, k] = [epsi, s, ey]
        return SS_glob
    LMPController.SSset.SS_glob = replaybuffer2globals(LMPController.SSset)

    print("===== LMPC terminated")
    # PLOT TRACK/PREDICTIONS
    # with open("lmpc_simulation_output.txt", 'w') as f:
    #     s = "\n".join([f"Lap time at iteration {i} is {LMPController.SSset.Qfun[0, i]*dt} seconds" for i in range(LMPController.it)] )
    #     print(s)
    #     f.write(s)

    print("===== Start Plotting")
    plt.close()
    plotClosedLoopLMPC(LMPController, car.map)
    plt.show()
    # for i in range(Laps-1):
    #     # animation_xy(car.map, LMPCOpenLoopData, LMPController, i)
    #     plt.close()
    #saveGif_xyResults(map, LMPCOpenLoopData, LMPController, 6)
    # animation_states(car.map, LMPCOpenLoopData, LMPController, Laps-1)
    # Save_statesAnimation(map, LMPCOpenLoopData, LMPController, 5)

if __name__ == "__main__":
    main()
