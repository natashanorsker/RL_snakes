"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from unitgrade.unitgrade import QuestionGroup, Report, QPrintItem
from unitgrade.unitgrade import Capturing
import irlc
from irlc.car.sym_car_model import CarEnvironment
from irlc.ex04.pid_car import PIDCarAgent
from irlc.ex04.model_boing import BoingEnvironment
from irlc import train
import numpy as np

class Target:
    target = 0.4

class DiscretizationGroup(QuestionGroup):
    title = "Simulation and discretization"
    class RK4(QPrintItem):
        title = "RK4 integration test"
        tol = 1e-6
        def compute_answer_print(self):
            from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment
            env = GymSinCosPendulumEnvironment()
            env.reset()

            env.step([0.5])
            x, _, _, _ = env.step([0.9])
            return x
        def process_output(self, res, txt, numbers):
            return res

    class ExponentialIntegration(QPrintItem):
        title = "Exponential Integration"
        tol = 1e-6
        def compute_answer_print(self):
            from irlc.ex04.model_harmonic import DiscreteHarmonicOscilatorModel
            dmod = DiscreteHarmonicOscilatorModel()
            x = dmod.f_discrete(dmod.reset(), [0.5])
            return x

        def process_output(self, res, txt, numbers):
            return res

class PIDQuestionGroup(QuestionGroup):
    title = "PID Control"

    class PIDLocomotive(QPrintItem):
        tol = 1e-4
        target = 0
        Kp, Ki, Kd = 40, 0, 0
        slope = 0
        def compute_answer_print(self):
            dt = 0.08
            from irlc.ex04.pid_locomotive_agent import LocomotiveEnvironment, PIDLocomotiveAgent
            env = LocomotiveEnvironment(m=10, slope=self.slope, dt=dt, Tmax=5)
            agent = PIDLocomotiveAgent(env, dt=dt, Kp=self.Kp, Ki=self.Ki, Kd=self.Kd, target=self.target)
            stats, traj = train(env, agent, return_trajectory=True,verbose=False)
            return traj[0].state
        def process_output(self, res, txt, numbers):
            return res

    class PIDLocomotiveKd(PIDLocomotive):
        Kd = 10

    class PIDLocomotiveKi(PIDLocomotive):
        slope = 15
        Kd = 10
        Ki = 5


class PIDCar(QPrintItem):
    lt = -1
    def compute_answer_print(self):
        env = CarEnvironment(noise_scale=0, Tmax=80, max_laps=2)
        agent = PIDCarAgent(env, v_target=1.0)
        stats, trajectories = train(env, agent, num_episodes=1, return_trajectory=True)

        d = trajectories[0].state[:, 4]
        lt = len(d) * env.dt / 2
        print("lap time", lt)
        PIDCar.lt = lt
        return (trajectories[0].state, trajectories[0].action)

class PIDCarQuestionGroup(QuestionGroup):
    title = "Control of car environment using PID"

    class PIDCarCheck1QPrintItem(PIDCar):
        max_time = 60
        title = f"Lap time < {max_time}"
        def process_output(self, res, txt, numbers):
            return 0 < PIDCar.lt < 60

    class PIDCarCheck2(QPrintItem):
        max_time = 40
        title = f"Lap time < {max_time}"
        def compute_answer_print(self):
            print("Lap time", PIDCar.lt, "<", self.max_time)
            return 0 < PIDCar.lt < self.max_time

        def process_output(self, res, txt, numbers):
            return res

    class PIDCarCheck3(PIDCarCheck2):
        max_time = 30

    class PIDCarCheck4(PIDCarCheck2):
        max_time = 22

class DirectMethods(QuestionGroup):
    title = "Direct methods z, z0, z_lb/z_ub definitions+"

    def init(self): # self, *args, **kwargs):
        from irlc.ex05.direct import run_direct_small_problem
#        super().__init__(*args, **kwargs)
        from unitgrade.unitgrade import Capturing
        # with Capturing():
            # res = self.compute_answer_print()
        env, solution = run_direct_small_problem()
        self.solution = solution[-1]

    class ZItem(QPrintItem):
        key = 'z'
        def compute_answer_print(self, unmute=False):
            return self.question.solution['inputs'][self.key]

        def process_output(self, res, txt, numbers):
            if isinstance(res, np.ndarray):
                return res
            else:
                return str(res)

    class Z0Item(ZItem):
        key = 'z0'

    class Z_lb_Item(ZItem):
        key = 'z_lb'

    class Z_ub_Item(ZItem):
        key = 'z_ub'


class DirectAgentQuestion(QuestionGroup):
    title = "Direct agent: Basic test of pendulum environment"
    class DirectAgentItem(QPrintItem):
        tol = 0.03 # not tuned

        def compute_answer_print(self):
            from irlc.ex05.direct_agent import train_direct_agent
            stats = train_direct_agent(animate=False)
            return stats[0]['Accumulated Reward']

        def process_output(self, res, txt, numbers):
            return res

class SuccessItem_(QPrintItem):
    def compute_answer_print(self):
        return self.question.solutions[-1]['solver']['success']

class CostItem_(QPrintItem):
    tol = 0.01
    def compute_answer_print(self):
        return self.question.solutions[-1]['solver']['fun']

    def process_output(self, res, txt, numbers):
        return res

class ConstraintVioloationItem_(QPrintItem):
    tol = 0.01
    def compute_answer_print(self):
        return self.question.solutions[-1]['eqC_val']

    def process_output(self, res, txt, numbers):
        return res + 1e-5

class DirectSolverQuestion_(QuestionGroup):
    title = "Direct solver on a small problem"
    def compute_solutions(self):
        from irlc.ex05.direct import run_direct_small_problem
        env, solution = run_direct_small_problem()
        return solution

    def init(self):
        # super().__init__(*args, **kwargs)
        with Capturing():
            self.solutions = self.compute_solutions()

class SmallDirectProblem(DirectSolverQuestion_):
    def compute_solutions(self):
        from irlc.ex05.direct import run_direct_small_problem
        env, solution = run_direct_small_problem()
        return solution

    class SuccessItem(SuccessItem_):
        pass
    class CostItem_(CostItem_):
        pass
    class ConstraintVioloationItem_(ConstraintVioloationItem_):
        pass

class PendulumQuestion(DirectSolverQuestion_):
    title = "Direct solver on the pendulum problem"
    def compute_solutions(self):
        from irlc.ex05.direct_pendulum import compute_pendulum_solutions
        return compute_pendulum_solutions()[1]
    class SuccessItem(SuccessItem_):
        pass
    class CostItem_(CostItem_):
        pass
    class ConstraintVioloationItem_(ConstraintVioloationItem_):
        pass

class CartpoleTimeQuestion(DirectSolverQuestion_):
    title = "Direct solver on the cartpole (minimum time) task"
    def compute_solutions(self):
        from irlc.ex05.direct_cartpole_time import compute_solutions
        return compute_solutions()[1]
    class SuccessItem(SuccessItem_):
        pass
    class CostItem_(CostItem_):
        pass
    class ConstraintVioloationItem_(ConstraintVioloationItem_):
        pass

class CartpoleCostQuestion(DirectSolverQuestion_):
    title = "Direct solver on the cartpole (kelly) task"
    def compute_solutions(self):
        from irlc.ex05.direct_cartpole_kelly import compute_solutions
        return compute_solutions()[1]
    class SuccessItem(SuccessItem_):
        pass
    class CostItem_(CostItem_):
        pass
    class ConstraintVioloationItem_(ConstraintVioloationItem_):
        pass

class BrachistochroneQuestion(DirectSolverQuestion_):
    title = "Brachistochrone (unconstrained)"
    def compute_solutions(self):
        from irlc.ex05.direct_brachistochrone import compute_unconstrained_solutions
        return compute_unconstrained_solutions()[1]
    class SuccessItem(SuccessItem_):
        pass
    class CostItem_(CostItem_):
        pass
    class ConstraintVioloationItem_(ConstraintVioloationItem_):
        pass

class BrachistochroneConstrainedQuestion(DirectSolverQuestion_):
    title = "Brachistochrone (constrained)"
    def compute_solutions(self):
        from irlc.ex05.direct_brachistochrone import compute_constrained_solutions
        return compute_constrained_solutions()[1]
    class SuccessItem(SuccessItem_):
        pass
    class CostItem_(CostItem_):
        pass
    class ConstraintVioloationItem_(ConstraintVioloationItem_):
        pass

matrices = ['L', 'l', 'V', 'v', 'vc']
class LQRQuestion(QuestionGroup):
    title = "LQR, full check of implementation"

    # def __init__(self, *args, **kwargs):
    #     from irlc.ex06.dlqr_check import check_LQR
    #     (L, l), (V, v, vc) = check_LQR()
    #     LQRQuestion.M = list(zip(['L', 'l', 'V', 'v', 'vc'], [L, l, V, v, vc]))
    #     super().__init__(*args, **kwargs)

    def init(self):
        from irlc.ex06.dlqr_check import check_LQR
        (L, l), (V, v, vc) = check_LQR()
        self.M = list(zip(matrices, [L, l, V, v, vc]))

    class CheckMatrixItem(QPrintItem):
        tol = 1e-6
        i = 0
        title = "Checking " + matrices[i] #self.question.M[self.i][0]
        # def __init__(self, *args, **kwargs):
        #     super().__init__(*args, **kwargs)
        #     self.title =

        def compute_answer_print(self):
            return np.stack(self.question.M[self.i][1])

        def process_output(self, res, txt, numbers):
            return res

    class LQRMatrixItem1(CheckMatrixItem):
        i = 1

    class LQRMatrixItem2(CheckMatrixItem):
        i = 2

    class LQRMatrixItem3(CheckMatrixItem):
        i = 3

    class LQRMatrixItem4(CheckMatrixItem):
        i = 4

class BoingQuestion(QuestionGroup):
    title = "Boing flight control with LQR"
    class BoingItem(QPrintItem):
        tol = 1e-6
        def compute_answer_print(self):
            from irlc.ex06.boing_lqr import boing_simulation
            stats, trajectories, env = boing_simulation()
            return trajectories

        def process_output(self, res, txt, numbers):
            return res[-1].state

class RendevouzItem(QPrintItem):
    use_linesearch= False
    tol = 1e-2
    def compute_answer_print(self):
        from irlc.ex06.ilqr_rendovouz_basic import solve_rendovouz
        (xs, us, J_hist, l, L), env = solve_rendovouz(use_linesearch=self.use_linesearch)
        print(J_hist[-1])
        return l, L, xs

    def process_output(self, res, txt, numbers):
        return res[2][-1]

class BasicILQRRendevouzQuestion(QuestionGroup):
    title = "Rendevouz with iLQR (no linesearch)"
    class BasicRendevouzItem(RendevouzItem):
        pass

class ILQRRendevouzQuestion(QuestionGroup):
    title = "Rendevouz with iLQR (with linesearch)"
    class ILQRRendevouzItem(RendevouzItem):
        use_linesearch = True


class ILQRAgentQuestion(QuestionGroup):
    title = "iLQR Agent on Rendevouz"
    class ILQRAgentItem(QPrintItem):
        tol = 1e-2
        def compute_answer_print(self):
            from irlc.ex06.ilqr_agent import solve_rendevouz
            stats, trajectories, agent = solve_rendevouz()
            return trajectories[-1].state[-1]

        def process_output(self, res, txt, numbers):
            return res

class ILQRPendulumQuestion(QuestionGroup):
    title = "iLQR Agent on Pendulum"
    class ILQRAgentItem(QPrintItem):
        tol = 1e-2

        def compute_answer_print(self):
            from irlc.ex06.ilqr_pendulum_agent import Tmax, N
            from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment
            from irlc.ex06.ilqr_agent import ILQRAgent

            dt = Tmax / N
            env = GymSinCosPendulumEnvironment(dt, Tmax=Tmax, supersample_trajectory=True)
            agent = ILQRAgent(env, env.discrete_model, N=N, ilqr_iterations=200, use_linesearch=True)
            stats, trajectories = train(env, agent, num_episodes=1, return_trajectory=True)
            return trajectories[-1].state[-1]

        def process_output(self, res, txt, numbers):
            return np.abs(res)+1

""" Week 7 Questions """
class Week7Group(QuestionGroup):
    title = "MPC and learning-MPC"

class BoingQuestion_(QuestionGroup):
    def make_agent(self):
        return None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BoingItem_(QPrintItem):
    def compute_answer_print(self):
        pass

class LearningItem(QPrintItem):
    tol = 0.05
    def get_agent(self, env):
        return None

    def compute_answer_print(self):
        env = BoingEnvironment(output=[10, 0])
        agent = self.get_agent(env)
        from irlc.ex07.lqr_learning_agents import boing_experiment
        stats, trajectories = boing_experiment(env, agent, num_episodes=4, plot=False)
        return (stats[-1]['Accumulated Reward'], trajectories[-1].state[-1])

    def process_output(self, res, txt, numbers):
        return np.abs(np.concatenate( (np.asarray(res[0]).reshape( (1,) ), res[1] )))+1

class LearningLQRAgentQuestion(QuestionGroup):
    title = "MPCLearningAgent on Boing problem"
    class TrajectoryItem(LearningItem):
        def get_agent(self, env):
            from irlc.ex07.lqr_learning_agents import MPCLearningAgent
            return MPCLearningAgent(env)

class MPCLearningAgentQuestion(QuestionGroup):
    title = "MPCLocalLearningLQRAgent on Boing problem"
    tol = 0.1
    class TrajectoryItem(LearningItem):
        def get_agent(self, env):
            from irlc.ex07.lqr_learning_agents import MPCLocalLearningLQRAgent
            return MPCLocalLearningLQRAgent(env)
        def process_output(self, res, txt, numbers):
            return np.abs(res[1]) + 1

class MPCLearningAgentLocalOptimizeQuestion(QuestionGroup):
    title = "MPCLearningAgentLocalOptimize on Boing problem"
    tol = 0.05
    class TrajectoryItem(LearningItem):
        def get_agent(self, env):
            from irlc.ex07.lqr_learning_agents import MPCLearningAgentLocalOptimize
            return MPCLearningAgentLocalOptimize(env)

class MPCLocalAgentExactDynamicsQuestion(QuestionGroup):
    title = "MPCLocalAgentExactDynamics on Boing problem"
    class TrajectoryItem(LearningItem):
        def get_agent(self, env):
            from irlc.ex07.lqr_learning_agents import MPCLocalAgentExactDynamics
            return MPCLocalAgentExactDynamics(env)

class MPCLearningPendulum(QuestionGroup):
    title = "MPCLearningAgentLocalOptimize on pendulum problem"
    class PendulumItem(QPrintItem):
        title = "Pendulum swingup task"
        tl = 1-np.cos( np.pi/180 * 20)
        def compute_answer_print(self):
            from irlc.ex07.mpc_pendulum_experiment import mk_mpc_pendulum_env
            env_pendulum = mk_mpc_pendulum_env(Tmax=10)
            L = 12
            up = 100
            from irlc.ex07.lqr_learning_agents import MPCLearningAgentLocalOptimize
            agent = MPCLearningAgentLocalOptimize(env_pendulum, horizon_length=L, neighbourhood_size=50, min_buffer_size=50)
            # from irlc import VideoMonitor
            # env_pendulum = VideoMonitor(env_pendulum)
            for _ in range(7):
                stats, trajectories = train(env_pendulum, agent, num_episodes=1, return_trajectory=True)
                cos = trajectories[0].state[:,1]
                up = np.abs(  (cos - 1)[ int(len(cos) * .8): ] )

                if np.all(max(up) < self.tl):
                    break
            return up

        def process_output(self, res, txt, numbers):
            return all(res < self.tl)

class LMPCQuestion(QuestionGroup):
    title = "Learning MPC and the car (the final boss)"
    class PendulumItem(QPrintItem):
        title = "LMPC lap time"
        def compute_answer_print(self):
            from irlc.ex07.lmpc_run import setup_lmpc_controller
            car, LMPController = setup_lmpc_controller(max_laps=8)
            stats_, traj_ = train(car, LMPController, num_episodes=1)

        def process_output(self, res, txt, numbers):
            n = []
            for t in txt.splitlines():
                if not t.startswith("Lap"):
                    continue
                from unitgrade.unitgrade import extract_numbers
                nmb = extract_numbers(t)[1]
                n.append(nmb)
            return min(n) < 8.3


class Assignment2Control(Report): # 240 total.
    title = "Control of discrete and continuous problems"
    pack_imports = [irlc]
    individual_imports = []
    questions = [ # 240 / 4 = 60: [60, x, x]
        (DiscretizationGroup, 20),                  # ok
        (PIDQuestionGroup, 20),                     # ok
        (PIDCarQuestionGroup, 20),                  # ok
        (DirectMethods, 10),                        # ok
        (SmallDirectProblem, 10),                   # ok
        (PendulumQuestion, 5),                      # ok
        (DirectAgentQuestion, 10),                  # ok
        (CartpoleTimeQuestion, 5),                  # ok
        (CartpoleCostQuestion, 5),                  # ok
        (BrachistochroneQuestion, 5),               # ok
        (BrachistochroneConstrainedQuestion, 10),   # ok
        (LQRQuestion, 10),                          # ok ILQR
        (BasicILQRRendevouzQuestion, 10),           # ok
        (ILQRRendevouzQuestion, 10),                # ok
        (BoingQuestion, 10),                        # ok
        (ILQRAgentQuestion,10),                     # ok
        (ILQRPendulumQuestion, 10),                 # ok
        (LearningLQRAgentQuestion, 10),             # ok
        (MPCLearningAgentQuestion, 5),              # ok
        (MPCLearningAgentLocalOptimizeQuestion, 10),# ok
        (MPCLocalAgentExactDynamicsQuestion, 5),    # ok
        (MPCLearningPendulum, 10),                  # ok
        (LMPCQuestion, 20),                         # ok
    ]

if __name__ == '__main__':
    from unitgrade.unitgrade_helpers import evaluate_report_student
    evaluate_report_student(Assignment2Control(), unmute=False, ignore_missing_file=True)
