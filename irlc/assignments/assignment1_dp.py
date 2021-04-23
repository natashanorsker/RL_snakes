"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from unitgrade.unitgrade import QuestionGroup, QItem, Report, QPrintItem
import numpy as np
from irlc import train
import gym
from irlc.ex02.search_problem import GraphSP
from irlc.ex02.dp_forward import dp_forward
import irlc

class Week1QuestionGroup(QuestionGroup):
    title = "Problems from first week"

    class ChessTournamentQuestion(QPrintItem):
        tol = 0.05
        testfun = QPrintItem.assertL2Relative

        def compute_answer_print(self):
            from irlc.ex01.chess import main
            main()

        def process_output(self, res, txt, numbers):
            return numbers

    class GraphTraversalQuestion(QPrintItem):
        def compute_answer_print(self):
            from irlc.ex01 import graph_traversal
            graph_traversal.main()

    class FrozenLakeQuestion(QPrintItem):
        def compute_answer_print(self):
            from irlc.ex01.frozen_lake import FrozenAgentDownRight
            env = gym.make("FrozenLake-v0")
            stats, _ = train(env, FrozenAgentDownRight(env), num_episodes=1000, verbose=False)
            Ravg = np.mean([stat['Accumulated Reward'] for stat in stats])
            return np.round(Ravg/5, decimals=2)*5

        def process_output(self, res, txt, numbers):
            return res



class DPGroup(QuestionGroup):
    title = "Dynamical Programming group"

    class InventoryDP(QPrintItem):
        title = "Inventory control"
        def compute_answer_print(self):
            from irlc.ex02.inventory import main
            main()

    class DPChessmatch(QPrintItem):
        title = "Chessmatch"
        def compute_answer_print(self):
            from irlc.ex02.chessmatch import ChessMatch, policy_rollout
            from irlc.ex02.dp import DP_stochastic
            import numpy as np
            N = 2
            pw = 0.45
            pd = 0.8
            cm = ChessMatch(N, pw=pw, pd=pd)
            J, pi = DP_stochastic(cm)
            return pi, np.round(J[0][0], decimals=2)

        def process_output(self, res, txt, numbers):
            return res

class DPAgentInventory(QuestionGroup):
    title = "DP Agent and inventory control"

    class InventoryDP(QPrintItem):
        tol = 0.05
        testfun = QItem.assertL2Relative

        def compute_answer_print(self):
            from irlc.ex02.dp_agent import main
            main()

        def process_output(self, res, txt, numbers):
            return numbers[-2:]


class DPForwardGroup(QuestionGroup):
    title = "Forward-DP problems"

    class SmallGraphJ(QPrintItem):
        def compute_answer_print(self):
            sp = GraphSP(start=2, goal=5)
            J_sp, pi_sp, path = dp_forward(sp, N=sp.vertices - 1)
            print(J_sp)

    class SmallGraphPolicy(QPrintItem):
        def compute_answer_print(self):
            t = 5
            s = 2
            sp = GraphSP(start=2, goal=5)
            J_sp, pi_sp, path = dp_forward(sp, N=sp.vertices - 1)
            print(pi_sp)

    class Travelman(QPrintItem):
        def compute_answer_print(self):
            from irlc.ex02.travelman import main
            main()

        def process_output(self, res, txt, numbers):
            return numbers[:2]


    class NQueens(QPrintItem):
        def compute_answer_print(self):
            from irlc.ex02.queens import QueensDP
            from irlc.ex02.search_problem import DP2SP
            N = 4
            q = QueensDP(N)
            s = ()  # first state is the empty chessboard
            q_sp = DP2SP(q, initial_state=s)
            J, actions, path = dp_forward(q_sp, N)
            board = path[-2][0]
            return QueensDP.valid_pos_(q, board[:-1], board[-1] )

        def process_output(self, res, txt, numbers):
            return res

class FrozenDPGroup(QuestionGroup):
    title = "Frozen lake and DP"

    class FrozenLake(QPrintItem):
        def compute_answer_print(self):
            from irlc.ex02.frozen_lake_dp import Gym2DPModel
            import gym
            from irlc.ex02.dp import DP_stochastic
            env = gym.make("FrozenLake-v0")
            frozen_lake = Gym2DPModel(gym_env=env)
            J, pi = DP_stochastic(frozen_lake)
            return pi[0], np.round(J[0][env.reset()], decimals=2)
        def process_output(self, res, txt, numbers):
            return res


# class DPPacmanMoveCheck(QPrintItem):
#     from irlc.ex02.dp_pacman import compute_actual_step_distribution, SS2tiny, SS0tiny, SS1tiny, pacman_dp_J_sample
#     tol = 0.07
#     test = QPrintItem.assertL2
#     layout_str = SS2tiny
#
#     def compute_answer_print(self):
#         from irlc.ex02.dp_pacman import compute_actual_step_distribution, SS2tiny, SS0tiny, SS1tiny, pacman_dp_J_sample
#         Pw, Ew = compute_actual_step_distribution(layout_str=self.layout_str, T=1000)
#         val = sum([abs(Pw[s] - Ew[s]) for s in Pw])
#         return val
#
#     def process_output(self, res, txt, numbers):
#         return numbers[:6] + [res]
#
#
# class DPPacmanJCheck(QPrintItem):
#     from irlc.ex02.dp_pacman import compute_actual_step_distribution, SS2tiny, SS0tiny, SS1tiny, pacman_dp_J_sample
#     tol = 0.1
#     test = QPrintItem.assertL2
#     layout_str = SS2tiny
#
#     def compute_answer_print(self):
#         from irlc.ex02.dp_pacman import compute_actual_step_distribution, SS2tiny, SS0tiny, SS1tiny, pacman_dp_J_sample
#         pacman_dp_J_sample(layout_str=self.layout_str, N=5, T=1000)
#
#     def process_output(self, res, txt, numbers):
#         return numbers[-3:]
#
#
# class DPPacman0(QuestionGroup):
#     from irlc.ex02.dp_pacman import compute_actual_step_distribution, SS2tiny, SS0tiny, SS1tiny, pacman_dp_J_sample
#     title = "Pacman and dynamical programming with no ghosts"
#
#     class DPPacmanMovesNoGhosts(DPPacmanMoveCheck):
#         from irlc.ex02.dp_pacman import compute_actual_step_distribution, SS2tiny, SS0tiny, SS1tiny, pacman_dp_J_sample
#         layout_str = SS0tiny
#
#     class DPPacmanSamplingNoGhosts(DPPacmanJCheck):
#         from irlc.ex02.dp_pacman import compute_actual_step_distribution, SS2tiny, SS0tiny, SS1tiny, pacman_dp_J_sample
#         layout_str = SS0tiny
#
#
# class DPPacman1(QuestionGroup):
#     from irlc.ex02.dp_pacman import compute_actual_step_distribution, SS2tiny, SS0tiny, SS1tiny, pacman_dp_J_sample
#     title = "Pacman and dynamical programming with one ghost"
#
#     class DPPacmanMovesOneGhost(DPPacmanMoveCheck):
#         from irlc.ex02.dp_pacman import compute_actual_step_distribution, SS2tiny, SS0tiny, SS1tiny, pacman_dp_J_sample
#         layout_str = SS1tiny
#
#     class DPPacmanSamplingOneGhost(DPPacmanJCheck):
#         from irlc.ex02.dp_pacman import compute_actual_step_distribution, SS2tiny, SS0tiny, SS1tiny, pacman_dp_J_sample
#         layout_str = SS1tiny
#
#
# class DPPacman2(QuestionGroup):
#     title = "Pacman and dynamical programming with two ghosts"
#
#     class DPPacmanMovesTwoGhosts(DPPacmanMoveCheck):
#         from irlc.ex02.dp_pacman import compute_actual_step_distribution, SS2tiny, SS0tiny, SS1tiny, pacman_dp_J_sample
#         layout_str = SS2tiny
#
#     class DPPacmanSamplingTwoGhosts(DPPacmanJCheck):
#         from irlc.ex02.dp_pacman import compute_actual_step_distribution, SS2tiny, SS0tiny, SS1tiny, pacman_dp_J_sample
#         layout_str = SS2tiny

class GraphSearchingQuestion(QPrintItem):
    def compute_answer_print(self):
        raise Exception("Must implement this")

    def process_output(self, res, txt, numbers):
        return numbers[-2:]


class KarateSearchGroup(QuestionGroup):
    title = "Solve Karate search problem using BFS and Uniform search"

    class BFSKarateItem(QPrintItem):
        def compute_answer_print(self):
            from irlc.ex03.karate_search import KarateGraphSP, breadthFirstSearch
            Gk_unweighted = KarateGraphSP(start=14, goal=16, weighted=False)
            (actions, path), cost, visited, num_expanded = breadthFirstSearch(Gk_unweighted)
            return cost

        def process_output(self, res, txt, numbers):
            return res

    class DFSKarateItem(QPrintItem):
        def compute_answer_print(self):
            from irlc.ex03.karate_search import KarateGraphSP, breadthFirstSearch
            Gk_unweighted = KarateGraphSP(start=14, goal=16, weighted=False)
            (actions, path), cost, visited, num_expanded = breadthFirstSearch(Gk_unweighted)
            return path[0] == 14 and path[-1] == 16 and len(path) >= 3

        def process_output(self, res, txt, numbers):
            return res

    class UniformKarateItem(QPrintItem):
        def compute_answer_print(self):
            from irlc.ex03.karate_search import KarateGraphSP, uniformCostSearch
            G = KarateGraphSP(start=14, goal=16, weighted=True)
            (actions, path), cost, visited, num_expanded = uniformCostSearch(G)
            return cost

        def process_output(self, res, txt, numbers):
            return res


class GraphSearchGenericItem(QPrintItem):
    from irlc.ex03.pacsearch_agents import BFSAgent, DFSAgent, UniformCostAgent
    from irlc.ex03.pacman_problem_positionsearch import GymPositionSearchProblem
    layout = 'smallMaze'
    Agent = DFSAgent
    Problem = GymPositionSearchProblem

    def compute_answer_print(self):
        from irlc.ex03.pacman_problem_positionsearch import maze_search
        maze_search(layout=self.layout, SAgent=self.Agent, problem=self.Problem(), render=False)
        # gminmax(layout=self.layout, Agent=self.Agent, problem=self.Problem(), render=False)  # part 3

    def process_output(self, res, txt, numbers):
        return numbers[-2:]


class DepthFirstSearchGroup(QuestionGroup):
    title = "Finding a Fixed Food Dot using Depth First Search"

    class TinyFixedDotSearch(GraphSearchGenericItem):
        layout = 'tinyMaze'
        from irlc.ex03.pacsearch_agents import BFSAgent, DFSAgent, UniformCostAgent
        from irlc.ex03.pacman_problem_positionsearch import GymPositionSearchProblem
        Agent = DFSAgent
        Problem = GymPositionSearchProblem

    class MediumFixedDotSearch(TinyFixedDotSearch):
        layout = 'mediumMaze'

    class BigFixedDotSearch(TinyFixedDotSearch):
        layout = 'bigMaze'

class BFSSearchGroup(QuestionGroup):
    title = "Finding a Fixed Food Dot using Breadth First Search"

    class BFSTinyFixedDotSearch(GraphSearchGenericItem):
        layout = 'tinyMaze'
        from irlc.ex03.pacsearch_agents import BFSAgent, DFSAgent, UniformCostAgent
        from irlc.ex03.pacman_problem_positionsearch import GymPositionSearchProblem
        Agent = BFSAgent
        Problem = GymPositionSearchProblem

    class BFSMediumFixedDotSearch(BFSTinyFixedDotSearch):
        layout = 'mediumMaze'

    class BFSBigFixedDotSearch(BFSTinyFixedDotSearch):
        layout = 'bigMaze'


class AStarSearchGroup(QuestionGroup):
    title = "Finding a Fixed Food Dot using Astar search"

    class AstarTinyFixedDotSearch(GraphSearchGenericItem):
        layout = 'tinyMaze'
        from irlc.ex03.pacman_problem_positionsearch_astar import AStarAgent
        Agent = AStarAgent
        from irlc.ex03.pacsearch_agents import BFSAgent, DFSAgent, UniformCostAgent
        from irlc.ex03.pacman_problem_positionsearch import GymPositionSearchProblem
        Problem = GymPositionSearchProblem

    class BFSMediumFixedDotSearch(AstarTinyFixedDotSearch):
        layout = 'mediumMaze'

    class BFSBigFixedDotSearch(AstarTinyFixedDotSearch):
        layout = 'bigMaze'



## Food search groups
class FoodSearchGroup(QuestionGroup):
    title = "Food search problem using various search methods"

    class DFSFoodSearch(GraphSearchGenericItem):
        from irlc.ex03.pacsearch_agents import BFSAgent, DFSAgent, UniformCostAgent
        from irlc.ex03.pacman_problem_positionsearch import GymPositionSearchProblem
        Agent = DFSAgent
        from irlc.ex03.pacman_problem_foodsearch import GymFoodSearchProblem
        Problem = GymFoodSearchProblem
        layout = 'trickySearch'
        def process_output(self, res, txt, numbers):
            return numbers[-2:]

    class BFSFoodSearch(GraphSearchGenericItem):
        from irlc.ex03.pacsearch_agents import BFSAgent, DFSAgent, UniformCostAgent
        from irlc.ex03.pacman_problem_positionsearch import GymPositionSearchProblem
        Agent = BFSAgent
        from irlc.ex03.pacman_problem_foodsearch import GymFoodSearchProblem
        Problem = GymFoodSearchProblem
        layout = 'trickySearch'
        def process_output(self, res, txt, numbers):
            return numbers[-2:]

    class AStarFoodSearch(GraphSearchGenericItem):
        from irlc.ex03.pacman_problem_positionsearch_astar import AStarAgent
        Agent = AStarAgent
        from irlc.ex03.pacman_problem_foodsearch import GymFoodSearchProblem
        Problem = GymFoodSearchProblem
        layout = 'trickySearch'
        def process_output(self, res, txt, numbers):
            return numbers[-2:]

from irlc.ex03.multisearch_agents import gminmax, GymMinimaxAgent, GymExpectimaxAgent
from irlc.ex03.multisearch_alphabeta import GymAlphaBetaAgent
class MultiAgentItem(QPrintItem):
    layout = 'minimaxClassic'
    Agent = GymMinimaxAgent
    depth = 3
    episodes = 100
    results = None
    tol = .25
    testfun = QPrintItem.assertL2Relative

    def compute_answer_print(self):
        gminmax(layout=self.layout, Agent=self.Agent, depth=self.depth, episodes=self.episodes, render=False)

    def process_output(self, res, txt, numbers):
        win_pr = numbers[2]
        avg_length = numbers[3]
        search_nodes = numbers[4]
        MultiAgentItem.results = dict(win_pr=win_pr, avg_length=avg_length)
        return win_pr


class MultiAgentItemLength(MultiAgentItem):
    tol = 1
    testfun = QPrintItem.assertL2Relative

    def compute_answer_print(self):
        pass

    def process_output(self, res, txt, numbers):
        return MultiAgentItem.results['avg_length']


class MiniMaxGroup(QuestionGroup):
    title = "Multi-search Minimax questions (note: long runtime and may be influenced by chance)"
    Agent = GymExpectimaxAgent

    class MiniMaxWinItem(MultiAgentItem):
        pass

    class MiniMaxLengthItem(MultiAgentItemLength):
        pass


class ExpectiMaxGroup(QuestionGroup):
    title = "Multi-search Expectimax question (note: long runtime and may be influenced by chance)"

    class EmaxWinItem(MultiAgentItem):
        def compute_answer_print(self):
            MultiAgentItem.results = {}  # ['avg_length']
            MultiAgentItem.compute_answer_print(self)

        Agent = GymExpectimaxAgent

    class EmaxLengthItem(MultiAgentItemLength):
        Agent = GymExpectimaxAgent

class AlphaBetaGroup(QuestionGroup):
    title = "Multi-search Alpha-beta question (note: long runtime and may be influenced by chance)"

    class AlphaBetaWinItem(MultiAgentItem):
        def compute_answer_print(self):
            MultiAgentItem.results = {}
            MultiAgentItem.compute_answer_print(self)

        Agent = GymAlphaBetaAgent

    class AlphaBetaLengthItem(MultiAgentItemLength):
        Agent = GymAlphaBetaAgent

class Assignment1DP(Report): #240 total.
    title = "Dynamical Programming and search"
    pack_imports = [irlc]
    individual_imports = []
    questions = [(Week1QuestionGroup, 30),      # Week 1: Everything
                 (DPGroup, 30),                 # Week 2: Various DP Questions
                 (DPAgentInventory, 10),         # Week 2: Agent DP to inventory problem
                 (FrozenDPGroup, 10),            # Week 2: Frozen-lake DP question
                 (DPForwardGroup, 42),          # Week 2: (small graph; Queens, travelman)
                 (KarateSearchGroup, 30),       # Week 3: Basic search (karate)
                 (DepthFirstSearchGroup, 14),    # Week 3, pacman
                 (AStarSearchGroup, 10),
                 (FoodSearchGroup, 10),
                 (BFSSearchGroup, 14),           # Week 3, pacman
                 (MiniMaxGroup, 20),            # Week 3: Minimax agent
                 (ExpectiMaxGroup, 20),         # Week 3: Expectimax agent
                 (AlphaBetaGroup, 20),  # Week 3: Expectimax agent
                 ]
    # (DPPacman0, 2),
    # (DPPacman1, 2),
    # (DPPacman2, 2),


if __name__ == '__main__':
    from unitgrade.unitgrade_helpers import evaluate_report_student
    evaluate_report_student(Assignment1DP() )
