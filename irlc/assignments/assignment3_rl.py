"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import irlc
from unitgrade.unitgrade import QuestionGroup, QPrintItem, Report
import numpy as np
from irlc import train
import irlc.ex09.envs
import gym

def train_recording(env, agent, trajectories):
    for t in trajectories:
        env.reset()
        for k in range(len(t.action)):
            s = t.state[k]
            r = t.reward[k]
            a = t.action[k]
            sp = t.state[k+1]
            agent.pi(s,k)
            agent.train(s, a, r, sp, done=k == len(t.action)-1)

class BanditItem(QPrintItem):
    # tol = 1e-6
    tol = 1e-2 # tie-breaking in the gradient bandit is ill-defined.
    title = "Value (Q) function estimate"
    testfun = QPrintItem.assertL2

    def get_env_agent(self):
        from irlc.ex08.simple_agents import BasicAgent
        from irlc.ex08.bandits import StationaryBandit
        env = StationaryBandit(k=10, )
        agent = BasicAgent(env, epsilon=0.1)
        return env, agent

    def precompute_payload(self):
        env, agent = self.get_env_agent()
        _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
        return trajectories, agent.Q

    def compute_answer_print(self):
        trajectories, Q = self.precomputed_payload()
        env, agent = self.get_env_agent()
        train_recording(env, agent, trajectories)
        self.Q = Q
        self.question.agent = agent
        return agent.Q

    def process_output(self, res, txt, numbers):
        return res

    def test(self, computed, expected):
        super().test(computed, self.Q)

class BanditItemActionDistribution(QPrintItem):
    # Assumes setup has already been done.
    title = "Action distribution test"
    T = 10000
    tol = 1/np.sqrt(T)*5
    testfun = QPrintItem.assertL2

    def compute_answer_print(self):
        # print("In agent print code")
        from collections import Counter
        counts = Counter( [self.question.agent.pi(None, k) for k in range(self.T)] )
        distrib = [counts[k] / self.T for k in range(self.question.agent.env.k)]
        return np.asarray(distrib)

    def process_output(self, res, txt, numbers):
        return res

class BanditQuestion(QuestionGroup):
    title = "Simple bandits"
    class SimpleBanditItem(BanditItem):
        #title = "Value function estimate"
        def get_env_agent(self):
            from irlc.ex08.simple_agents import BasicAgent
            from irlc.ex08.bandits import StationaryBandit
            env = StationaryBandit(k=10, )
            agent = BasicAgent(env, epsilon=0.1)
            return env, agent
    class SimpleBanditActionDistribution(BanditItemActionDistribution):
        pass

class GradientBanditQuestion(QuestionGroup):
    title = "Gradient agent"
    class SimpleBanditItem(BanditItem):
        # title = "Simple agent question"
        def get_env_agent(self):
            from irlc.ex08.bandits import StationaryBandit
            from irlc.ex08.gradient_agent import GradientAgent
            env = StationaryBandit(k=10)
            agent = GradientAgent(env, alpha=0.05)
            return env, agent

        def precompute_payload(self):
            env, agent = self.get_env_agent()
            _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
            return trajectories, agent.H

        def compute_answer_print(self):
            trajectories, H = self.precomputed_payload()
            env, agent = self.get_env_agent()
            train_recording(env, agent, trajectories)
            self.H = H
            self.question.agent = agent
            return agent.H

        def test(self, computed, expected):
            self.testfun(computed, self.H)

    class SimpleBanditActionDistribution(BanditItemActionDistribution):
        pass

class UCBAgentQuestion(QuestionGroup):
    title = "UCB agent"
    class UCBAgentItem(BanditItem):
        def get_env_agent(self):
            from irlc.ex08.bandits import StationaryBandit
            from irlc.ex08.ucb_agent import UCBAgent
            env = StationaryBandit(k=10)
            agent = UCBAgent(env)
            return env, agent

    class UCBAgentActionDistribution(BanditItemActionDistribution):
        pass

class NonstatiotnaryAgentQuestion(QuestionGroup):
    title = "Nonstationary bandit environment"
    class NonstationaryItem(BanditItem):
        def get_env_agent(self):
            epsilon = 0.1
            from irlc.ex08.nonstationary import NonstationaryBandit, MovingAverageAgent
            bandit = NonstationaryBandit(k=10)
            agent = MovingAverageAgent(bandit, epsilon=epsilon, alpha=0.15)
            return bandit, agent

    class NonstationaryActionDistribution(BanditItemActionDistribution):
        pass

# Now, the ValueIteartionAgent group
class GridworldDPItem(QPrintItem):
    testfun = QPrintItem.assertL2
    title = "Small Gridworld"
    tol = 1e-3

    def get_value_function(self):
        from irlc.ex09.small_gridworld import SmallGridworldMDP
        from irlc.ex09.policy_evaluation import policy_evaluation
        env = SmallGridworldMDP()
        pi0 = {s: {a: 1 / len(env.A(s)) for a in env.A(s)} for s in env.nonterminal_states}
        V = policy_evaluation(pi0, env, gamma=.83)
        return V, env

    def compute_answer_print(self):
        V, env = self.get_value_function()
        return np.asarray( [V[s] for s in env.states] )

    def process_output(self, res, txt, numbers):
        return res


class DynamicalProgrammingGroup(QuestionGroup):
    title = "Dynamical Programming test"

    class PolicyEvaluationItem(GridworldDPItem):
        title = "Iterative Policy evaluation"

    class PolicyIterationItem(GridworldDPItem):
        title = "policy iteration"
        def get_value_function(self):
            from irlc.ex09.small_gridworld import SmallGridworldMDP, plot_value_function
            from irlc.ex09.policy_iteration import policy_iteration
            env = SmallGridworldMDP()
            pi, v = policy_iteration(env, gamma=0.91)
            return v, env
    class ValueIteartionItem(GridworldDPItem):
        title = "value iteration"

        def get_value_function(self):
            from irlc.ex09.value_iteration import value_iteration
            from irlc.ex09.small_gridworld import SmallGridworldMDP
            env = SmallGridworldMDP()
            policy, v = value_iteration(env, gamma=0.92, theta=1e-6)
            return v, env

class GamlerQuestion(QuestionGroup):
    title = "Gamblers problem"
    class GamlerItem(GridworldDPItem):
        title = "Value-function test"
        def get_value_function(self):
            # from irlc.ex09.small_gridworld import SmallGridworldMDP, plot_value_function
            # from irlc.ex09.policy_iteration import policy_iteration
            from irlc.ex09.value_iteration import value_iteration
            from irlc.ex09.gambler import GamblerEnv
            env = GamblerEnv()
            pi, v = value_iteration(env, gamma=0.91)
            return v, env

class JackQuestion(QuestionGroup):
    title ="Jacks car rental problem"
    class JackItem(GridworldDPItem):
        title = "Value function test"
        max_cars = 5
        tol = 0.01
        def get_value_function(self):
            from irlc.ex09.value_iteration import value_iteration
            from irlc.ex09.jacks_car_rental import JackRentalMDP
            env = JackRentalMDP(max_cars=self.max_cars, verbose=True)
            pi, V = value_iteration(env, gamma=.9, theta=1e-3, max_iters=1000, verbose=True)
            return V, env

class DPAgentRLQuestion(QuestionGroup):
    title = "Value-iteration agent test"
    class ValueAgentItem(GridworldDPItem):
        title = "Evaluation on Suttons small gridworld"
        tol = 1e-2
        def get_env(self):
            from irlc.gridworld.gridworld import BookGridEnvironment, SuttonCornerGridEnvironment
            return SuttonCornerGridEnvironment(living_reward=-1)

        def compute_answer_print(self):
            env = self.get_env()
            from irlc.ex09.value_iteration_agent import ValueIterationAgent
            agent = ValueIterationAgent(env, mdp=env.mdp)
            # env = VideoMonitor(env, agent=agent, agent_monitor_keys=('v',))
            stats, _ = train(env, agent, num_episodes=1000)
            return np.mean( [s['Accumulated Reward'] for s in stats])

        def process_output(self, res, txt, numbers):
            return res

    class BookItem(ValueAgentItem):
        title = "Evaluation on alternative gridworld (Bookgrid)"
        def get_env(self):
            from irlc.gridworld.gridworld import BookGridEnvironment
            return BookGridEnvironment(living_reward=-0.6)

class SnakesQuestion(QuestionGroup):
    title = "Snakes and ladders"
    class SnakesItem(GridworldDPItem):
        title = "Value function test"
        def get_value_function(self):
            from irlc.ex09.value_iteration import value_iteration
            from irlc.ex09.snakes import SnakesMDP
            mdp = SnakesMDP()
            pi, v = value_iteration(mdp, gamma=0.91)
            return v, mdp

## MC Agents
class QExperienceItem(QPrintItem):
    tol = 1e-6
    title = "Q-value test"
    testfun = QPrintItem.assertL2
    gamma = 0.8

    def get_env_agent(self):
        return None, None

    def precompute_payload(self):
        env, agent = self.get_env_agent()
        _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
        return trajectories, agent.Q.to_dict()

    def compute_answer_print(self):
        trajectories, Q = self.precomputed_payload()
        env, agent = self.get_env_agent()
        train_recording(env, agent, trajectories)
        self.Q = Q
        self.question.agent = agent
        return agent.Q.to_dict()

    def process_output(self, res, txt, numbers):
        return res

    def test(self, computed, expected):
        Qc = []
        Qe = []
        for s, Qs in expected.items():
            for a, Q in Qs.items():
                Qe.append(Q)
                Qc.append(computed[s][a])

        super().test(Qe, Qc)

class VExperienceItem(QPrintItem):
    tol = 1e-6
    title = "V-value test"
    testfun = QPrintItem.assertL2

    def get_env_agent(self):
        env=None
        agent = None
        raise Exception("Overwrite")
        return env, agent

    def precompute_payload(self):
        env, agent = self.get_env_agent()
        _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
        return trajectories, dict(agent.v)

    def compute_answer_print(self):
        trajectories, V = self.precomputed_payload()
        env, agent = self.get_env_agent()
        train_recording(env, agent, trajectories)
        self.V = V
        self.question.agent = agent
        return dict(agent.v)

    def process_output(self, res, txt, numbers):
        return res

    def test(self, computed, expected):
        Vc = []
        Ve = []
        for s, v in expected.items():
            Ve.append(v)
            Vc.append(computed[s])

        super().test(Ve, Vc)

class MCAgentQuestion(QuestionGroup):
    title = "Test of MC Agent"
    class EvaluateTabular(QExperienceItem):
        title = "Q-value test"
        def get_env_agent(self):
            from irlc.ex10.mc_agent import MCAgent
            import gym
            env = gym.make("SmallGridworld-v0")
            from gym.wrappers import TimeLimit
            env = TimeLimit(env, max_episode_steps=1000)
            gamma = .8
            agent = MCAgent(env, gamma=gamma, first_visit=True)
            return env, agent

class MCEvaluationQuestion(QuestionGroup):
    title = "Test of MC evaluation agent"
    class EvaluateTabular(VExperienceItem):
        title = "Value-function test"
        def get_env_agent(self):
            from irlc.ex10.mc_agent import MCAgent
            from irlc.ex10.mc_evaluate import MCEvaluationAgent
            import gym
            from irlc.ex09 import envs
            env = gym.make("SmallGridworld-v0")
            from gym.wrappers import TimeLimit
            env = TimeLimit(env, max_episode_steps=1000)
            gamma = .8
            agent = MCEvaluationAgent(env, gamma=gamma, first_visit=True)
            return env, agent

class BlackjackQuestion(QuestionGroup):
    title = "MC policy evaluation agent and Blacjack"
    class BlackjackItem(QPrintItem):
        tol = 2
        title = "Test of value function"
        test = QPrintItem.assertNorm
        def compute_answer_print(self):
            nenv = "Blackjack-v0"
            env = gym.make(nenv)
            episodes = 50000
            gamma = 1
            # experiment = f"experiments/{nenv}_first_{episodes}"
            # Instantiate the agent and call the training method here. Make sure to pass the policy=policy20 function to the MCEvaluationAgent
            # and set gamma=1.
            from irlc.ex10.mc_evaluate import MCEvaluationAgent
            from irlc.ex10.mc_evaluate_blackjack import get_by_ace, to_matrix, policy20

            agent = MCEvaluationAgent(env, policy=policy20, gamma=1)
            train(env, agent, num_episodes=episodes)
            from irlc.ex10.mc_agent import MCAgent
            # return agent.v[env.reset()]
            w = get_by_ace(agent.v, ace=True)
            X, Y, Z = to_matrix(w)
            return np.asarray(Z.flat)

        def process_output(self, res, txt, numbers):
            return res

class TD0Question(QuestionGroup):
    title = "Test of TD(0) evaluation agent"
    class EvaluateTabular(VExperienceItem):
        title = "Value-function test"
        gamma = 0.8
        def get_env_agent(self):
            from irlc.ex10.td0_evaluate import TD0ValueAgent
            env = gym.make("SmallGridworld-v0")
            from gym.wrappers import TimeLimit
            env = TimeLimit(env, max_episode_steps=1000)
            gamma = .8
            agent = TD0ValueAgent(env, gamma=self.gamma)
            return env, agent

class NStepSarseEvaluationQuestion(QuestionGroup):
    title = "Test of TD-n evaluation agent"
    class EvaluateTabular(VExperienceItem):
        title = "Value-function test"
        gamma = 0.8
        def get_env_agent(self):
            envn = "SmallGridworld-v0"
            from irlc.ex11.nstep_td_evaluate import TDnValueAgent
            env = gym.make(envn)
            agent = TDnValueAgent(env, gamma=self.gamma, n=5)
            return env, agent

class QAgentQuestion(QuestionGroup):
    title = "Test of Q Agent"
    class EvaluateTabular(QExperienceItem):
        title = "Q-value test"

        def get_env_agent(self):
            from irlc.ex11.q_agent import QAgent
            import gym
            env = gym.make("SmallGridworld-v0")
            # from gym.wrappers import TimeLimit
            # env = TimeLimit(env, max_episode_steps=1000)
            # gamma = .8
            agent = QAgent(env, gamma=self.gamma)
            return env, agent

class LinearWeightVectorTest(QPrintItem):
    tol = 1e-6
    title = "Weight-vector test"
    testfun = QPrintItem.assertL2
    gamma = 0.8
    num_episodes = 10
    alpha = 0.8

    def get_env(self):
        return gym.make("MountainCar500-v0")

    def get_env_agent(self):
        return None, None

    def precompute_payload(self):
        env, agent = self.get_env_agent()
        _, trajectories = train(env, agent, return_trajectory=True, num_episodes=self.num_episodes)
        return trajectories, agent.Q.w

    def compute_answer_print(self):
        trajectories, Q = self.precomputed_payload()
        env, agent = self.get_env_agent()
        train_recording(env, agent, trajectories)
        self.Q = Q
        self.question.agent = agent
        return agent.Q.w

    def process_output(self, res, txt, numbers):
        return res

class LinearValueFunctionTest(LinearWeightVectorTest):
    title = "Linear value-function test"
    def compute_answer_print(self):
        trajectories, Q = self.precomputed_payload()
        env, agent = self.get_env_agent()
        train_recording(env, agent, trajectories)
        self.Q = Q
        self.question.agent = agent
        vfun = [agent.Q[s,a] for s, a in zip(trajectories[0].state, trajectories[0].action)]
        return vfun

class LinearQAgentQuestion(QuestionGroup):
    title = "Test of Linear Q Agent"

    class LinearQAgentItem(LinearWeightVectorTest):
        def get_env_agent(self):
            env = self.get_env()
            alpha = 0.1
            from irlc.ex11.semi_grad_q import LinearSemiGradQAgent
            agent = LinearSemiGradQAgent(env, gamma=1, alpha=alpha, epsilon=0)
            return env, agent

class SarsaReturnTypeItem(QPrintItem):
    title = "Average return over many simulated episodes"
    # Test the return from Sarsa agent.
    gamma = 0.95
    epsilon = 0.2
    tol = 0.1

    def get_env(self):
        return gym.make("SmallGridworld-v0")

    def get_env_agent(self):
        return None, None

    def compute_answer_print(self):
        env, agent = self.get_env_agent()
        stats, _ = train(env, agent, num_episodes=5000)
        # self.question.stats = stats
        self.question.agent = agent
        self.question.env = env
        return stats

    def process_output(self, res, txt, numbers):
        return np.mean([s['Accumulated Reward'] for s in res])

class SarsaTypeQItem(QPrintItem):
    title = "Q-values (inexact test) based on simulated data"
    tol = 0.3
    def compute_answer_print(self):
        s = self.question.env.reset()
        actions, qs = self.question.agent.Q.get_Qs(s)
        return qs

    def process_output(self, res, txt, numbers):
        return res

class SarsaQuestion(QuestionGroup):
    title = "Test of Sarsa Agent"
    class SarsaReturnItem(SarsaReturnTypeItem):
        def get_env_agent(self):
            from irlc.ex11.sarsa_agent import SarsaAgent
            agent = SarsaAgent(self.get_env(), gamma=self.gamma)
            return agent.env, agent

    class SarsaQItem(SarsaTypeQItem):
        title = "Sarsa action distribution item"


class NStepSarsaQuestion(QuestionGroup):
    title = "N-step Sarsa"
    class SarsaReturnItem(SarsaReturnTypeItem):
        def get_env_agent(self):
            from irlc.ex11.nstep_sarsa_agent import SarsaNAgent
            agent = SarsaNAgent(self.get_env(), gamma=self.gamma, n=5)
            return agent.env, agent

    class SarsaQItem(SarsaTypeQItem):
        title = "Sarsa action distribution"

class LinearSarsaAgentQuestion(QuestionGroup):
    title = "Sarsa Agent with linear function approximators"

    class LinearExperienceItem(LinearWeightVectorTest):
        tol = 1e-6
        title = "Linear sarsa agent"
        # testfun = QPrintItem.assertL2
        # gamma = 0.8
        alpha = 0.1
        num_episodes = 150

        def get_env_agent(self):
            env = self.get_env()
            from irlc.ex11.semi_grad_sarsa import LinearSemiGradSarsa
            agent = LinearSemiGradSarsa(env, gamma=1, alpha=self.alpha, epsilon=0)
            return env, agent

class LinearSarsaNstepAgentQuestion(QuestionGroup):
    title = "Test of Linear n-step sarsa Agent"

    class LinearExperienceItem(LinearValueFunctionTest):
        tol = 2200
        title = "Value function for linear n-step sarsa agent"
        testfun = QPrintItem.assertNorm
        num_episodes = 150
        gamma = 1
        def get_env_agent(self):
            env = self.get_env()
            from irlc.ex12.semi_grad_nstep_sarsa import LinearSemiGradSarsaN
            from irlc.ex12.semi_grad_sarsa_lambda import alpha
            agent = LinearSemiGradSarsaN(env, gamma=self.gamma, alpha=alpha, epsilon=0)
            return env, agent

class LinearSarsaLambdaAgentQuestion(QuestionGroup):
    title = "Test of Linear sarsa(Lambda) Agent"
    class LinearExperienceItem(LinearValueFunctionTest):
        tol = 2200
        testfun = QPrintItem.assertNorm
        title = "Linear sarsa agent"
        num_episodes = 150
        gamma = 1
        def get_env_agent(self):
            env = self.get_env()
            from irlc.ex12.semi_grad_sarsa_lambda import LinearSemiGradSarsaLambda, alpha
            agent = LinearSemiGradSarsaLambda(env, gamma=self.gamma, alpha=alpha, epsilon=0)
            return env, agent

## WEEK 12:
class SarsaLambdaQuestion(QuestionGroup):
    title = "Sarsa(lambda)"
    class SarsaReturnItem(SarsaReturnTypeItem):
        def get_env_agent(self):
            from irlc.ex12.sarsa_lambda_agent import SarsaLambdaAgent
            agent = SarsaLambdaAgent(self.get_env(), gamma=self.gamma, lamb=0.7)
            return agent.env, agent

    class SarsaLambdaQItem(SarsaTypeQItem):
        title = "Sarsa(lambda) Q-value test"

## Week 13
class DoubleQQuestion(QuestionGroup):
    title = "Double Q learning"
    class DQReturnItem(SarsaReturnTypeItem):
        def get_env_agent(self):
            from irlc.ex13.tabular_double_q import TabularDoubleQ
            agent = TabularDoubleQ(self.get_env(), gamma=self.gamma)
            return agent.env, agent

    class DoubleQItem(SarsaTypeQItem):
        tol = 1
        def compute_answer_print(self):
            s = self.question.env.reset()
            actions, qs = self.question.agent.Q1.get_Qs(s)
            return qs
        title = "Double Q action distribution"

class DynaQQuestion(QuestionGroup):
    title = "Dyna Q learning"
    class DynaQReturnItem(SarsaReturnTypeItem):
        def get_env_agent(self):
            from irlc.ex13.dyna_q import DynaQ
            agent = DynaQ(self.get_env(), gamma=self.gamma)
            return agent.env, agent

    class DynaQItem(SarsaTypeQItem):
        title = "Dyna Q action distribution"

week8bandits = [
    (BanditQuestion, 10),
    (GradientBanditQuestion, 10),
    (UCBAgentQuestion, 5),
    (NonstatiotnaryAgentQuestion, 5)
                ]
week9dp = [ (DynamicalProgrammingGroup, 20),
        (GamlerQuestion, 10),
        (JackQuestion, 10),
        (DPAgentRLQuestion, 5),
        (SnakesQuestion, 5),]
week10mc = [(MCAgentQuestion, 10),
        (MCEvaluationQuestion, 10),
        (TD0Question, 10),
        (BlackjackQuestion,5),]
week11nstep = [
        (NStepSarseEvaluationQuestion, 10),
        (QAgentQuestion, 10),
        (LinearQAgentQuestion, 10),
        (LinearSarsaAgentQuestion, 10),
        (SarsaQuestion, 10),
        (NStepSarsaQuestion, 5),
        ]
week12linear = [
        (SarsaLambdaQuestion, 15),
        (LinearSarsaAgentQuestion, 15),
        (LinearSarsaLambdaAgentQuestion, 10),
        (LinearSarsaNstepAgentQuestion, 10),]
week13dqn = [(DoubleQQuestion, 10),
             (DynaQQuestion, 10)
             ]

class Assignment3RL(Report):
    title = "Reinforcement Learning"
    pack_imports = [irlc]
    """ Comment out some of the weeks to only run part of the test"""
    questions = week8bandits  + week9dp + week10mc + week11nstep + week12linear + week13dqn
    # questions = week12linear

if __name__ == '__main__':
    from unitgrade.unitgrade_helpers import evaluate_report_student
    # from unitgrade_private.hidden_create_files import setup_answers
    # setup_answers(Assignment3RL())
    evaluate_report_student(Assignment3RL(), ignore_missing_file=True)
