import numpy as np
from utils import defaultdict2
import inspect
import itertools
import os
import sys
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from utils import load_time_series, log_time_series, existing_runs, cache_read, cache_write
import shutil


# Base Agent Class:
class Agent:
    """ Main agent class. See (Her21, Subsection 1.4.3) for additional details.  """

    def __init__(self, env):
        self.env = env

    def pi(self, s, k=None):
        """ Compute the policy pi_k(s).
        For discrete application (dynamical programming/search and reinforcement learning), k is discrete k=0, 1, 2, ...
        For control applications, k is continious and denote simulation time t.

        :param s: Current state
        :param k: Current time index.
        :return: action
        """
        return self.env.action_space.sample()

    def train(self, s, a, r, sp, done=False):
        """
        Called at each step of the simulation after a = pi(s,k) and environment transition to sp.
        Allows the agent to learn from experience

        :param s: Current state x_k
        :param a: Action taken
        :param r: Reward obtained by taking action a_k in x_k
        :param sp: State environment transitioned to x_{k+1}
        :param done: Whether environment terminated when transitioning to sp
        :return: None
        """
        pass

    def __str__(self):
        """ Optional: A unique name for this agent. Used for labels when plotting, but can be kept like this. """
        return super().__str__()

    def extra_stats(self):
        """ Optional: Can be used to record extra information from the Agent while training.
        You can safely ignore this method. """
        return {}


class TabularQ:
    """
    Tabular Q-values. This is a helper class for the Q-agent to store Q-values without too much hassle with
    state-dependent action spaces and so on.
    """

    def __init__(self, env):
        # This may need to be changed (s in P)
        qfun = lambda s: OrderedDict({a: 0 for a in (env.P[s] if hasattr(env, 'P') else range(env.action_space.n))})
        self.q_ = defaultdict2(lambda s: qfun(s))
        self.env = env

    def get_Qs(self, state):
        (actions, Qa) = zip(*self.q_[state].items())
        return tuple(actions), tuple(Qa)

    def get_optimal_action(self, state):
        actions, Qa = self.get_Qs(state)
        a_ = np.argmax(np.asarray(Qa) + np.random.rand(len(Qa)) * 1e-8)
        return actions[a_]

    def __getitem__(self, state_comma_action):
        s, a = state_comma_action
        return self.q_[s][a]

    def __setitem__(self, state_comma_action, q_value):
        s, a = state_comma_action
        self.q_[s][a] = q_value

    def items(self):  # not sure this is used
        raise Exception()
        return self.q_.items()

    def to_dict(self):
        # Convert to a regular dictionary
        d = {s: {a: Q for a, Q in Qs.items()} for s, Qs in self.q_.items()}
        return d


class TabularAgent(Agent):
    """
    The self.Q variable is a custom datastructure to save the Q(s,a)-values.
    There are multiple ways to implement the Q-values, most of which will get us in troubles
    if we want to implement the examples in Sutton.

    I solve this using the TabularQ class above. What it amounts to is that you can access Q-values as

    >>> q_value = self.Q[s,a]

    and set them as

    >>> self.Q[s,a] = new_q_value

    It also provides helpful methods. For instance if Q[s,a1] = q1, Q[s,a2] = q2, ...
    then

    >>> actions, qs = self.Q.Qs(s)

    defines actions=[a1,a2,...] and qs=[q1,q2,...]
    """

    def __init__(self, env, gamma=0.99, epsilon=0):
        super().__init__(env)
        self.gamma, self.epsilon = gamma, epsilon
        self.Q = TabularQ(env)

    def pi(self, s, k=None):
        return self.random_pi(s)

    def pi_eps(self, s):
        """ Implement epsilon-greedy exploration. Return random action with probability self.epsilon,
        else be greedy wrt. the Q-values. """
        return self.random_pi(s) if np.random.rand() < self.epsilon else self.Q.get_optimal_action(s)

    def random_pi(self, s):
        """ Generates a random action given s.

        It might seem strange why this is useful, however many policies requires us to to random exploration, and it is
        possible to get the method wrong.
        We will implement the method depending on whether self.env defines an MDP or just contains an action space.
        """
        if hasattr(self.env, 'P'):
            return np.random.choice(list(self.env.P[s].keys()))
        else:
            return self.env.action_space.sample()


class ValueAgent(TabularAgent):
    """
    This is a simple wrapper class around the Agent class above. It fixes the policy and is therefore useful for doing
    value estimation.
    """

    def __init__(self, env, gamma=0.95, policy=None, v_init_fun=None):
        self.env = env
        self.policy = policy  # policy to evaluate
        """ Value estimates. 
        Initially v[s] = 0 unless v_init_fun is given in which case v[s] = v_init_fun(s). """
        self.v = defaultdict2(float if v_init_fun is None else v_init_fun)
        super().__init__(env, gamma=gamma)

    def pi(self, s, k=None):
        return self.random_pi(s) if self.policy is None else self.policy(s)

    def value(self, s):
        return self.v[s]


# class QAgent(TabularAgent):
#     """
#     Implement the Q-learning agent here. Note that the Q-datastructure already exist
#     (see agent class for more information)
#     """
#     def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1):
#         self.alpha = alpha
#         super().__init__(env, gamma, epsilon)
#
#     def pi(self, s, k=None):
#         """
#         Return current action using epsilon-greedy exploration. Look at the TabularAgent class
#         for ideas.
#         """
#         return self.pi_eps()
#         # raise NotImplementedError("Implement function body")
#
#     def train(self, s, a, r, sp, done=False):
#         """
#         Implement the Q-learning update rule, i.e. compute a* from the Q-values.
#         As a hint, note that self.Q[sp][a] corresponds to q(s_{t+1}, a) and
#         that what you need to update is self.Q[s][a] = ...
#         """
#
#         #raise NotImplementedError("Implement function body")
#
#     def __str__(self):
#         return f"QLearner_{self.gamma}_{self.epsilon}_{self.alpha}"




#################
class QAgent(TabularAgent):
    """
    Implement the Q-learning agent (SB18, Section 6.5)
    Note that the Q-datastructure already exist, as do helper functions useful to compute an epsilon-greedy policy
    (see TabularAgent class for more information)
    """
    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1):
        self.alpha = alpha
        super().__init__(env, gamma, epsilon)

    def pi(self, s, k=None):
        """
        Return current action using epsilon-greedy exploration. Look at the TabularAgent class
        for ideas.
        """
        return self.pi_eps()
        raise NotImplementedError("Implement function body")

    def train(self, s, a, r, sp, done=False):
        """
        Implement the Q-learning update rule, i.e. compute a* from the Q-values.
        As a hint, note that self.Q[sp,a] corresponds to q(s_{t+1}, a) and
        that what you need to update is self.Q[s, a] = ...

        You may want to look at self.Q.get_optimal_action(state) to compute a = argmax_a Q[s,a].
        """
        astar = self.Q.get_optimal_action(sp)
        maxQ = self.Q[sp][astar]
        self.Q[s][a] = self.Q[s][a] + self.alpha * (r + self.gamma * maxQ - self.Q[s][a])
        # raise NotImplementedError("Implement function body")

    def __str__(self):
        return f"QLearner_{self.gamma}_{self.epsilon}_{self.alpha}"





def train(env, agent, experiment_name=None, num_episodes=1, verbose=True, reset=True, max_steps=1e10,
          max_runs=None,
          return_trajectory=False, # Return the current trajectories as a list
          resume_stats=None, # Resume stat collection from last save. None implies same as saveload_model
          log_interval=1, # Only log every log_interval steps. Reduces size of log files.
          delete_old_experiments=False,
          ):
    """
    Implement the main training loop, see (Her21, Subsection 1.4.4).
    Simulate the interaction between agent `agent` and the environment `env`.
    The function has a lot of special functionality, so it is useful to consider the common cases. An old:

    >>> stats, _ = train(env, agent, num_episodes=2)

    Simulate interaction for two episodes (i.e. environment terminates two times and is reset).
    `stats` will be a list of length two containing information from each run

    >>> stats, trajectories = train(env, agent, num_episodes=2, return_Trajectory=True)

    `trajectories` will be a list of length two containing information from the two trajectories.

    >>> stats, _ = train(env, agent, experiment_name='experiments/my_run', num_episodes=2)

    Save `stats`, and trajectories, to a file which can easily be loaded/plotted (see course software for examples of this).
    The file will be time-stamped so using several calls you can repeat the same experiment (run) many times.

    >>> stats, _ = train(env, agent, experiment_name='experiments/my_run', num_episodes=2, max_runs=10)

    As above, but do not perform more than 10 runs. Useful for repeated experiments.

    :param env: Environment (Gym instance)
    :param agent: Agent instance
    :param experiment_name: Save outcome to file for easy plotting (Optional)
    :param num_episodes: Number of episodes to simulate
    :param verbose: Display progress bar
    :param reset: Call env.reset() before simulation start.
    :param max_steps: Terminate if this many steps have elapsed (for non-terminating environments)
    :param max_runs: Maximum number of repeated experiments (requires `experiment_name`)
    :param return_trajectory: Return trajectories list (Off by default since it might consume lots of memory)
    :param resume_stats: Resume stat collection from last run (requires `experiment_name`)
    :param log_interval: Log stats less frequently
    :return: stats, trajectories (both as lists)
    """
    saveload_model = False
    temporal_policy = None
    save_stats = True

    if delete_old_experiments and experiment_name is not None and os.path.isdir(experiment_name):
        shutil.rmtree(experiment_name)

    if experiment_name is not None and max_runs is not None and existing_runs(experiment_name) >= max_runs:
        stats, recent = load_time_series(experiment_name=experiment_name)
        if return_trajectory:
            trajectories = cache_read(recent+"/trajectories.pkl")
        else:
            trajectories = []
        return stats, trajectories
    stats = []
    steps = 0
    ep_start = 0
    resume_stats = saveload_model if resume_stats is None else resume_stats

    recent = None
    if resume_stats:
        stats, recent = load_time_series(experiment_name=experiment_name)
        if recent is not None:
            ep_start, steps = stats[-1]['Episode']+1, stats[-1]['Steps']
    if temporal_policy is None:
        a = inspect.getfullargspec(agent.pi)
        temporal_policy = len(a.args) >= 3  # does the policy include a time step?

    trajectories = []
    include_metadata = len(inspect.getfullargspec(agent.train).args) >= 7

    with tqdm(total=num_episodes, disable=not verbose, file=sys.stdout) as tq:
        for i_episode in range(num_episodes):
            if reset or i_episode > 0:
                s = env.reset()
            elif hasattr(env, "s"):
                s = env.s
            elif hasattr(env, 'state'):
                s = env.state
            else:
                s = env.model.s
            time = 0
            reward = []
            trajectory = Trajectory(time=[], state=[], action=[], reward=[], env_info=[])
            for _ in itertools.count():
                a = agent.pi(s,time) if temporal_policy else agent.pi(s)
                sp, r, done, metadata = env.step(a)
                if not include_metadata:
                    agent.train(s, a, r, sp, done)
                else:
                    agent.train(s, a, r, sp, done, metadata)

                if return_trajectory:
                    trajectory.time.append(np.asarray(time))
                    trajectory.state.append(s)
                    trajectory.action.append(a)
                    trajectory.reward.append(np.asarray(r))
                    trajectory.env_info.append(metadata)

                reward.append(r)
                steps += 1

                time += metadata['dt'] if 'dt' in metadata else 1
                if done or steps >= max_steps:
                    trajectory.state.append(sp)
                    trajectory.time.append(np.asarray(time))
                    break
                s = sp
            if return_trajectory:
                trajectory = Trajectory(**{field: np.stack([np.asarray(x_) for x_ in getattr(trajectory, field)]) for field in fields}, env_info=trajectory.env_info)
                trajectories.append(trajectory)
            stats.append({"Episode": i_episode + ep_start,
                          "Accumulated Reward": sum(reward),
                          "Average Reward": np.mean(reward),
                          "Length": len(reward),
                          "Steps": steps, **agent.extra_stats()}) if (i_episode+1) % log_interval == 0 else None
            tq.set_postfix(ordered_dict=OrderedDict(list(OrderedDict(stats[-1]).items() )[:5] )) if len(stats) > 0 else None
            tq.update()
    sys.stderr.flush()

    if resume_stats and save_stats and recent is not None:
        os.remove(recent+"/log.txt")

    if experiment_name is not None and save_stats:
        path = log_time_series(experiment=experiment_name, list_obs=stats)
        if return_trajectory:
            cache_write(trajectories, path+"/trajectories.pkl")

        print(f"Training completed. Logging {experiment_name}: '{', '.join( stats[0].keys()) }'")

    for i, t in enumerate(trajectories):
        from collections import defaultdict
        nt = defaultdict(lambda: [])
        if "supersample" in t.env_info[0]:
            for f in fields:
                for k, ei in enumerate(t.env_info):
                    z = ei['supersample'].__getattribute__(f).T
                    if k == 0:
                        pass
                    else:
                        z = z[1:]
                    nt[f].append(z)

            # traj = Trajectory(time=np.concatenate(nt['time']), state=
            for f in fields:
                nt[f] = np.concatenate([z for z in nt[f]],axis=0)
                # nt[f] =  np.concatenate( [ei['supersample'].__getattribute__(f) for ei in t.env_info] , axis=0)
            traj2 = Trajectory(**nt, env_info=[])
            trajectories[i] = traj2 # .__setattr__(f, nt[f])

    return stats, trajectories

