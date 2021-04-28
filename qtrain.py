import inspect
import itertools
import os
import sys
from collections import OrderedDict, namedtuple
import numpy as np
from tqdm import tqdm
from irlc.utils.common import load_time_series, log_time_series
from irlc.utils.irlc_plot import existing_runs
import shutil

fields = ('time', 'state', 'action', 'reward')
Trajectory = namedtuple('Trajectory', fields + ("env_info",))

def qtrain(env, agent, experiment_name=None, num_episodes=1, verbose=True, reset=True, max_steps=1e10,
          max_runs=None,
          return_trajectory=False, # Return the current trajectories as a list
          resume_stats=None, # Resume stat collection from last save. None implies same as saveload_model
          log_interval=1, # Only log every log_interval steps. Reduces size of log files.
          delete_old_experiments=False,
          return_agent=False,
          decay_epsilon=(False, None, None)):
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
    :param decay_epsilon: Decay epsilon over time
    :return: stats, trajectories (both as lists)
    """
    from irlc import cache_write
    from irlc import cache_read
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

    # Epsilon decay value
    decay_eps, stop_ratio, every = decay_epsilon
    if decay_eps:
        stop_episode = num_episodes // stop_ratio
        decay_value = agent.epsilon / stop_episode * every

    with tqdm(total=num_episodes, disable=not verbose, file=sys.stdout) as tq:
        for i_episode in range(num_episodes):
            if not i_episode % every:
                agent.epsilon = max(agent.epsilon - decay_value, 0)
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
                try:
                    trajectory = Trajectory(**{field: np.stack([np.asarray(x_) for x_ in getattr(trajectory, field)]) for field in fields}, env_info=trajectory.env_info)
                except Exception as e:
                    pass

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

    if return_agent:
        return stats, trajectories, agent
    return stats, trajectories
