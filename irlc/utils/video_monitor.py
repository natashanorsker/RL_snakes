"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import os, shutil
from gym.wrappers import Monitor
import warnings
import time
from PIL import Image



import matplotlib.pyplot as plt

class VideoMonitor(Monitor):
    def __init__(self, env2, *args, fps=30, continious_recording=True, video_file=None, agent=None,
                 agent_monitor_keys=('Q', ), render_kwargs={},
                 directory=None, force=True, snapshot_base=None, frame_snapshot_callable = None, **kwargs):
        k = 45
        if snapshot_base is not None or frame_snapshot_callable is not None:
            if frame_snapshot_callable is None:
                frame_snapshot_callable = lambda episode, frame: frame == frame == 0 and episode == 0
            # print(frame_snapshot_callable)
            if isinstance(frame_snapshot_callable, int):
                k = frame_snapshot_callable  # required; if frame_snapshot_callable is used below code does not work
                frame_snapshot_callable = lambda episode, frame: (frame == k and episode == 0)

        if frame_snapshot_callable is not None and snapshot_base is None:
            snapshot_base = "pdf/snapshot.pdf"
        self.frame_snapshot_callable = frame_snapshot_callable
        self.snapshot_base = snapshot_base

        self.agent = agent
        if directory is None: # TODO: Use randomly named directory.
            directory = "monitor_recording"

        env2.metadata['video.frames_per_second'] = fps
        self.video_file = video_file
        self.continious_recording = continious_recording
        self.monitor_keys = agent_monitor_keys
        self.steps = 0
        self.episodes = 0
        self.fps = fps
        if agent is not None:
            def _render(*args, **kwargs):
                return type(env2).render(env2, *args, **self.compute_render_args(), **render_kwargs, **kwargs)
            env2.render = _render

            def _train(*args, **kwargs):
                out = type(agent).train(agent, *args, **kwargs)
                self._after_step(*self.op)
                return out

            agent.train = _train

        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # I don't give a shit the environment is not created by gym.
            super().__init__(env2, directory, *args, force=force, **kwargs)
        self.env2_ = env2

    def step(self, action):
        self._before_step(action)
        observation, reward, done, info = self.env.step(action)
        self.op = (observation, reward, done, info)
        if self.agent is None: # this is required for the PacMan environment with agent=None
            done = self._after_step(observation, reward, done, info) # removed this to include agent train.
        time.sleep(1/ self.fps)
        return observation, reward, done, info

    def compute_render_args(self):
        return {k: getattr(self.agent, k) for k in self.monitor_keys}

    def reset_video_recorder(self):
        self.try_snap_frame(called_after_step=True)
        if self.video_recorder and self.continious_recording:
            self.video_recorder.capture_frame()
        else:
            super().reset_video_recorder()

    def close(self):
        # self.env2_.close()
        super().close()
        if self.video_file is not None:
            dn = os.path.dirname(self.video_file)
            if len(dn) > 0 and not os.path.exists(dn):
                os.makedirs(dn)

            if self.video_file.endswith(".mp4"):
                self.video_file = self.video_file[:-4]

            for vd, (v,json) in enumerate(self.videos):
                num = v[-10:-4]
                if self.continious_recording:
                    vout = self.video_file + ".mp4"
                    if vd > 0:
                        raise Exception("very strange stuff...")
                else:
                    vout = self.video_file + num + ".mp4"
                if os.path.exists(vout):
                    os.remove(vout)
                print("Writing video", os.path.abspath(vout))
                shutil.copy(v, vout)
                from irlc import is_this_my_computer
                if is_this_my_computer():
                    from irlc import get_students_base


                    sp = os.path.join(get_students_base(),"movies")
                    if not os.path.isdir(sp):
                        os.mkdir(sp)
                    sp = os.path.join(sp, os.path.basename(vout))
                    shutil.copy(v, sp)
        # self.env2_.close()

    def plot(self):
        frame = self.render(mode="rgb_array")
        im = Image.fromarray(frame)
        plt.imshow(im)
        plt.axis('off')

    def savepdf(self, file):
        frame = self.render(mode="rgb_array")
        im = Image.fromarray(frame)
        snapshot_base = file
        if snapshot_base.endswith(".png"):
            sf = snapshot_base[:-4]
            fext = 'png'
        else:
            fext = 'pdf'
            if snapshot_base.endswith(".pdf"):
                sf = snapshot_base[:-4]
            else:
                sf = snapshot_base

        sf = f"{sf}.{fext}"
        dn = os.path.dirname(sf)
        if len(dn) > 0 and not os.path.isdir(dn):
            os.makedirs(dn)
        print("Saving snapshot of environment to", os.path.abspath(sf))
        if fext == 'png':
            im.save(sf)
            from irlc import _move_to_output_directory
            _move_to_output_directory(sf)
        else:
            plt.imshow(im)
            plt.axis('off')
            from irlc import savepdf
            savepdf(sf)

    def try_snap_frame(self, called_after_step=True):
        if self.snapshot_base is not None and self.frame_snapshot_callable is not None:
            episode = max(0, self.episode_id - 1)
            step = self.stats_recorder.steps
            # print(episode, step, self.frame_snapshot_callable, self.frame_snapshot_callable(episode, step))
            if self.frame_snapshot_callable(episode, step):
                frame = self.render(mode="rgb_array")
                im = Image.fromarray(frame)
                if self.snapshot_base.endswith(".png"):
                    sf = self.snapshot_base[:-4]
                    fext = 'png'
                else:
                    fext = 'pdf'
                    if self.snapshot_base.endswith(".pdf"):
                        sf = self.snapshot_base[:-4]
                    else:
                        sf = self.snapshot_base

                sf = f"{sf}_{episode:04d}_{step:04d}.{fext}"
                dn = os.path.dirname(sf)
                if len(dn) > 0 and not os.path.isdir(dn):
                    os.makedirs(dn)
                print("Saving snapshot of environment to", os.path.abspath(sf))
                # print("fext", fext)
                if fext == 'png':
                    im.save(sf)
                    from irlc import _move_to_output_directory
                    _move_to_output_directory(sf)
                else:
                    plt.imshow(im)
                    plt.axis('off')
                    from irlc import savepdf
                    savepdf(sf)


    def _after_step(self, observation, reward, done, info):
        "Overwrite this function to allow capture of frame snapshots. Note it should be nearly identical to original version"
        if not self.enabled: return done

        if done and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
            self.reset_video_recorder()
            self.episode_id += 1
            self._flush()

        # Record stats
        self.stats_recorder.after_step(observation, reward, done, info)
        # Record video
        self.video_recorder.capture_frame()
        self.try_snap_frame()
        return done
""" 

def main():
    from irlc.ex11.sarsa_agent import SarsaAgent
    from irlc.ex01.agent import train
    from irlc.berkley.rl.berkley_gridworld_environment import BerkleyBookGridEnvironment
    env = BerkleyBookGridEnvironment(adaptor='gym')
    agent = SarsaAgent(env, gamma=0.95, alpha=0.3)
    experiment = "experiments/berksarsa"

    env = VideoMonitor(env, agent=agent, video_file="videos/SarsaGridworld.mp4", fps=30, continious_recording=True, label="SADSF",
                       agent_monitor_keys=("Q",))
    # env.reset()
    train(env, agent, experiment_name=experiment, num_episodes=100)
    env.close()
    from irlc import main_plot
    main_plot(experiment)
    import matplotlib.pyplot as plt
    plt.show()

if __name__ == "__main__":
    main()

"""
