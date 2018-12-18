from collections import deque

import numpy as np
import pickle, os, datetime, subprocess
from mujoco_py import MujocoException

import matplotlib
from matplotlib import pyplot as plt
from baselines.her.util import convert_episode_to_batch_major, store_args
from PIL import Image

import gym
import gym.spaces

class RolloutWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, codebook=None, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.dims = dims
        self.codebook = codebook
        self.dirparams = {
            'env_name': kwargs['env_name'],
            'policy_number': kwargs['policy_number']
                         }
        if self.codebook is not None:
            print("Codebook used for the environments: {}".format(codebook))
        if callable(make_env):
            self.envs = [make_env() for _ in range(rollout_batch_size)]
        elif isinstance(make_env, list): # list?
            self.envs = []
            if callable(make_env[0]): # list of factory functions?
                for factory in make_env:
                    self.envs += [factory() for _ in range(rollout_batch_size)]
            if isinstance(make_env[0], str): # list of envnames as strings?
                for envname in make_env:
                    self.envs += [gym.make(envname) for _ in range(rollout_batch_size)]
        else:
            raise Exception("could not initialize envs")


        assert self.T > 0

        self.wf = kwargs['with_forces']

        if self.wf:
            self.dims['o'] = 77
            for e in self.envs:
                e.env.use_forces()

        self.plot_forces = kwargs['plot_forces']

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals

        self.reset_all_rollouts()
        self.clear_history()

        if self.plot_forces:
            self.NUM_SENSORS = 16
            self.NUM_PLOT_LOOKBACK = 101

            self.xdata = [i for i in range(self.NUM_PLOT_LOOKBACK)]
            self.forcedata = [[0.0 for _ in range(self.NUM_PLOT_LOOKBACK)] for _ in range(self.NUM_SENSORS)]

            self.fig, self.ax = plt.subplots(1, 1)

            plt.show(False)
            plt.draw()

            axes = plt.gca()
            axes.set_xlim([0, 100])
            axes.set_ylim([-0.1, 15])

            # cache the background
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

            self.lines = [self.ax.plot(self.xdata, self.forcedata[i])[0] for i in range(self.NUM_SENSORS)]

        self.render_and_save_png = True  # ndrw
        self.render = False


    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        obs['observation'] = np.concatenate([obs['observation'], [1.0, 0.0]])
        if self.wf:
            self.initial_o[i] = np.append(obs['observation'], np.ndarray([0 for _ in range(16)]))
        else:
            self.initial_o[i] = obs['observation']

        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self, n=0):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        directory = './png/'#+datetime.datetime.now().strftime("%m%d_%H%M%S") + os.sep
        directory += '{}'.format(self.dirparams['policy_number'])+'/'+self.dirparams['env_name']+'/'

        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)

            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_env = self.envs[i]
                    # print(curr_env.name)
                    curr_o_new, _, _, info = curr_env.step(u[i])
                    if self.codebook is not None:
                        one_hot_code = self.codebook[curr_env.name]
                        curr_o_new['observation'] = np.concatenate([curr_o_new['observation'], one_hot_code])
                        # print(curr_o_new['observation'][-2:])

                    if self.plot_forces:
                        for j in range(self.NUM_SENSORS):
                            l = self.forcedata[j][1:].copy()
                            l.append(self.envs[i].env.sim.data.sensordata[-(j+1)])
                            self.forcedata[j] = l

                            self.lines[j].set_data(self.xdata, self.forcedata[j])

                        self.fig.canvas.draw()
                        self.fig.canvas.flush_events()

                        plt.pause(0.000000000001)

                    if 'is_success' in info:
                        success[i] = info['is_success']
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                    elif self.render_and_save_png:  # ndrw
                        rgb_array = self.envs[i].render(mode='rgb_array')
                        im = Image.fromarray(rgb_array)
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        im.save(directory + "pic_"+str(n)+"{0:05d}.png".format(t))
                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new

        if self.render_and_save_png:  # ndrw - makes video out of the saved .png files
            cmd = "cd " + directory + "; ffmpeg -y -framerate 30 -pattern_type glob -i '*.png' -vf 'crop=1366:748:840:200' -c:v libx264 -pix_fmt yuv420p _rollout.mp4"
            #  ffmpeg -framerate 30 -pattern_type glob -i '*.png' -vf 'crop=1280:1280:600:600' -c:v libx264 -pix_fmt yuv420p _rollout.mov
            subprocess.call(cmd, shell=True)

        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size


        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)
