import os
import sys

import click
import numpy as np
import json
from mpi4py import MPI

import tensorflow as tf

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork

from subprocess import CalledProcessError


def load_stats(epoch_log_path):
    # loads epoch counter and highest_success rate
    epoch = 0
    success_rate = -1
    try:
        with open(epoch_log_path) as file:
            line = (file.readlines()[-1]).split(' ')
            epoch = int(line[0])
            success_rate = float(line[1])
            return epoch, success_rate
    except FileNotFoundError as e:
        logger.info(e)
        logger.info('creating epoch counter file')
        try:
            with open(epoch_log_path, 'w+') as file:
                file.write('0 -1\n')
                return epoch, success_rate
        except Exception as e:
            logger.info(e)
            logger.info('epoch counting file error, continuing without')
            return epoch, success_rate


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(env, policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies=True, policy_path=None, **kwargs):

    saver = tf.train.Saver()
    model_name='model.ckpt'
    if policy_path is not None:
        saver.restore(policy.sess, policy_path+model_name)
        logger.info("Successfully restored policy from {}".format(policy_path))
    if policy_path is None:
        policy_path = ''.join(['./trained/', env, '_', policy.scope, '/'])
        if not os.path.exists(policy_path):
            os.makedirs(policy_path)
    epoch_log_path = logger.get_dir()+'/epoch.txt'
    trained_epochs, best_success_rate = load_stats(epoch_log_path)

    policy_path += model_name

    rank = MPI.COMM_WORLD.Get_rank()

    logger.info("Training...")
    for epoch in range(trained_epochs+1, n_epochs):
        # train
        logger.info("Episode {}/{}".format(epoch, n_epochs))
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, policy_path))
            pth = saver.save(policy.sess, policy_path)
            try:
                with open(epoch_log_path, 'a') as file:
                    file.write(str(epoch)+' '+str(best_success_rate)+'\n')
            except Exception as e:
                logger.info(e)
            print("saved policy to {}".format(pth))

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(
    env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, policy_path, with_forces, plot_forces,
    scope=None, override_params={}, save_policies=True
):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if logdir is None:
        logdir='./logs/'
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['with_forces'] = with_forces
    params['plot_forces'] = plot_forces
    params['env_name'] = env
    params['replay_strategy'] = replay_strategy
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)
    dims['o'] = dims['o'] + 2
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return, scope=scope)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'with_forces': with_forces,
        'plot_forces': plot_forces,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'with_forces': with_forces,
        'plot_forces': plot_forces,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    train(
        env=env, logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies,
        policy_path=policy_path)


@click.command()
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default=None, help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--with_forces', type=bool, default=False)
@click.option('--plot_forces', type=bool, default=False)
@click.option('--policy_path', type=str, default=None, help='path to policy to be loaded and trained')
@click.option('--scope', type=str, default=None, help='name of scope for tf')
def main(**kwargs):
    launch(**kwargs)


if __name__ == '__main__':
    main()
