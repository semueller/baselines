import argparse
import os
import sys

import tensorflow as tf
import numpy as np
import pickle
import gym
import gym.spaces
from mpi4py import MPI

from baselines.her.util import mpi_fork
from baselines.common.mpi_moments import mpi_moments
from baselines import logger
from baselines.her.ddpg import DDPG
from baselines.her.experiment import config
from rollout import RolloutWorker
from replay_buffer import ReplayBuffer

from helper import getfunc, print_kernels
from baselines.her.experiment.train import mpi_average
from subprocess import CalledProcessError

# print("original codebook")
codebook = {'HandManipulateBlock-v0': [1, 0],
            'HandManipulatePen-v0': [0, 1]
            }


def simple_rollout_worker(env_id='', codebook=None, params=None):
    # copied the whole setup from train.py
    params['with_forces'] = False
    params['plot_forces'] = False
    params['env_name'] = env_id
    params['replay_strategy'] = None
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)
    dims['o'] = dims['o'] + len(codebook)  # +2 for one hot encoding

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'with_forces': False,
        'plot_forces': False,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]

    return RolloutWorker(params['make_env'], policy=None, dims=dims, logger=logger, codebook=codebook, **rollout_params)


def blank_ddpg(dims, params, reuse=False, use_mpi=True, clip_return=True, scope=None, env_names=None):
    logger.info("creating ddpg agent with env: {} on scope {}".format(env_names, scope))
    sample_her_transitions = config.configure_her(params)
    # sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    # gamma = params['gamma']
    gamma = 0.05
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = params['ddpg_params']

    input_dims = dims.copy()

    # DDPG agent
    # env = cached_make_env(params['make_env'])
    # env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': None,
                        'sample_transitions': sample_her_transitions,
                        'gamma': gamma,
                        'scope': scope,
                        })
    ddpg_params['info'] = {
        'env_name': env_names,
    }
    policy = DDPG(reuse=reuse, **ddpg_params, use_mpi=use_mpi)

    return policy


def load_policy(path, sess, var_list=None):
    '''
    :param path: path to policy
    :param sess: session the data will be loaded in
    :param var_list: variables of session that shall be looked for in checkpoint
    :return:
    '''
    saver = tf.train.Saver(var_list=var_list)
    model_name = '/model.ckpt'
    logger.info('loading policy from {}'.format(path))
    saver.restore(sess, path+model_name)
    logger.info("Successfully restored policy from {}".format(path))


def run(args):
    num_cpu = args.num_cpu
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

    num_epochs = args.num_epochs
    env_ids = args.envs
    num_extra_inputs = len(env_ids)

    rollout_workers, experts = {}, {}

    sess = tf.Session()

    with sess.as_default():
        assert tf.get_default_session() is sess
        # load expert policies
        for env_id, expert in zip(env_ids, args.experts):
            cfg = config.DEFAULT_PARAMS.copy()
            worker = simple_rollout_worker(env_id, codebook, cfg)
            rollout_workers[env_id] = worker
            scope = expert.split('/')[-1]
            observation_size = worker.dims['o']
            action_size = worker.dims['u']  # env.action_space.shape[0]  # should be 20
            dims = {'u': action_size,  # see config.configure_dims
                    'o': observation_size,
                    'g': 7  # ? goal size?
                    }
            cfg['replay_strategy'] = None
            cfg['T'] = worker.envs[0]._max_episode_steps
            policy = blank_ddpg(dims=dims, params=cfg, env_names=env_id, scope=scope)
            vars_to_load = [n for n in tf.global_variables() if scope in n.name]
            load_policy(expert, sess, var_list=vars_to_load)  # this hopefully initializes the previously constructed
            experts[env_id] = policy

        # config and load student
        cfg = config.DEFAULT_PARAMS.copy()
        cfg['replay_strategy'] = None
        cfg['T'] = experts[env_ids[0]].T  # T IS TIME HORIZON FOR TRAINING?
        cfg['env_name'] = env_ids
        cfg = config.prepare_params(cfg)
        scope_student = args.student.split('/')[-1]
        student = blank_ddpg(dims=dims, params=cfg, scope=scope_student, env_names=env_ids)
        student._env_specific_buffers = {k:
                                            ReplayBuffer(
                                                            student.buffer.buffer_shapes,
                                                            student.buffer.size_in_transitions,
                                                            student.buffer.T,
                                                            student.buffer.sample_transitions
                                                        )
                                            for k in env_ids
                                        }
        b = student.buffer  # remove student original buffer
        student.buffer = None
        del b
        vars_to_load = [n for n in tf.global_variables() if scope_student in n.name]
        load_policy(path=args.student, sess=sess, var_list=vars_to_load )
        cfg = config.DEFAULT_PARAMS.copy()
        cfg['replay_strategy'] = None
        cfg['T'] = experts[env_ids[0]].T  # T IS TIME HORIZON FOR TRAINING?
        cfg['env_name'] = env_ids
        cfg = config.prepare_params(cfg)
        # build a dummy for temporary storage
        dummy = blank_ddpg(dims=dims, params=cfg, scope='dummy', env_names=env_ids)

        # still necessary?
        # prep_input_injection(student, num_extra_inputs=num_extra_inputs) # call global_variables_initializer here

        # get policy interpolation routine
        pir = getfunc(args.policy_interpolation_routine)

        best_performance = [-1]*len(env_ids)

        # repeat
        with open('training_data.pkl', 'a+b') as training_data_log:
            for epoch in range(num_epochs):
                beta = float(epoch+1) / float(num_epochs)
                logger.info('epoch: {}\nbeta: {}'.format(epoch, beta))
                # generate observations
                for id in env_ids:
                    rollout_worker = rollout_workers[id]
                    expert = experts[id]

                    # get dummy policy to be used for this iteration
                    pir(beta=beta, temp=dummy, expert=expert, student=student)
                    logger.info("generate observations in {}".format(id))
                    rollout_worker.policy = dummy
                    episode = rollout_worker.generate_rollouts()

                    #  select the env specific buffer to store stuff in
                    student.buffer = student._env_specific_buffers[id]
                    student.store_episode(episode)

                    # training_data_log.write(str(episode)+'\n')
                    pickle.dump(episode, training_data_log)  # thread safety?

                    n_batches = config.DEFAULT_PARAMS['n_batches']  # since we are single threaded lets increase this?
                    logger.info("starting training in epoch {}".format(epoch))
                    # replace critic network of student with actor network of expert
                    sess.run([tf.assign(s, t) for t, s in zip(
                                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=expert.scope + '/main'),
                                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=student.scope + '/target')
                                    )])
                    #  train on env with new buffer
                    for b in range(n_batches):
                        logger.info("   training batch {}".format(b))
                        student.train_distillation(sess, expert)

                    rollout_worker.clear_history()

                n_test_rollouts = config.DEFAULT_PARAMS['n_test_rollouts']
                current_success_rate = [-1]*len(env_ids)

                # test
                for i, id in enumerate(env_ids):
                    rollout_worker = rollout_workers[id]
                    rollout_worker.policy = student
                    for _ in range(n_test_rollouts):
                        rollout_worker.generate_rollouts()
                    # current_success_rate[i] = rollout_worker.current_success_rate()
                    current_success_rate[i] = mpi_average(rollout_worker.current_success_rate())
                # save better policy
                if rank == 0 and max(current_success_rate) >= max(best_performance) and False:
                    best_performance = [sr if sr >= bp else bp for sr, bp in zip(current_success_rate, best_performance)]
                    path = args.student + '/' + '_'.join(['%.2f' % b for b in best_performance])+'/'
                    logger.info('saving student to {}'.format(best_performance))
                    if not os.path.exists(path):
                        os.makedirs(path)
                    vars_to_save = [v for v in tf.global_variables() if student.scope in v.name]
                    saver = tf.train.Saver(var_list=vars_to_save)
                    saver.save(sess=sess, save_path=path+'model.ckpt')
                    rollout_worker.save_policy(path+'policy.pkl')  # for easy use with play.py
                    with open(path+'epoch.txt', 'a') as file:
                        file.write(str(epoch)+'\n')  # logs in which epoch(s) the policy was saved in path

                local_uniform = np.random.uniform(size=(1,))
                root_uniform = local_uniform.copy()
                MPI.COMM_WORLD.Bcast(root_uniform, root=0)
                if rank != 0:
                    assert local_uniform[0] != root_uniform[0]


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--envs', nargs='+', required=True)
    parser.add_argument('--experts', nargs='+', required=True)
    parser.add_argument('--iterations', type=float, default=0.9, help='number of episodes; if in i in (0,1) repeated '
                                                                      'until i*performance_of_expert reached')
    parser.add_argument('--num_trajectories', type=int, default=5, help='number of rollouts per episode per '
                                                                        'environment')
    parser.add_argument('--num_cpu', type=int, default=1, help='number of mpi threads that will be created')
    parser.add_argument('--max_trajectories', type=int, default=50, help='number of trajectories per environment')
    parser.add_argument('--beta_start', type=float, default=1.0)
    parser.add_argument('--beta_end', type=float, default=0.0)
    parser.add_argument('--beta_delta', type=float, default=0.0001)#anneals beta to zero after 10k steps
    parser.add_argument('--policy_interpolation_routine', type=str, help='args module.functionname ; '
                                                                         'function to generate new policy')
    parser.add_argument('--student', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--temperature', type=float, default=1)
    args = parser.parse_args()

    print("ARGS: {}".format(args))

    return args


if __name__ == '__main__':
    args = parse_args()

    run(args)
