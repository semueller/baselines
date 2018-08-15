import argparse
import numpy as np
import pickle
import os

from baselines import logger
from baselines.common import set_global_seeds
from baselines.her.ddpg import DDPG
from baselines.her.experiment import config

import tensorflow as tf
from tensorflow.contrib import graph_editor
import gym
import gym.spaces

import numpy as np

from helper import getfunc, print_kernels

def configure_student(dims, params, reuse=False, use_mpi=True, clip_return=True, scope=None, env_names=None):
    #sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    # gamma = params['gamma']
    gamma = 0.05
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = params

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
                        'sample_transitions': None,
                        'gamma': gamma,
                        'scope': scope,
                        })
    ddpg_params['info'] = {
        'env_name': env_names,
    }
    policy = DDPG(reuse=reuse, **ddpg_params, use_mpi=use_mpi)

    return policy


def prep_input_injection(network, num_extra_inputs=0):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=network.scope+'/main')
    vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=network.scope+'/target')
    sess = tf.get_default_session()
    eye_dim = int(vars[0][0].shape[0] - num_extra_inputs)
    # e is an identity matrix with the last num_extra_inputs cols and rows being zero
    e = tf.Variable(tf.pad(tf.eye(eye_dim),
                            tf.constant([[0, num_extra_inputs], [0, num_extra_inputs]])
                           )
                    )

    sess.run(tf.global_variables_initializer())
    # print(sess.run(e))
    # print_kernels(network)
    for v in vars:
        if '_0/kernel' not in v.name:
            continue
        # zero values from last num_extra_inputs cols in v with e v = v*e
        v = tf.assign(v, tf.matmul(v, e))
        sess.run(v)
        pass
    # print_kernels(network)


def run(args):
    num_epochs = args.num_epochs
    env_ids = args.envs
    num_extra_inputs = len(env_ids)

    envs, experts = {}, {}
    encoding = np.eye(num_extra_inputs) # one hot encoding label vectors
    aggregated_dataset = []

    with tf.Session() as sess:

        # load policies
        for env_id, expert in zip(env_ids, args.experts):
            print("init env: {}, with expert: {}".format(env_id, expert))
            with open(expert, 'rb') as e:
                policy = pickle.load(e)
                experts[env_id] = policy
            envs[env_id] = gym.make(env_id)

        # add encoding parameter to environment
        for idx, env_id in enumerate(env_ids):
            envs[env_id].encoding = encoding[idx] #

        # config for student/ dummy
        observation_size = envs[env_ids[0]].observation_space.spaces['observation'].shape
        observation_size = observation_size[0] + num_extra_inputs# for one hot encoding
        action_size = envs[env_ids[0]].action_space.shape[0]  # should be 20
        dims = { 'u': action_size, # see config.configure_dims
                 'o': observation_size,
                 'g': 7 # ? goal size?
                 }
        cfg = config.DEFAULT_PARAMS
        cfg['replay_strategy'] = None
        cfg['T'] = experts[env_ids[0]].T ## T IS TIME HORIZON FOR TRAINING?
        student = configure_student(dims=dims, params=cfg, scope='distilled', env_names=env_ids)
        dummy = configure_student(dims=dims, params=cfg, scope='dummy', env_names=env_ids)
        # sess.run(tf.global_variables_initializer())
        prep_input_injection(student, num_extra_inputs=num_extra_inputs) # call global_variables_initializer here
        # print_kernels(student)

        # get policy interpolation routine
        pir = getfunc(args.policy_interpolation_routine)
        # test pir
        # for expert in experts.values():
        #     print("merging with expert {}".format(expert.scope))
        #     pir(1000, dummy, expert, student)
        #     print_kernels(dummy)

        # repeat
        for epoch in range(num_epochs):
            beta = float(epoch+1) / float(num_epochs)
            for id in env_ids:
                env = envs[id]
                expert = experts[id]

                # get dummy policy to be used for this iteration
                pir(beta=beta, temp=dummy, expert=expert, student=student)

                # run env with dummy
                state_actions = rollout_dummy(env, dummy, max_steps=500, temperature=args.temperature)

                # get action from expert from every state visited by dummy

                #append to dataset

                aggregated_dataset.append(state_actions)

        #   train
        #   LOSS: expert.q_val(state) - student.q_val(state)
        #   evaluate
        #   save

    pass


def rollout_dummy(env, dummy, max_steps, temperature=1):
    sess = tf.get_default_session()
    state_action = []
    state = env.reset()
    o, g, ag = state['observation'], state['desired_goal'], state['achieved_goal']
    o_extended = np.concatenate([o, env.encoding]) # add one hot encoding
    prestates, poststates, actions, rewards = [], [], [], []
    for t in range(max_steps):
        # env.render()
        policy_output = dummy.get_actions(
            o_extended, ag, g,
            compute_Q=True
        )

        u, Q = policy_output
        # actions.append(action)
        # prestates.append(state)
        state, reward, terminal, _ = env.step(u)

        o, g, ag = state['observation'], state['desired_goal'], state['achieved_goal']
        o_extended = np.concatenate([o, env.encoding])
        # poststates.append(state)
        # rewards.append(rewards)

        if terminal:
            print("return")
            break

    return state_action
    # merge prestates/ poststates/ actions/ rewards


def evaluate(policy, envs):
    pass


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--envs', nargs='+', required=True)
    parser.add_argument('--experts', nargs='+', required=True)
    parser.add_argument('--iterations', type=float, default=0.9, help='number of episodes; if in i in (0,1) repeated until '
                                                                    'i*performance_of_expert reached')
    parser.add_argument('--num_trajectories', type=int, default=5, help='number of rollouts per episode per environment')
    parser.add_argument('--max_trajectories', type=int, default=50, help='number of trajectories per environment')
    parser.add_argument('--beta_start', type=float, default=1.0)
    parser.add_argument('--beta_end', type=float, default=0.0)
    parser.add_argument('--beta_delta', type=float, default=0.0001)#anneals beta to zero after 10k steps
    parser.add_argument('--policy_interpolation_routine', type=str, help='args module.functionname ; function to generate new policy')
    parser.add_argument('--student', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--temperature', type=float, default=1)
    args = parser.parse_args()

    print("ARGS: {}".format(args))

    return args

if __name__ == '__main__':
    args = parse_args()
    run(args)
