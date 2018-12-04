import click
import numpy as np
import pickle

from baselines import logger
from baselines.common import set_global_seeds
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker


@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=10)
@click.option('--render', type=int, default=1)
@click.option('--with_forces', type=bool, default=False)
@click.option('--env', type=str, default=None)
def main(policy_file, seed, n_test_rollouts, render, with_forces, env):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']
    env_name = 'HandManipulateBlock-v0'

    # one hot encoding
    # codebook as in trianing:
    print("original codebook")
    codebook = {'HandManipulateBlock-v0': [1, 0],
                'HandManipulatePen-v0': [0, 1]
                }
    # flipped codebook
    # print("flipped codebook")
    # codebook = {'HandManipulateBlock-v0': [0, 1],
    #             'HandManipulatePen-v0': [1, 0]
    #             }

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['scope'] = policy.scope
    params['with_forces'] = with_forces
    params['plot_forces'] = False
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    # params['env_name'] = 'HandManipulateBlock-v0'
    params['env_name'] = env_name
    if env is not None:
        params['env_name'] = env
    print(params['env_name'])
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)
    dims['o'] = dims['o'] + len(codebook)  # add dims for one hot encoding

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': True,
        'rollout_batch_size': 1,
        'with_forces': with_forces,
        'plot_forces': False,
        'render': bool(0),
    }

    for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]
    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, codebook=codebook, **eval_params)
    evaluator.seed(seed)

    # Run evaluation.
    evaluator.clear_history()
    for i in range(n_test_rollouts):
        evaluator.generate_rollouts(i)

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main()
