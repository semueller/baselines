#!/bin/sh
python train_multi_env.py --env HandManipulateBlock-v0 --env HandManipulatePen-v0 --policy_path ./trained/multi_policy1/ --scope policy1 --n_epochs 10000 --logdir ./logs/multi_policy1 --num_cpu 2



