#!/bin/sh
python train.py --env HandManipulatePen-v0 --policy_path ../save/trained/HandManipulatePen-v0_policy1/ --scope policy1 --n_epochs 10000 --logdir ../save/logs/HandManipulatePen-v0_policy1/ --num_cpu 2



