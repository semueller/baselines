#!/bin/sh
python train.py --env HandManipulateBlock-v0 --scope policy1 --n_epochs 10000 --logdir ./logs/HandReach-v0_policy1 --num_cpu 2



