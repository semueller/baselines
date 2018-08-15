#!/bin/sh
python train.py --env HandReach-v0 --scope test --n_epochs 42 --logdir ./logs/HandReach-v0_test --policy_path ./trained/HandReach-v0_test/

