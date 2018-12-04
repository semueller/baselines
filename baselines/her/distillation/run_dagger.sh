#!/bin/bash

#python dagger.py --envs HandManipulateBlock-v0 HandManipulatePen-v0 --experts ./policies/block ./policies/pen --student ./policies/distilled --policy_interpolation_routine experiment.utils.merging_routine
python distillation/dagger.py --envs "HandManipulateBlock-v0" "HandManipulatePen-v0" --experts "/home/bing/git/baselines/baselines/her/distillation/policies/block/policy.pkl" "/home/bing/git/baselines/baselines/her/distillation/policies/pen/policy.pkl" --policy_interpolation_routine experiment.utils.merging_routine
