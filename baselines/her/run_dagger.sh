#!/bin/sh

python dagger.py --envs "HandManipulateBlock-v0" "HandManipulatePen-v0" --experts "/home/bing/git/baselines/baselines/her/distillation/policies/block" "/home/bing/git/baselines/baselines/her/distillation/policies/pen" --policy_interpolation_routine distillation.experiment.utils.merging_routine --student "/home/bing/git/baselines/baselines/her/distillation/policies/distilled" 
