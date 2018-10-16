#!/bin/bash

python dagger.py --envs HandManipulateBlock-v0 HandManipulatePen-v0 --experts ./policies/block ./policies/pen --student ./policies/distilled --policy_interpolation_routine experiment.utils.merging_routine
