#!/bin/bash
policy_prefix=$1
envs=("HandManipulateBlock-v0" "HandManipulatePen-v0")
policies=("$policy_prefix/HandManipulateBlock/policy_best.pkl" "$policy_prefix/HandManipulatePen/policy_best.pkl")
policy_interpolation_routine="experiment.utils.merging_routine"

python dagger.py --envs ${envs[*]} --experts ${policies[*]} --policy_interpolation_routine $policy_interpolation_routine


#--envs "HandManipulateBlock-v0" "HandManipulatePen-v0" --experts "/home/bing/git/robo_arms/results/HandManipulateBlock/policy_best.pkl" "/home/bing/git/robo_arms/results/HandManipulatePen/policy_best.pkl" --policy_interpolation_routine experiment.utils.merging_routine


<class 'list'>: [<tf.Variable 'dummy/main/Q/_0/kernel:0' shape=(90, 256) dtype=float32_ref>, <tf.Variable 'dummy/main/Q/_0/bias:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'dummy/main/Q/_1/kernel:0' shape=(256, 256) dtype=float32_ref>, <tf.Variable 'dummy/main/Q/_1/bias:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'dummy/main/Q/_2/kernel:0' shape=(256, 256) dtype=float32_ref>, <tf.Variable 'dummy/main/Q/_2/bias:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'dummy/main/Q/_3/kernel:0' shape=(256, 1) dtype=float32_ref>, <tf.Variable 'dummy/main/Q/_3/bias:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'dummy/main/pi/_0/kernel:0' shape=(70, 256) dtype=float32_ref>, <tf.Variable 'dummy/main/pi/_0/bias:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'dummy/main/pi/_1/kernel:0' shape=(256, 256) dtype=float32_ref>, <tf.Variable 'dummy/main/pi/_1/bias:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'dummy/main/pi/_2/kernel:0' shape=(256, 256) dtype=float32_ref>, <tf.Variable 'dummy/main/pi/_2/bias:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'dummy/main/pi/_3/kernel:0' shape=(256, 20) dtype=float32_ref>, <tf.Variable 'dummy/main/pi/_3/bias:0' shape=(20,) dtype=float32_ref>]


main/pi/_0/kernel:0 -> input matrix
