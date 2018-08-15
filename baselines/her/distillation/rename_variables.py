import tensorflow as tf

import pickle

'''
to rename scope in pickle file, uncomment 
        #self.scope = input("scope? ")
in baselines.her.ddpg.__init__ and enter new scope name
'''

path = '/home/bing/git/robo_arms/results/HandManipulatePen/policy_best.pkl'

with open(path, 'r+b') as f:
    x = pickle.load(f)

with open(path,'wb') as f:
    pickle.dump(x, f)
