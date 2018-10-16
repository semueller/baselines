import tensorflow as tf

import numpy as np


def print_kernels(network):
    print('PRINTNG {}'.format(network.scope))
    sess = tf.get_default_session()
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=network.scope+'/main')
    vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=network.scope+'/target')
    for v in vars:
        if '_0/kernel' not in v.name:
            continue
        print(v.name)
        print(sess.run(v))
    print('#### DONE ####')


def get_hidden_units(network):
    # return bias and kernel variabels from /main /target scope
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=network.scope+'/main')+tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=network.scope+'/target')


def pad_with(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector


def merging_routine(beta, temp, expert, student):
    sess = tf.get_default_session()

    # move everything from student to temp
    sess.run([
        tf.assign(t, s) for t, s in zip(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=temp.scope),
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=student.scope))
    ])

    exp_vars = get_hidden_units(expert)
    std_vars = get_hidden_units(student)  # sanity check: student and temp variables should hold the same values
    tmp_vars = get_hidden_units(temp)
    # print_kernels(temp)
    # interpolate hidden units of main and target scopes
    sess.run([
        tf.assign(t,  # assign value
                          # (1-beta)*student + beta*expert
                          tf.add(tf.scalar_mul(1.-beta, s),
                                 tf.scalar_mul(beta,
                                               # pad expert matrix to match shape of student
                                               # ie append columns with zeros
                                               tf.pad(e, tf.constant([[0, int(s.shape[0]-e.shape[0])], [0, 0]]))
                                               )
                                 )
                          ) if '_0/kernel' in e.name
        else
            tf.assign(t, tf.add(tf.scalar_mul(1.-beta, s), tf.scalar_mul(beta, e)))
        for t, e, s in zip(tmp_vars, exp_vars, std_vars)
    ])
