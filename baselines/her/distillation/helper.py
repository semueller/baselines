import tensorflow as tf

def getfunc(str):
    str = str.split('.')
    mod = __import__('.'.join(str[:-1]), globals(), locals(), [str[:-1]])
    return getattr(mod, str[-1])


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