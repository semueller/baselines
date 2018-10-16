# from rabintang https://pythonexample.com/user/rabintang

import sys, getopt

import tensorflow as tf

def rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                new_name = new_name.replace(replace_from, replace_to)
            if add_prefix:
                new_name = add_prefix + new_name

            if new_name == var_name:
                continue

            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                print('Renaming %s to %s.' % (var_name, new_name))
                # Rename the variable
                var = tf.Variable(var, name=new_name)

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, './'+replace_to)


def main(argv):
    checkpoint_dir = './distillation/policies/distilled'
    replace_from = 'policy1'
    replace_to = 'distilled'
    add_prefix = None
    dry_run = False

    rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run)


if __name__ == '__main__':
    main(sys.argv[1:])
