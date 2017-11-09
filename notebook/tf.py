'''
Created on Apr 27, 2017

@author: optas
'''

import tensorflow as tf


def reset_tf_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
