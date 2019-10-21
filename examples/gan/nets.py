#!/usr/bin/env python

"""nets.py: Generator and Discriminator networks are defined here."""

import tensorflow as tf
import tensorflow.contrib.layers as tcl

class G_mlp(object):

    def __init__(self, output_dim, batch_norm):
        self.name = 'G_mlp'
        self.output_dim = output_dim
        self.batch_norm = batch_norm

    def dense_batch_relu(self, x, is_training, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            h = tf.contrib.layers.fully_connected(x, 500,
                                                  activation_fn = None,
                                                  scope='dense')
            if self.batch_norm == True:
                h = tf.contrib.layers.batch_norm(h,
                                                 decay=0.9,
                                                 scale=True,
                                                 updates_collections = None,
                                                 is_training = is_training,
                                                 scope = 'bn')
        return tf.nn.leaky_relu(h)

    def __call__(self, z, is_training):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            g = self.dense_batch_relu(z, is_training=is_training, scope='layer_1')
            g = self.dense_batch_relu(g, is_training=is_training, scope='layer_2')
            g = tcl.fully_connected(g, self.output_dim, activation_fn=None)
        return g

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]

class D_mlp(object):
    def __init__(self):
        self.name = "D_mlp"

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            d = x
            for _ in range(3):
                d = tcl.fully_connected(d, 400, activation_fn=tf.nn.relu)
            logit = tcl.fully_connected(d, 1, activation_fn=None)
        return logit

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]
