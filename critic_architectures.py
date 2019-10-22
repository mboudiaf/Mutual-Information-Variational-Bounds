#!/usr/bin/env python

"""Critic_architectures.py: All the architecture of critic networks used."""

__author__ = "Malik Boudiaf"
__version__ = "0.1"
__maintainer__ = "Malik Boudiaf"
__email__ = "malik-abdelkrim.boudiaf.1@ens.etsmtl.ca"
__status__ = "Development"

import tensorflow as tf
import numpy as np
from ops import conv2d

class joint_critic(object):
    def __init__(self, args):
        self.dim_x = args.dim_x
        self.dim_z = args.dim_z
        self.critic_activation = eval("tf.nn.{}".format(args.critic_activation))
        self.critic_layers = args.critic_layers

        if len(self.dim_x) == 1 and len(self.dim_z) == 1:
            self.critic_archi = "fc"
        elif len(self.dim_x) > 1 and len(self.dim_z) > 1:
            self.critic_archi = "conv"
        else:
            self.critic_archi = "semi_conv"

    def __call__(self, x, z):
        #circular_permut = tfp.bijectors.Permute(permutation=[self.batch_size - 1] + [i for i in range(self.batch_size - 1)], axis=-2)  # transform [1,2,3] into [3,1,2]
        #z_permut = z
        #contatenated_permuted_z = z
        #for _ in range(self.batch_size):
        #    z_permut = circular_permut.forward(z_permut)
        #    contatenated_permuted_z = tf.concatenate([contatenated_permuted_z,z_permut],axis=0)
        #x_tiled= tf.tile(x,[self.batch_size,1])
        #scores_vec = eval("self.{}_critic(x_tiled, contatenated_permuted_z)".format(self.critic_type))
        #scores_matrix = tf.reshape(scores_vec,[self.batch_size, self.batch_size])

        z_shuffle = tf.gather(z, tf.random.shuffle(tf.range(tf.shape(z)[0]))) # Workaround to make shuffle operation differentiable in the graph
        T_joint = eval("self.{}(x, z)".format(self.critic_archi))
        T_product = eval("self.{}(x, z_shuffle)".format(self.critic_archi))

        return T_joint, T_product

    def semi_conv(self, x, z):
        df_dim = 16
        if len(self.dim_x) > 1:
            a = x
            b = z
        else:
            a = z
            b = x
        with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
            a0 = conv2d(a, df_dim, name='conv_2D_1')
            a0_dim = a0.get_shape().as_list()[1:]
            b0_flatten_dim = np.prod(a0_dim)
            b0_flat = tf.layers.dense(b, b0_flatten_dim, scope='tf.layers.dense_1')
            b0 = tf.reshape(b0_flat, [-1] + a0_dim)
            h0 = self.critic_activation(tf.add(a0, b0))

            a1 = conv2d(h0, 2 * df_dim, name='conv_2D_2')
            a1_dim = a1.get_shape().as_list()[1:]
            a1_flatten_dim = np.prod(a1_dim)
            b1_flat = tf.layers.dense(b0_flat, a1_flatten_dim, scope='tf.layers.dense_2')
            b1 = tf.reshape(b1_flat, [-1] + a1_dim)
            h1 = self.critic_activation(tf.add(a1, b1))

            a2 = conv2d(h1, 4 * df_dim, name='conv_2D_3')
            a2_dim = a2.get_shape().as_list()[1:]
            a2_flatten_dim = np.prod(a2_dim)
            b2_flat = tf.layers.dense(b1_flat, a2_flatten_dim, scope='tf.layers.dense_3')
            b2 = tf.reshape(b2_flat, [-1] + a2_dim)
            h2 = self.critic_activation(tf.add(a2, b2))

            h3 = tf.layers.dense(tf.layers.flatten(h2), 512, scope='tf.layers.dense_4')
            h4 = tf.layers.dense(h3, 1, scope='tf.layers.dense_5')
        return h4

    def conv(self, x, z):
        df_dim = 16
        k_h = 5
        k_w = 5
        d_h = 2
        d_w = 2
        stddev = 0.02
        dim_x = x.shape[1]
        dim_z = z.shape[1]
        if dim_x > dim_z: # a always has the widest dimension
            a = x
            b = z
        else:
            a = z
            b = x

        with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
            a0 = conv2d(a, output_dim=df_dim, k_h=k_h + tf.abs(dim_x - dim_z), k_w=k_w + tf.abs(dim_x - dim_z),
                        d_h=d_h, d_w=d_w, stddev=stddev, name='conv_2D_a')
            b0 = conv2d(b, output_dim=df_dim, k_h=k_h, k_w=k_w, d_h=d_w, d_w=d_h, stddev=stddev, name='conv_2D_b')
            assert a0.get_shape() == b0.get_shape()
            h0 = self.critic_activation(tf.add(a0, b0))

            h1 = tf.nn.elu(conv2d(h0, output_dim=2 * df_dim, k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev, name='conv_2D_2'))
            h2 = tf.nn.elu(conv2d(h1, output_dim=4 * df_dim, k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev, name='conv_2D_3'))
            h3 = tf.layers.dense(tf.layers.flatten(h2), 1024, scope='tf.layers.dense_4')
            h4 = tf.layers.dense(h3, 1, scope='tf.layers.dense_5')

        return h4

    def fc(self, x, z):
        init_weigths = tf.glorot_normal_initializer()
        x = tf.contrib.layers.flatten(x)
        z = tf.contrib.layers.flatten(z)
        xz = tf.concat([x, z], axis=1)
        h = xz
        with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
            for i, unit in enumerate(self.critic_layers):
                h = tf.layers.dense(h, unit, activation=self.critic_activation, name="fc_{}".format(i), kernel_initializer=init_weigths)  #, bias_initializer=init_bias)
            output = tf.layers.dense(h, 1, activation=None)
        return output


# ==========================================================================================================================================================================
# ==========================================================================================================================================================================


class separate_critic(object):

    def __init__(self, args):
        self.dim_x = args.dim_x
        self.dim_z = args.dim_z
        self.batch_size = args.batch_size
        self.critic_activation = args.critic_activation
        self.critic_layers = args.critic_layers

        if len(self.dim_x) == 1:
            self.x_critic_archi = "fc"
        else:
            self.x_critic_archi = "conv"

        if len(self.dim_z) == 1:
            self.z_critic_archi = "fc"
        else:
            self.z_critic_archi = "conv"
        self.num_output_units = 10

    def __call__(self, x, z):

        T_x = eval("self.{}(x, 'X_network')".format(self.x_critic_archi))
        T_z = eval("self.{}(z, 'Z_network')".format(self.z_critic_archi))
        score_matrix = tf.matmul(T_x, tf.transpose(T_z))
        mask = tf.eye(self.batch_size)
        complem_mask = 1 - mask

        T_joint = tf.multiply(score_matrix, mask)
        T_product = tf.multiply(score_matrix, complem_mask)
        return T_joint, T_product

    def conv(self, x, scope):
        df_dim = 16
        with tf.variable_scope('critic_{}'.format(scope), reuse=tf.AUTO_REUSE) as vs:
            x0 = conv2d(x, df_dim, name='conv_2D_1')
            h0 = tf.nn.elu(x0)

            a1 = conv2d(h0, 2 * df_dim, name='conv_2D_2')
            h1 = tf.nn.elu(a1)

            a2 = conv2d(h1, 4 * df_dim, name='conv_2D_3')
            h2 = tf.nn.elu(a2)

            h3 = tf.layers.dense(tf.layers.flatten(h2), 1024, scope='tf.layers.dense_1')
            h4 = tf.layers.dense(tf.layers.flatten(h3), self.num_output_units, scope='tf.layers.dense_2')
        return h4

    def fc(self, x, scope):
        init_weigths = tf.glorot_normal_initializer()
        x = tf.contrib.layers.flatten(x)
        h = x
        with tf.variable_scope('critic_{}'.format(scope), reuse=tf.AUTO_REUSE):
            for i, unit in enumerate(self.critic_layers):
                h = tf.layers.dense(h, unit, activation=self.critic_activation, name="fc_{}".format(i), kernel_initializer=init_weigths)  #, bias_initializer=init_bias)
            output = tf.layers.dense(h, self.num_output_units, activation=None)
        return output
