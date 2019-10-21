import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
from ops import *
from tqdm import tqdm, trange
from critic_architectures import joint_critic, separate_critic

class mi_regularizer(object):

    def __init__(self, args, regu_name):
        self.regu_name = regu_name
        self.critic_lr = args.critic_lr
        self.ema_decay = args.ema_decay
        self.batch_size = args.batch_size
        self.critic_activation = args.critic_activation
        self.critic_layers = args.critic_layers 
        self.ema_decay = args.ema_decay
        self.critic = eval('{}_critic(args)'.format(args.critic_type))

    def nwj(self, x, z):

        T_joint, T_product = self.critic(x, z)     
        mi = tf.reduce_mean(T_joint) - 1 / np.e * tf.reduce_mean(tf.exp(T_product))
        mi_for_grads = mi
        
        return mi, mi_for_grads

    def mine(self, x, z):

        T_joint, T_product = self.critic(x, z)

        E_joint = tf.reduce_mean(T_joint)
        E_product = tf.log(1 / self.batch_size) + tf.reduce_logsumexp(T_product)
        mi =  E_joint - E_product

        ema_denominator = tf.Variable(tf.exp(tf.reduce_logsumexp(T_product)))
        ema_denominator -= (1 - self.ema_decay) * (ema_denominator - tf.exp(tf.reduce_logsumexp(T_product)))
        mi_for_grads = E_joint - 1/tf.stop_gradient(ema_denominator)*tf.exp(tf.reduce_logsumexp(T_product))

        return mi, mi_for_grads

    def nce(self, x, z):
        #circular_permut = tfp.bijectors.Permute(permutation=[self.batch_size - 1] + [i for i in range(self.batch_size - 1)], axis=-2)  # transform [1,2,3] into [3,1,2]
        #z_permut = z
        #contatenated_permuted_z = z
        #for _ in range(self.batch_size):
        #    z_permut = circular_permut.forward(z_permut)
        #    contatenated_permuted_z = tf.concatenate([contatenated_permuted_z,z_permut],axis=0)
        #x_tiled= tf.tile(x,[self.batch_size,1])
        #scores_vec = eval("self.{}_critic(x_tiled, contatenated_permuted_z)".format(self.critic_type))
        #scores_matrix = tf.reshape(scores_vec,[self.batch_size, self.batch_size])
        T_joint, T_product = self.critic(x, z)

        E_joint = tf.reduce_mean(T_joint)
        E_product = tf.log(1 / self.batch_size) + tf.reduce_mean(tf.reduce_logsumexp(tf.add(T_joint, T_product), axis=1))
        
        mi = E_joint - E_product
        mi_for_grads = mi

        return mi, mi_for_grads

    @property
    def vars(self):
        vars = {}
        vars['critic'] = [var for var in tf.global_variables() if 'critic' in var.name]
        return vars


    def __call__(self, x, z, optimizer=None):

        train_ops = {}
        quantities = {}

        dim_x = x.get_shape().as_list()[1:]
        dim_z = z.get_shape().as_list()[1:]

        mi, mi_for_grads = eval("self.{}(x,z)".format(self.regu_name))

        quantities['mi'] = mi
        quantities['mi_for_grads'] = mi_for_grads
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        train_ops['critic'] = optimizer.minimize(- mi_for_grads, var_list=self.vars['critic'])
        
        return train_ops, quantities

    def get_info(self, sess, feed_dict, quantities, step):
        info = {}
        info['step'] = "Step=" + str(step)
        values = sess.run(list(quantities.values()), feed_dict=feed_dict)
        for name, value in zip(quantities.keys(), values):
            if name not in info:
                info[name] = name + "={:.3g}".format(value)
        return info

    def fit(self, x_data, z_data, batch_size, epochs, eval_size=-1):

        tf.reset_default_graph()
        x_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size] + list(x_data.shape[1:]))
        z_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size] + list(z_data.shape[1:]))

        train_ops, quantities = self(x_ph, z_ph)

        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            n_batchs = int(x_data.shape[0]/batch_size)
            epoch_bar = trange(epochs)
            for epoch in epoch_bar:
                batch_bar = trange(n_batchs)
                for i in batch_bar:
                    x_batch = x_data[i*batch_size:(i+1)*batch_size]
                    z_batch = z_data[i*batch_size:(i+1)*batch_size]
                    feed_dict = {x_ph:x_batch, z_ph:z_batch}
                    if epoch+i == 0:
                        _ = sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
                    _ = sess.run(train_ops['critic'], feed_dict=feed_dict) 
                    info = self.get_info(sess, feed_dict, quantities, i)
                    batch_bar.set_description('   '.join(list(info.values())))
            if eval_size == -1:
                eval_size = x_data.shape[0]
            x_eval = x_data[:eval_size]
            z_eval = z_data[:eval_size]
            eval_feed_dict = {x_ph:x_eval, z_ph:z_eval}
            mi_eval = sess.run(quantities['mi'], feed_dict=feed_dict)
        return mi_eval







