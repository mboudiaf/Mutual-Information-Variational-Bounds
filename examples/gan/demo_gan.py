#!/usr/bin/env python

"""demo_gan.py: Showcase of how to use mutual information as a regularization term."""

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import csv
import argparse
import os
import sys
from nets import G_mlp, D_mlp
from datas import *
from tqdm import trange
sys.path.append("../../")
from regularizers import mi_regularizer


def get_args():

    parser = argparse.ArgumentParser(description='Hyperparams')

    # Bound hyperparam
    parser.add_argument('--regularizer', type=str, required=True)

    # Architecture hyperparm
    parser.add_argument('--critic_layers', type=int, nargs='+', default=[256, 256, 256], 
        help='Layers defining the critic')
    parser.add_argument('--critic_activation', type=str, default='relu')
    parser.add_argument('--critic_type', type=str, default='joint',
        help='Type of critic network used between "joint" or "separate"')

    # Optim hyperparm
    parser.add_argument('--gen_lr', type=float, default=1e-4,
        help='Learning Rate used to train the generator network')
    parser.add_argument('--disc_lr', type=float, default=1e-4,
        help='Learning Rate used to train the discriminator network')
    parser.add_argument('--critic_lr', type=float, default=1e-4,
        help='Learning Rate used to train the critic network')
    parser.add_argument('--beta1', type=float, default=0.5,
        help='Beta1 parameter used in AdamOptimizer for the generator')
    parser.add_argument('--batch_norm', type=bool, default=True,
        help='Whether or not to use batch norm in the generator')
    # Test params
    parser.add_argument('--beta', type=float, default=1,
        help='Regularization strength')
    parser.add_argument('--ema_decay', type=float, default=0.99,
        help='Expoential moving decay for correcting biased gradients')
    parser.add_argument('--unroll_critic', type=int, default=1,
        help='Regularization strength')
    parser.add_argument('--scheduling', type=float, default=400,
        help='If one wishes to remove regularization from objectives after ? epochs')
    parser.add_argument('--seed', type=int, default=0,
        help='If one wishes to remove regularization from objectives after ? epochs')
    

    
    # Training hyperparams
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--data_size', type=int, default=100000)

    # Data hyperparams
    parser.add_argument('--dim_x', type=list, default=[2],
        help='Dimension for X RV')
    parser.add_argument('--dim_z', type=list, default=[2],
        help='Dimension for Z RV')
    parser.add_argument('--data_name', type=str, default="mixture_25_gaussians")
    parser.add_argument('--data_noise', type=float, default=0.05)


    # Display hyperparam
    parser.add_argument("--num_samples_eval", type=int, default=2500, 
        help="Number of samples when computing evaluation metrics")
    parser.add_argument("--plot_every", type=int, default=1, 
        help="In fraction of n_epochs")
    parser.add_argument("--plot_dir", type=str, default='plots/', 
        help="Where to save the plots")
    parser.add_argument("--labels_test_params", type=str, nargs='+', default=['seed'], 
        help="Which parameters are being looped over")


    args = parser.parse_args()
    return args

def sample_z(random_state, m, n):
    #out = random_state.uniform(-1., 1., size=[m, n])
    #return out
    return random_state.randn(m, n)

def perform_adaptive_clipping(regul_term, variables, grad_upper_bound):
    g_r = tf.gradients(regul_term, variables)
    g_r_norm = tf.norm([tf.norm(grad) for grad in g_r])
    coef = tf.minimum(g_r_norm, grad_upper_bound) / g_r_norm
    clipped_regul_term = tf.stop_gradient(coef) * regul_term
    return clipped_regul_term


class CGAN():

    def __init__(self, args, generator, discriminator, data, regularizers, random_state):

        # Recover some quantities
        self.generator = generator
        self.discriminator = discriminator
        self.data = data
        self.regularizer = regularizer
        self.random_state = random_state
        self.beta = args.beta
        self.ema_decay = args.ema_decay
        self.scheduling = args.scheduling
        self.unroll_critic = args.unroll_critic
        self.batch_size = args.batch_size
        self.plot_every = args.plot_every
        self.disc_lr = args.disc_lr
        self.gen_lr = args.gen_lr
        self.beta1 = args.beta1
        self.critic_lr = args.critic_lr

        # Define placeholders for data
        self.z_dim = self.data.z_dim
        self.x_dim = self.data.x_dim

        self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim], name='Input')
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='Latent')
        self.is_training = tf.placeholder(tf.bool, shape=[], name="is_training")

        # Build graph
        self.G_sample = self.generator(self.z, self.is_training)
        self.D_real = self.discriminator(self.x)
        self.D_fake = self.discriminator(self.G_sample)

        # Define losses
        self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake))) \
            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real)))
        self.G_loss_unreg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

        # Define optimizers
        disc_optmizer = tf.train.AdamOptimizer(learning_rate=self.disc_lr, beta1=self.beta1)
        gen_optimizer = tf.train.AdamOptimizer(learning_rate=self.gen_lr, beta1=self.beta1)

        # Define training operations
        gu_and_vars = gen_optimizer.compute_gradients(self.G_loss_unreg, var_list=self.generator.vars)
        gu_and_vars = [(grad, var) for grad, var in gu_and_vars]
        self.gu_norm = tf.norm(tf.convert_to_tensor([tf.norm(grad) for grad, var in gu_and_vars if grad is not None]))
        self.G_solver_unreg = gen_optimizer.apply_gradients(gu_and_vars)
        self.D_solver = disc_optmizer.minimize(self.D_loss, var_list=self.discriminator.vars)

        if self.regularizer is not None:
            train_ops, quantities = self.regularizer(self.z, self.G_sample)
            self.Reg_quantities = quantities
            self.Reg_train_ops = train_ops
            reg_term = quantities['mi_for_grads']
            clipped_reg_term = perform_adaptive_clipping(reg_term, self.generator.vars, self.gu_norm)
            self.G_solver_reg = gen_optimizer.minimize(self.G_loss_unreg - self.beta * clipped_reg_term, var_list=self.generator.vars)

        # Create session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    def train(self, sample_dir, args):

        # Create a batch of data
        X_full_batch = self.data(args.data_size, args.batch_size)

        # Initialize the graph
        feed_dict = {self.x: X_full_batch[0],
                     self.z: sample_z(self.random_state, args.batch_size, self.z_dim),
                     self.is_training: False
                     }
        self.sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

        # Initializwe the csv file where all metrics will be reported
        self.metrics_path = os.path.join(sample_dir, "metrics.csv")
        self.field_names = ["Epoch", "Modes covered", "High quality", "KL"]
        with open(self.metrics_path, "w") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
            writer.writeheader()

        # Start actual training
        fig_count = 0
        epoch_bar = trange(args.epochs)
        for epoch in epoch_bar:
            iter_bar = trange(len(X_full_batch))
            for iteration in iter_bar:
                X_batch = X_full_batch[iteration]

                # Update D
                z = sample_z(self.random_state, args.batch_size, self.z_dim)
                self.sess.run(self.D_solver, feed_dict={self.x: X_batch, self.z: z, self.is_training:False})

                # Update G
                if self.regularizer is not None and epoch < self.scheduling:
                    self.sess.run(self.G_solver_reg, feed_dict={self.z: z, self.is_training: True})
                else:
                    self.sess.run(self.G_solver_unreg, feed_dict={self.z: z, self.is_training: True})

                # Update the bound from regularizer
                if self.regularizer is not None:
                    for _ in range(self.unroll_critic):
                        self.sess.run(
                            self.Reg_train_ops['critic'],
                            feed_dict={self.is_training: False, self.z: sample_z(self.random_state, args.batch_size, self.z_dim)})

                # Update display information
                if iteration % 10 == 0:
                    D_loss_curr, G_loss_curr, gu_norm_curr = self.sess.run([self.D_loss, self.G_loss_unreg, self.gu_norm], feed_dict={self.x: X_batch, self.z: z, self.is_training:False})
                    iter_bar.set_description('Iter: {}; D loss: {:.3g}; G_loss: {:.3g}; |gu|: {:.3g}'.format(epoch, D_loss_curr, G_loss_curr, gu_norm_curr))

                    if self.regularizer is not None:
                        values = self.sess.run(list(self.Reg_quantities.values()), feed_dict={self.is_training: False, self.z: z})
                        iter_bar.set_description(' ; '.join(["{}: {:.3g}".format(name, value) for name, value in zip(self.Reg_quantities.keys(), values)]))

            # Plot current model and update metrics in csv file
            samples_G = self.sess.run(self.G_sample, feed_dict={self.is_training: False, self.z: sample_z(self.random_state, args.num_samples_eval, self.z_dim)})
            fig = self.data.data2fig(samples_G)
            plt.savefig('{}/{}.png'.format(sample_dir, str(fig_count)), bbox_inches='tight')
            metrics = self.data.metrics(samples_G)
            with open(self.metrics_path, "a") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.field_names)
                metrics.update({"Epoch": epoch})
                writer.writerow(metrics)
            fig_count += 1
            plt.close(fig)

if __name__ == '__main__':

    # Recover all parameters for training
    args = get_args()
    labels_test_params = args.labels_test_params

    # Some lines to initialize the directory where plots and metrics will be reported
    if args.seed == 'random':
        args.seed = np.random.randint(10000)
    tf.reset_default_graph()
    tf.set_random_seed(args.seed)
    random_state = np.random.RandomState(args.seed)
    dic = vars(args)
    if args.regularizer != "none":
        regularizer = mi_regularizer(args, args.regularizer)
        sample_dir = '{}/{}_reg={}_'.format(args.plot_dir, args.data_name, args.regularizer) + '-'.join(["{}:{}".format(key, dic[key]) for key in labels_test_params])
    else:
        sample_dir = '{}/{}_seed={}'.format(args.plot_dir, args.data_name, args.seed)
        regularizer = None
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Create whole graph
    data = eval('{}({}, random_state)'.format(args.data_name, args.data_noise))
    generator = G_mlp(data.x_dim, args.batch_norm)
    discriminator = D_mlp()
    cgan = CGAN(args, generator, discriminator, data, regularizer, random_state)

    # Train model
    cgan.train(sample_dir, args)
