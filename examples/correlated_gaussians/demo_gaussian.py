#!/usr/bin/env python

"""demo_gaussian.py: Showcase of how to use mutual information as an estimator. """

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from regularizers import mi_estimator

def get_args():

    parser = argparse.ArgumentParser(description='Hyperparams')

    # Bound hyperparam
    parser.add_argument('--regularizer', type=str, required=True)

    # MI estimator params
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--critic_type', type=str, default='joint',
                        help='Type of critic network used between "joint" or "separate"')
    parser.add_argument('--critic_layers', type=int, nargs='+', default=[256, 256, 256],
                        help='Layers defining the critic')
    parser.add_argument('--critic_activation', type=str, default='relu')
    parser.add_argument('--critic_lr', type=float, default=1e-4,
                        help='Learning Rate used to train the critic network')
    parser.add_argument('--negative_samples', type=int, default=1,
                        help='Number of negative samples used in estimation of the product term')

    # Training hyperparams
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_size', type=int, default=100000)


    # Data hyperparams
    parser.add_argument('--rho_range', nargs=2, type=float, default=[-0.95, 0.95],
                        help='Range for the correlation coefficient Rho')
    parser.add_argument('--rho_points', type=float, default=20,
                        help='Number of points used to discretize the interval defined by --rho_range')
    parser.add_argument('--dim_x', type=list, default=[20],
                        help='Dimension for X and Z RV')



    # Display hyperparam
    parser.add_argument("--print_every", type=int, default=0.01, 
                        help="In fraction of n_epochs_bound, training information refresh cadency")
    parser.add_argument("--plot_dir", type=str, default='plots/', 
                        help="Where to save the plots")

    args = parser.parse_args()
    return args

def generate_correlated_gaussian(data_size, rho, dim_x):
    I = np.eye(dim_x)
    cov = np.block([[I, rho * I], [rho * I, I]])
    mu = np.zeros(2 * dim_x)
    xz = np.random.multivariate_normal(mu, cov, data_size)
    x = xz[:, :dim_x]
    z = xz[:, dim_x:]
    mi = -dim_x / 2 * np.log(1 - rho**2)
    return x, z, mi

def plot(regularizer_name, rho_values, estimated_mis, true_mis, save_path):
    plt.plot(rho_values, estimated_mis, label=regularizer_name)
    plt.plot(rho_values, true_mis, 'r--', linewidth=3.0, label="True")
    plt.xlabel('Rho')
    plt.ylabel('MI (nats)')
    plt.legend()
    plt.savefig(save_path)

def run(args):
    # STEP : Loop over all the combinations of parameters we want to try
    rho_values = np.linspace(args.rho_range[0], args.rho_range[1], args.rho_points)
    true_mis = []
    estimated_mis = []
    for rho in rho_values:
        x, z, mi = generate_correlated_gaussian(args.data_size, rho, args.dim_x[0])
        regularizers = mi_estimator(args.regularizer, args.critic_layers, args.critic_lr, args.critic_activation, args.critic_type, args.ema_decay, args.negative_samples)
        estimated_mis.append(regularizers.fit(x, z, args.batch_size, args.epochs, mi))
        true_mis.append(mi)
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)
    plot(args.regularizer, rho_values, estimated_mis, true_mis, "{}/{}.png".format(args.plot_dir, args.regularizer))


# ------------------------------------------------------------ main ----------------------------------------------------------------------------------- #

if __name__ == "__main__":
    args = get_args()
    d = vars(args)
    d['dim_z'] = args.dim_x
    run(args)
