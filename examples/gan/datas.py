import os,sys
from PIL import Image
import scipy.misc
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
import itertools



class spiral2D():
    def __init__(self, noise_std, random_state):
        self.x_dim = 2
        self.z_dim = 2
        self.num_modes = 3
        self.std = noise_std
        self.random_state = random_state

    def __call__(self, data_size, batch_size):

        xy1, xy2, xy3 = self.generate_spiral(data_size)
        xy = np.concatenate([xy1, xy2, xy3], axis=0)
        xy += self.std*self.random_state.randn(xy.shape[0], xy.shape[1])

        self.random_state.shuffle(xy)

        batch = []
        for i in range(int(data_size/batch_size)):
            batch.append(xy[i*batch_size:(i+1)*batch_size , :])
        return batch


    def generate_spiral(self, data_size):
        theta_max = 2*np.pi
        rmax = 2.5
        r_outter = 6

        modes_prob = [1/3, 1/3, 1/3]

        size1 = int(data_size*modes_prob[0])
        size2 = int(data_size*modes_prob[1])
        size3 = int(data_size*modes_prob[2])

        theta1 = np.expand_dims(theta_max*self.random_state.rand(size1), 1)
        theta2 = np.expand_dims(theta_max*self.random_state.rand(size2), 1)
        theta3 = np.expand_dims(theta_max*self.random_state.rand(size3), 1)

        x1 = (0.5 + rmax/(theta_max)*theta1)*np.sin(theta1)
        y1 = (0.5 + rmax/(theta_max)*theta1)*np.cos(theta1)
        xy1 = np.concatenate((x1,y1), axis=1)

        x2 = -(0.5 + rmax/(theta_max)*theta2)*np.sin(theta2)
        y2 = -(0.5 + rmax/(theta_max)*theta2)*np.cos(theta2)
        xy2 = np.concatenate((x2,y2), axis=1)

        x3 = r_outter*np.cos(theta3)
        y3 = r_outter*np.sin(theta3)
        xy3 = np.concatenate((x3,y3), axis=1)
        xy  = np.concatenate((xy1,xy2,xy3), axis=0)
        return [xy1, xy2, xy3]


    def data2fig(self, samples, samples_proposal=None):

        #ref_points
        num_samples = samples.shape[0]
        true_points = self(num_samples,num_samples)[0]
        true_x = true_points[:,0]
        true_y = true_points[:,1]

        fig = plt.figure()
        xdata = samples[:,0]
        ydata = samples[:,1]

        #plt.scatter(xdata, ydata, 1, c='black', label='generated samples', )
        if samples_proposal is not None:
            plt.scatter(samples_proposal[:,0], samples_proposal[:,1])
        plt.scatter(true_x, true_y, 1, c='black', label='data samples')
        
        return fig

    def metrics(self, samples_G):
        n_samples = samples_G.shape[0]
        xy1, xy2, xy3 = self.generate_spiral(100000)
        modes={0:xy1, 1:xy2, 2:xy3}
        modes_captured = np.zeros(self.num_modes)
        high_quality_samples = 0
        for sample in samples_G:
            for mode_num, samples_mode in modes.items():
                dist_2_samples = np.sqrt(np.sum((samples_mode - sample)**2, axis=1))
                if np.min(dist_2_samples) <= 3 * self.std:
                    modes_captured[mode_num] += 1
                    high_quality_samples += 1
        assert(high_quality_samples<=n_samples)
        q_hat = modes_captured / high_quality_samples
        reverse_kl = np.sum(np.log(np.power(self.num_modes * q_hat, q_hat)))
        metrics = {'High quality': high_quality_samples / n_samples * 100, 'Modes covered': np.sum(modes_captured >= 1), 'KL':reverse_kl}
        return metrics


class mixture_25_gaussians():
    def __init__(self, noise_var, random_state):
        self.x_dim = 2
        self.z_dim = 2
        self.std = noise_var
        self.num_components = 25
        self.random_state = random_state

    def __call__(self, data_size, batch_size):
        chosen_mode = self.random_state.choice(self.num_components, size=data_size)

        mus = [np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),range(-4, 5, 2))]
        cov = self.std**2 * np.eye(2)#[np.array([s,s]).astype(float32) for i in range(self.num_components)]
        data = np.array([self.random_state.multivariate_normal(mean=mus[i],cov=cov) for i in chosen_mode])
        
        batch = []
        for i in range(int(data_size / batch_size)):
            batch.append(data[i * batch_size:(i + 1) * batch_size, :])
        return batch

    def data2fig(self, samples_G, samples_prop=None):

        #ref_points
        num_samples = samples_G.shape[0]
        true_points = self(num_samples, num_samples)[0]
        true_x = true_points[:, 0]
        true_y = true_points[:, 1]

        fig = plt.figure()
        xdata = samples_G[:, 0]
        ydata = samples_G[:, 1]


        #plt.scatter(true_x, true_y, 1,c='black', label='data samples')
        plt.scatter(xdata, ydata, 1, c='black', label='generated samples', )
        if samples_prop is not None:
            xprop = samples_prop[:, 0]
            yprop = samples_prop[:, 1]
            plt.scatter(xprop, yprop, alpha=0.4, label='proposal samples')
        return fig

    def metrics(self, samples_G):
        modes = [np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2), range(-4, 5, 2))]
        n_samples = samples_G.shape[0]
        modes_captured = np.zeros(self.num_components)
        high_quality_samples = 0
        for sample in samples_G:
            dist_2_modes = np.sqrt(np.sum((modes - sample)**2, axis=1))
            if np.min(dist_2_modes) <= 3 * self.std:
                modes_captured[np.argmin(dist_2_modes)] += 1
                high_quality_samples += 1
        q_hat = modes_captured / high_quality_samples
        reverse_kl = np.sum(np.log(np.power(self.num_components * q_hat, q_hat)))
        metrics = {'High quality': high_quality_samples / n_samples * 100, 'Modes covered': np.sum(modes_captured >= 1), 'KL':reverse_kl}
        return metrics
