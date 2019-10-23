import numpy as np
from Mi_estimator import Mi_estimator

data_size = 1000000
batch_size = 100
dim_x = [15, 15, 3]
dim_z = [15, 15, 3]
x_data = np.random.rand(data_size, *dim_x)
z_data = x_data + 3

my_mi_estimator = Mi_estimator(regu_name = 'mine',
                               dim_x = dim_x,
                               dim_z = dim_z,
                               batch_size= 128,
                               critic_layers=[256, 256, 256],
                               critic_lr=1e-4, critic_activation='relu',
                               critic_type='joint',
                               ema_decay=0.99,
                               negative_samples=1)

mi_estimate = my_mi_estimator.fit(x_data, z_data, epochs=10, ground_truth_mi=None)

print('Final estimate : {}'.format(mi_estimate))