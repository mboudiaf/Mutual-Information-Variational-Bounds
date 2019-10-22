# Yet to do

- [ ] Include more architectures for the critic network
- [ ] More comments in functions
- [ ] Testing code when X and Z are images

# Variational Bounds of Mutual Information

Throughout this repo, I offer a ready-to-use implementation of state-of-the-art variational methods for mutual information estimation in Tensorflow.
This includes:
  * MINE estimator [1], based on the Donsker-Varadhan representation of the KL divergence
  * NWJ estimator [2], based on the f-divergence variational representation of the KL divergence
  * NCE estimator [3], based on noise contrastive estimation principle
  

### Prerequisites

Required packages are list in requirement.txt. Make sure all these packages are installed on your local machine.

```
pip3 install -r requirements.txt
```

## Getting Started

This implementation offers two main functionalities that fit two use cases:

  * For simple MI estimation between two random variables:

  ```python
      from estimator import mi_estimator

      estimation_methods = "mine"
      my_mi_estimator = mi_estimator(args, estimation method) 

      mi_estimate = my_mi_estimator.fit(x_data, z_data)
  ```

  * For use in a Tensorflow graph (as a regulazation term for instance)

  ```python
      from estimator import mi_estimator

      estimation_methods = "mine"
      my_mi_estimator = mi_estimator(args, estimation method) 
      
      x = tf.placeholder(shape=[batch_size, dim_x], tf.float32)
      z = tf.placeholder(shape=[batch_size, dim_x], tf.float32)
      mi_regularization_term = my_mi_estimator(x, z)
      
      loss = loss_unregularized + beta * mi_regularization_term
  ```
 

## Examples

We provide code to showcase these two functionalities 

### Estimation of mutual information

In the case of correlated Gaussian, we have access to the mutual information:

<img src="https://github.com/mboudiaf/Variational-Bound-Method/blob/master/screens/gaussian_mi.png" width="400">

First go the example directory
```
cd examples/correlated_gaussians/
```
Then, run some tests to make sure the code doesn't yield any bug:
```
python3 run_test.py
```
Finally, to run the experiments, you can check all the available options in "demo_gaussian.py", and loop over any parameters by modifying the header of run_exp.py
When that is done, simply run:
```
python3 run_exp.py
```
After training, plots are avaible in /plots/. You should obtain something like:

<img src="https://github.com/mboudiaf/Variational-Bound-Method/blob/master/screens/20_seeds.png" width="400">

The plot above represents the estimated mutual info after training for one estimation method.


## Using MI as a regularization term


## Authors

* **Malik Boudiaf**

## References

[1] Ishmael  Belghazi,  Sai  Rajeswar,  Aristide  Baratin,  R.  Devon  Hjelm,  and  Aaron  C.Courville.   MINE:  mutual  information  neural  estimation.CoRR,  https://arxiv.org/abs/1801.04062
[2] Sebastian Nowozin, Botond Cseke, Ryota Tomioka, f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization, https://arxiv.org/abs/1606.00709
[3] Aaron van den Oord, Yazhe Li, Oriol Vinyals, Representation Learning with Contrastive Predictive Coding, https://arxiv.org/abs/1807.03748
