# Yet to do

- [ ] Include more architectures for the critic network
- [ ] More comments in functions
- [ ] Testing code when X and Z are images
- [ ] Include units test

# Variational Bounds on Mutual Information

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

### Defining the estimator

<img src="https://github.com/mboudiaf/Variational-Bound-Method/blob/master/screens/graph_init.png" width="200">

To define an estimator, one need no provide two arguments. The first is args and contains all the required information about the critic network. The second is simply the regularization name. Concretly, you can follow this scheme:
  ```python
      from estimator import mi_estimator

      estimation_methods = "mine"
      args = {}
      my_mi_estimator = mi_estimator(args, estimation method)
  ```
  This implementation offers two main functionalities that fit two use cases:
 ### Measuring the MI between static data
 
<img src="https://github.com/mboudiaf/Variational-Bound-Method/blob/master/screens/graph_fit.png" width="200">
For simple MI estimation between two random variables:

  ```python
      from estimator import mi_estimator

      estimation_methods = "mine"
      my_mi_estimator = mi_estimator(args, estimation method) 

      mi_estimate = my_mi_estimator.fit(x_data, z_data)
  ```

### Using MI as a regularizer in TensorFlow graph

<img src="https://github.com/mboudiaf/Variational-Bound-Method/blob/master/screens/graph_call.png" width="200">
For use in a Tensorflow graph (as a regulazation term for instance)

  ```python
      from estimator import mi_estimator

      estimation_methods = "mine"
      my_mi_estimator = mi_estimator(args, estimation method) 
      
      x = tf.placeholder(shape=[batch_size, dim_x], tf.float32)
      z = tf.placeholder(shape=[batch_size, dim_x], tf.float32)
      estimator_train_op, estimator_quantities = my_mi_estimator(x, z)
      
      loss = loss_unregularized + beta * estimator_quantities['mi_for_grads']
      
      main_train_op = optimizer.minimize(loss, var_list=...)
    
      ... 
      
      # Perfom main training operation to optimize loss
      sess.run([main_train_op], feed_dict={x: ..., z= ..., ...})
      # Then update estimator 
      sess.run(estimator_train_op, feed_dict={x: ..., z= ...})   
  ```
 

## Examples

We provide code to showcase these two functionalities 

### Estimation of mutual information

In the case of correlated Gaussian random variables:

<img src="https://github.com/mboudiaf/Variational-Bound-Method/blob/master/screens/gaussian_rvs.png" width="200">

We have access to the mutual information:

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

<img src="https://github.com/mboudiaf/Mutual-Information-Variational-Bounds/blob/master/screens/cor_gaussian_mine.png" width="250"><img src="https://github.com/mboudiaf/Mutual-Information-Variational-Bounds/blob/master/screens/cor_gaussian_nwj.png" width="250"><img src="https://github.com/mboudiaf/Mutual-Information-Variational-Bounds/blob/master/screens/cor_gaussian_nce.png" width="250">

The plots above present the estimated MI after 10 epochs of training on a 100k samples dataset with batch size 128, for the three estimation methods.


## Using MI as a regularization term

The most interesting use case of these bounds is in the context of mutual information maximization. A typical example is reduction of mode collapse in GANs. In the context of GANs, the mutual information I(Z;X) is used as a proxy for the entropy of the generator H(X), where X represents the output of the generator, and Z the noise vector. The maximization of I(X;Z) results in the maximization of H(X).

```
loss_regularized = loss_gan - beta * I(X;Z)
```
We provide a simple example in 2D referred to as "25 gaussians experiments" where the target distribution is:

<img src="https://github.com/mboudiaf/Mutual-Information-Variational-Bounds/blob/master/screens/gan_target.png" width="250">

The simple GAN will produce, with the provided generator and discriminator architecture distributions like:
<img src="https://github.com/mboudiaf/Mutual-Information-Variational-Bounds/blob/master/screens/gan _noreg.png" width="250">

While the above plot clearly exposes a mode collapse, one can easily reduce this mode collapse by adding a MI regularization:
<img src="https://github.com/mboudiaf/Mutual-Information-Variational-Bounds/blob/master/screens/gan_mine.png" width="250">

To see the code for this example, first go the example directory
```
cd examples/gan/
```
Then, run some tests to make sure the code doesn't yield any bug:
```
python3 run_test.py
```
Finally, to run the experiments, you can check all the available options in "demo_gaussian.py", and loop over any parameters by modifying the header of run_exp.py.
Then simply run:
```
python3 run_exp.py
```


## Authors

* **Malik Boudiaf**

## References

[1] Ishmael  Belghazi,  Sai  Rajeswar,  Aristide  Baratin,  R.  Devon  Hjelm,  and  Aaron  C.Courville.   MINE:  mutual  information  neural  estimation.CoRR,  https://arxiv.org/abs/1801.04062

[2] Sebastian Nowozin, Botond Cseke, Ryota Tomioka, f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization, https://arxiv.org/abs/1606.00709

[3] Aaron van den Oord, Yazhe Li, Oriol Vinyals, Representation Learning with Contrastive Predictive Coding, https://arxiv.org/abs/1807.03748
