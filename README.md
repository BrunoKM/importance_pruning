# Importance Critorion Pruned Deep Neural Networks

The idea behind this method is to compute the relative importance in determining the output of each node in a layer of a trained neural network. This way, the parameters that contribute little (on average) to classification of the output will get a low importance score. This happens independently of whether they contributed towards minimising the value of the cost function, contrary to the training algorithms. Based on those importance scores, the network can then be pruned, reducing its size. The new network can potentially lead to a better performance, due to a higher generalisation potential.

## Files:

### fully_connected_experiment.ipynb

First implementation of the method to a generic fully connected neural network trained on the MNIST dataset, laid out in a notebook format.

### custom_conv2d.py

Functions necessary to implement a conv2d algorithm in TensorFlow with access to the intermediary stages of the convolution. This implementation is slower than the built in TensorFlow conv2d function; however, it allows the implementation of im2col separately.
