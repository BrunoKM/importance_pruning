# Importance Critorion Pruned Deep Neural Networks

The idea behind this method is to compute the relative importance in determining the output of each node in a layer of a trained neural network. This way, the parameters that contribute little (on average) to classification of the output will get a low importance score. This happens independently of whether they contributed towards lowering the cost function (unlike the training algorithms). Based on those importance scores, the network can than be pruned, reducing its size. The new network can potentially lead to a better performance, due to a higher generalisation potential.

### CNN implementation

To implement this method with CNN neural networks one needs access to the intermediate stages of a convolution operation, the tensor resulting from applying the im2col algorithm. For this reason these algortihms need to be reacreated in numpy and tensorflow.
