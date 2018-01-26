# Deep Learning Seminar project
-----------------------------

Seminar web: http://web.eng.tau.ac.il/deep_learn/

Project members:

 - Ilya Nelkenbaum (ilya@nelkenbaum.com)

 - Maxim Roshior   (maximus.1987@gmail.com)
 
 
# Project summary:
----------------
 
 Implement CNN compression method described in:
 
 "ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression"
 

 Apply the compression method on cifar10 CNN implemented in Google
 Tensorflow framework.

# Based on code from:
-------------------
https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10


# Hardware used:
--------------

> GPU: GeForce GTX TITAN X


> CPU: i7-4770 @ 3.40GHz


Training settings:
------------------

> Training set:     50,000 images


> Compression set:  50,000 images


> Test set:         10,000 images


> Epochs:           128


> Batch size:       128


> Optimizer:        SGD


> Initialization:   truncated random distribution with std=5e-2 for conv and std=0.04 for fully connected


> Learning rate:    initial 0.1 with exponential decay (factor: 0.1)


> Regularization:   weight decay 0.004 for fully connected layers only


> Drop-out:         no dropout 


# Baseline model results:
------------------

> Batch average run time (on CPU):  XXX [ms]


> Accuracy:   85.7 [%]

