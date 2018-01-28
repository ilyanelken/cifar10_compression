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


# Experiment results:
------------------
> 64 conv1 channels, trained from scratch : accuracy top-1 = 85.413 [%]

> 64 conv1 channels, trained from scratch : accuracy top-5 = 99.209 [%]

> 48 conv1 channels, trained from scratch : accuracy top-1 = 85.384 [%]

> 48 conv1 channels, w/o reconstruction, w/o fine tuning : accuracy top-1 = 82.021 [%]

> 48 conv1 channels, with reconstruction, w/o fine tuning : accuracy top-1 = 82.219 [%]

> 48 conv1 channels, w/o reconstruction, with fine tuning : accuracy top-1 = 85.166 [%]

> 48 conv1 channels, with reconstruction, with fine tuning : accuracy top-1 = 84.988 [%]

> 48 conv1 channels, trained from scratch : accuracy top-5 = 99.150 [%]

> 48 conv1 channels, w/o reconstruction, w/o fine tuning : accuracy top-5 = 98.883 [%]

> 48 conv1 channels, with reconstruction, w/o fine tuning : accuracy top-5 = 98.883 [%]

> 48 conv1 channels, w/o reconstruction, with fine tuning : accuracy top-5 = 99.248 [%]

> 48 conv1 channels, with reconstruction, with fine tuning : accuracy top-5 = 99.308 [%]

> 32 conv1 channels, trained from scratch : accuracy top-1 = 85.018 [%]

> 32 conv1 channels, w/o reconstruction, w/o fine tuning : accuracy top-1 = 67.652 [%]

> 32 conv1 channels, with reconstruction, w/o fine tuning : accuracy top-1 = 69.116 [%]

> 32 conv1 channels, w/o reconstruction, with fine tuning : accuracy top-1 = 83.999 [%]

> 32 conv1 channels, with reconstruction, with fine tuning : accuracy top-1 = 83.544 [%]

> 32 conv1 channels, trained from scratch : accuracy top-5 = 99.209 [%]

> 32 conv1 channels, w/o reconstruction, w/o fine tuning : accuracy top-5 = 96.479 [%]

> 32 conv1 channels, with reconstruction, w/o fine tuning : accuracy top-5 = 96.915 [%]

> 32 conv1 channels, w/o reconstruction, with fine tuning : accuracy top-5 = 99.298 [%]

> 32 conv1 channels, with reconstruction, with fine tuning : accuracy top-5 = 99.239 [%]

> 24 conv1 channels, trained from scratch : accuracy top-1 = 84.523 [%]

> 24 conv1 channels, w/o reconstruction, w/o fine tuning : accuracy top-1 = 44.215 [%]

> 24 conv1 channels, with reconstruction, w/o fine tuning : accuracy top-1 = 49.229 [%]

> 24 conv1 channels, w/o reconstruction, with fine tuning : accuracy top-1 = 83.356 [%]

> 24 conv1 channels, with reconstruction, with fine tuning : accuracy top-1 = 82.180 [%]

> 24 conv1 channels, trained from scratch : accuracy top-5 = 99.100 [%]

> 24 conv1 channels, w/o reconstruction, w/o fine tuning : accuracy top-5 = 82.239 [%]

> 24 conv1 channels, with reconstruction, w/o fine tuning : accuracy top-5 = 87.134 [%]

> 24 conv1 channels, w/o reconstruction, with fine tuning : accuracy top-5 = 99.169 [%]

> 24 conv1 channels, with reconstruction, with fine tuning : accuracy top-5 = 99.288 [%]

> 16 conv1 channels, trained from scratch : accuracy top-1 = 83.366 [%]

> 16 conv1 channels, w/o reconstruction, w/o fine tuning : accuracy top-1 = 31.794 [%]

> 16 conv1 channels, with reconstruction, w/o fine tuning : accuracy top-1 = 36.452 [%]

> 16 conv1 channels, w/o reconstruction, with fine tuning : accuracy top-1 = 81.507 [%]

> 16 conv1 channels, with reconstruction, with fine tuning : accuracy top-1 = 79.856 [%]

> 16 conv1 channels, trained from scratch : accuracy top-5 = 99.140 [%]

> 16 conv1 channels, w/o reconstruction, w/o fine tuning : accuracy top-5 = 76.266 [%]

> 16 conv1 channels, with reconstruction, w/o fine tuning : accuracy top-5 = 79.717 [%]

> 16 conv1 channels, w/o reconstruction, with fine tuning : accuracy top-5 = 99.061 [%]

> 16 conv1 channels, with reconstruction, with fine tuning : accuracy top-5 = 98.784 [%]

> 8 conv1 channels, trained from scratch : accuracy top-1 = 81.260 [%]

> 8 conv1 channels, w/o reconstruction, w/o fine tuning : accuracy top-1 = 16.169 [%]

> 8 conv1 channels, with reconstruction, w/o fine tuning : accuracy top-1 = 23.408 [%]

> 8 conv1 channels, w/o reconstruction, with fine tuning : accuracy top-1 = 75.554 [%]

> 8 conv1 channels, with reconstruction, with fine tuning : accuracy top-1 = 72.785 [%]

> 8 conv1 channels, trained from scratch : accuracy top-5 = 98.754 [%]

> 8 conv1 channels, w/o reconstruction, w/o fine tuning : accuracy top-5 = 74.525 [%]

> 8 conv1 channels, with reconstruction, w/o fine tuning : accuracy top-5 = 76.216 [%]

> 8 conv1 channels, w/o reconstruction, with fine tuning : accuracy top-5 = 98.250 [%]

> 8 conv1 channels, with reconstruction, with fine tuning : accuracy top-5 = 97.696 [%]
