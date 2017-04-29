# -*- coding: utf-8 -*-

""" Convolutional Neural Network for MNIST dataset classification task.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from numpy import random,array,delete,mean,float32
import os

from scipy import misc
class_1_dir = r'D:\IMEA\Tasking\2017\LearningNeuralNets\MyExample\images\shoes\shoes_preprocessed'
class_2_dir = r'D:\IMEA\Tasking\2017\LearningNeuralNets\MyExample\images\merchendise\merchendise_preprocessed'

X = []
Y = []
for filename in os.listdir(class_1_dir):
    input_file = os.path.join(class_1_dir, filename)
    X.append(misc.imread(input_file))
    Y.append([1,0])

for filename in os.listdir(class_2_dir):
    input_file = os.path.join(class_2_dir, filename)
    X.append(misc.imread(input_file))
    Y.append([0,1])

X = [mean(item,2) for item in X] # flatten
X = array(X)
Y = array(Y)

test_fraction = 0.1
test_idx = random.rand(len(X)) < test_fraction
X_test = X[test_idx]
Y_test = Y[test_idx]
X = array([item for i,item in enumerate(X) if ~test_idx[i]])
Y = array([item for i,item in enumerate(Y) if ~test_idx[i]])

X = X.reshape([-1, 128, 128, 1])
X_test = X_test.reshape([-1, 128, 128, 1])
X = X.astype(float32) / max(X.flatten())
Y = Y.astype(float32)

# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

# Building convolutional network
network = input_data(shape=[None, 128, 128, 1], name='input', data_preprocessing=img_prep,
                     data_augmentation=img_aug)

network = conv_2d(network, 5, 7, activation='sigmoid', name='conv1')
network = max_pool_2d(network, 4)
network = local_response_normalization(network)

network = conv_2d(network, 5, 5, activation='sigmoid', name='conv2')
network = max_pool_2d(network, 4)
network = local_response_normalization(network)

#network = fully_connected(network, 2, activation='sigmoid')
#network = dropout(network, 0.5)

network = fully_connected(network, 2, activation='softmax')

network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=10,
           validation_set=({'input': X_test}, {'target': Y_test}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')

layer1_var = tflearn.variables.get_layer_variables_by_name('conv1')
model.session.as_default()
tflearn.variables.get_value(layer1_var[0])

#pred = model.predict(X_test)
#import numpy as np
#predicted_class = np.array([item.index(max(item)) for item in pred])
#target_class = np.array([list(item).index(max(item)) for item in Y_test])
#np.mean(predicted_class == target_class)

######
#pred = model.predict(X)
#test_class = np.array([list(item).index(max(item)) for item in pred])
#test_target = np.array([list(item).index(max(item)) for item in Y])
#np.mean(test_class == test_target)

