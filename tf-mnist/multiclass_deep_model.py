#!/usr/bin/env python
"""multiclass_deep_model.py
   The architecture with 2 cnn 2 fully connected.
   author: randy xu
   date: 2018.07.23
"""

import tensorflow as tf

class Deep(object):
    def __init__(self, x, keep_prob, num_classes):
        # Parser input arguments into classes variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob

        # Call the create function to build computational graph of CNN2
        self.create()

    def create(self):
        """Create the network graph."""
        with tf.name_scope('reshape'):
            self.X = tf.reshape(self.X, [-1, 28, 28, 1])

        # 1st layer convolutional - maps one grapscale image to 32 feature
        # maps.
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
            b_conv1 = bias_variable([32], 'b_conv1')
            h_conv1 = tf.nn.relu(conv2d(self.X, W_conv1) + b_conv1,
                                 name='h_conv1')
            h_pool1 = max_pool_2x2(h_conv1, 'h_pool1')

        # 2nd layer convolutional - maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
            b_conv2 = bias_variable([64], 'b_conv2')
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2,
                                 name='h_conv2')
            h_pool2 = max_pool_2x2(h_conv2, 'h_pool2')

        # 3rd layer fully connected -- after 2 round of downsampling, our 28x28
        # image is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([7 * 7 * 64, 1024], 'W_fc1')
            b_fc1 = bias_variable([1024], 'b_fc1')
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], 'h_pool2_flat')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,
                               name='h_fc1')
            h_fc1_drop = tf.nn.dropout(h_fc1, self.KEEP_PROB, name='h_fc1_drop')
        
        # 4th layer
        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([1024, self.NUM_CLASSES],'W_fc2')
            b_fc2 = bias_variable([self.NUM_CLASSES],'b_fc2')
            self.logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # 5th layer
        with tf.name_scope('softmax'):
            self.output = tf.nn.softmax(self.logits)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)
