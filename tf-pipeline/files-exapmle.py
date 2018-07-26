#!/usr/bin/env python
"""The files queue example of tensorflow.
   author: randy xu
   date: 20180717
"""

import tensorflow as tf


# Build files name queue
# Which create a FIFO queue to store file names
filename_queue = tf.train.string_input_producer(["example1.txt", "example2.txt"])

# Select reader
reader = tf.TextLineReader()

# Read file
key, value = reader.read(filename_queue)


