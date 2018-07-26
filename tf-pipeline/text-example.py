#!/usr/bin/env python
"""The tensorflow reader example of text file format.
   author: randy xu
   date: 2018.07.17
"""

import tensorflow as tf

dataset = tf.data.TextLineDataset("example.txt")

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(3):
        print(sess.run(next_element).decode('UTF-8'))

dataset = tf.data.TextLineDataset("example.txt")
dataset = dataset.map(lambda string: tf.string_split([string]).values)
dateset = dataset.shuffle(buffer_size=3)
dataset = dataset.batch(2)
dataset = dataset.prefetch(1)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    print(sess.run(next_element))

print("Use initializable iterators")
dataset = tf.data.TextLineDataset("example.txt")

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
init_op = iterator.initializer
with tf.Session() as sess:
    sess.run(init_op)
    # Initialize the iterator
    print(sess.run(next_element))
    print(sess.run(next_element))
    # Move the iterator back to the begining
    sess.run(init_op)
    print(sess.run(next_element))
