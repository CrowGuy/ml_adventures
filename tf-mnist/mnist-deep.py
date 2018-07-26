#!/usr/bin/env python
"""mnist-deep.py
   mnist with deep architecture.
   author: randy xu
   date: 2018.07.20
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import csv
import time

from multiclass_deep_model import Deep

BATCH_SIZE = 100
NUM_CLASSES = 10
TOTAL_EPOCHS = 50

# Hyperparameter
dropout_rate = 0.5
init_learning_rate = 1e-3

def parse_function(filename, label):
    """Parse image file and label.
    Args:
        filename (str): the image name.
        label (str): the label of image.

    Returns:
        image (tensor): the image tensor.
        label (str): the label of image.
    """
    print("================= Image parse ===============")
    
    # Convert label number into one-hot-encoding
    label = tf.one_hot(label, NUM_CLASSES)
    
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=1)

    # This will convert to float values in [0 ,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image
    image = tf.image.resize_images(image, [28, 28])

    image = image[:, :, ::-1]
    image = tf.reshape(image, [-1])

    return image, label

def flip_up_down(image, label):
    image = tf.image.flip_up_down(image)
    return image, label

def flip_left_right(image, label):
    image = tf.image.flip_left_right(image)
    return image, label

def rot_90(image, label):
    image = tf.image.rot90(image, k=1)
    return image, label

def rot_180(image, label):
    image = tf.image.rot90(image, k=2)
    return image, label

def rot_270(image, label):
    image = tf.image.rot90(image, k=3)
    return image, label

def brightness(image, label):
    image = tf.image.random_brightness(image, max_delta=128.0/255.0)
    return image, label

def saturation(image, label):
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    return image, label

def hue(image, label):
    image = tf.image.random_hue(image, max_delta=0.05)
    return image, label

def contrast(image, label):
    image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    return image, label

def read_text_file(text_file):
    filenames = []
    labels = []
    if text_file.endswith('.txt'):
        # Read data from txt format file
        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(',')
                filenames.append(items[0])
                labels.append(int(items[1][:-1]))
    elif text_file.endswith('.csv'):
        # Read data from csv format
        with open(text_file) as csvf:
            reader = csv.DictReader(csvf)
            for row in reader:
                filenames.append(row['IMG_PATH'])
                labels.append(int(row['LABEL']))
    return filenames, labels, len(filenames)

def generate_dataset(filenames, labels, shuffle=False, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    
    if augment:
        # Data augmentation - Flip
        data_up_down = dataset.map(flip_up_down, num_parallel_calls=4)
        data_left_right = dataset.map(flip_left_right, num_parallel_calls=4)

        # Data augmentation - Rotation
        data_rot_90 = dataset.map(rot_90, num_parallel_calls=4)
        data_rot_180 = dataset.map(rot_180, num_parallel_calls=4)
        data_rot_270 = dataset.map(rot_270, num_parallel_calls=4)

        # Data augmentation - Brightness
        data_brightness = dataset.map(brightness, num_parallel_calls=4)

        # Data augmentation - Stauration
        data_saturation = dataset.map(saturation, num_parallel_calls=4)

        # Data augmentation - Hue
        data_hue = dataset.map(hue, num_parallel_calls=4)

        # Data augmentation - Contrast
        data_contrast = dataset.map(contrast, num_parallel_calls=4)

        dataset = dataset.concatenate(data_up_down)
        dataset = dataset.concatenate(data_left_right)
        dataset = dataset.concatenate(data_rot_90)
        dataset = dataset.concatenate(data_rot_180)
        dataset = dataset.concatenate(data_rot_270)
        dataset = dataset.concatenate(data_brightness)
        dataset = dataset.concatenate(data_saturation)
        dataset = dataset.concatenate(data_hue)
        dataset = dataset.concatenate(data_contrast)

    if shuffle:
        if augment:
            dataset = dataset.shuffle(len(filenames)*10)
        else:
            dataset = dataset.shuffle(len(filenames))

    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)
    return dataset


# Read the data pathes and labels
train_files, train_labels, train_num = read_text_file('texts/mnist_train.txt')
test_files, test_labels, test_num = read_text_file('texts/mnist_test.txt')

# Build data set
train_data = generate_dataset(train_files, train_labels)
test_data = generate_dataset(test_files, test_labels)

train_data_iterator = train_data.make_initializable_iterator()
test_data_iterator = test_data.make_initializable_iterator()

next_train_data = train_data_iterator.get_next()
next_test_data = test_data_iterator.get_next()
init_train_data_op = train_data_iterator.initializer
init_test_data_op = test_data_iterator.initializer

train_steps = int(np.floor(train_num) / BATCH_SIZE)
test_steps = int(np.floor(test_num) / BATCH_SIZE)

# TF placeholder for graph input and output
x_in = tf.placeholder(tf.float32, [None, 784])
y_out = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = Deep(x_in, dropout_rate, NUM_CLASSES).output

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_out,
                                                              logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_out,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

time_start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(TOTAL_EPOCHS):
        sess.run(init_train_data_op)
        for step in range(train_steps):
            images, labels = sess.run(next_train_data)
            train_feed_dict = {x_in: images, y_out: labels,
                               learning_rate: init_learning_rate}
            _, loss = sess.run([train_op, loss_op], feed_dict=train_feed_dict)
            if step % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict=train_feed_dict)
                print("Step:", step, "Loss:", loss, "Training accuracy:",
                      train_accuracy)
        sess.run(init_test_data_op)
        accuracy_rate = 0.
        test_count= 0
        for step in range(test_steps):
            test_images, test_labels = sess.run(next_test_data)
            test_feed_dict = {x_in: test_images, y_out: test_labels}
            test_accuracy = sess.run(accuracy, feed_dict=test_feed_dict)
            accuracy_rate += test_accuracy
            test_count += 1
        test_accuracy /= test_count
        print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy = ',
        accuracy_rate)
print('Computing time:', (time.time() - time_start))
