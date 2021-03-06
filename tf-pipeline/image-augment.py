#!/usr/bin/env python
"""image-augment.py
   The image augmentation switch case.
   author: randy xu
   date: 2018.07.19
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

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
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0 ,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image
    image = tf.image.resize_images(image, [128, 128])

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

filenames = ['images/angry_bird.jpg']
labels = ['bird']
batch_size = 1

# Visualize the original image
img_bird = mpimg.imread(filenames[0])
print("The shape of %s is %s" % (filenames[0], img_bird.shape))
plt.imshow(img_bird)
plt.show()

# Build data set
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(parse_function, num_parallel_calls=4)

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

dataset = dataset.shuffle(10)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
init_op = iterator.initializer

with tf.Session() as sess:
    sess.run(init_op)
    image_count = 0
    plt.figure(num='Angry Birds Dataset', figsize=(10,10))
    while True:
        try:
            image, label = sess.run(next_element)
            plt.subplot(5,2,image_count + 1)
            plt.title(label[0].decode('UTF-8'))
            plt.imshow(image.reshape(128,128,3))
            plt.axis('off')
            image_count += 1
        except tf.errors.OutOfRangeError:
            break
    plt.show()
