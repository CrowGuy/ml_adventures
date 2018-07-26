#!/usr/bin/env python
"""image-process.py
   The image process example for image data set. These functions could be
   used to do data preprocess and data augmentation.
   author: randy xu
   date: 2018.07.17
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage import transform

def parse_function(filename, label):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0 ,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image
    image = tf.image.resize_images(image, [128, 128])

    return image, label

filenames = ['images/angry_bird.jpg']
labels = ['bird']

# Visualize the original image
img_cat = mpimg.imread(filenames[0])
print("The shape of %s is %s" % (filenames[0], img_cat.shape))
plt.imshow(img_cat)
plt.show()

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(parse_function, num_parallel_calls=4)
dataset = dataset.batch(1)
dataset = dataset.prefetch(1)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
init_op = iterator.initializer

with tf.Session() as sess:
    sess.run(init_op)
    image, label = sess.run(next_element)
    print(image.shape, label)
    
    # Image process - Flip
    plt.figure(num='Image process - Flip', figsize=(15,3))
    plt.subplot(1,5,1)
    plt.title('Original')
    plt.imshow(image.reshape(128,128,3))
    plt.axis('off')

    up_down = tf.image.flip_up_down(image.reshape(128,128,3))
    up_down_image = sess.run(up_down)
    plt.subplot(1,5,2)
    plt.title('Flip up down')
    plt.imshow(up_down_image)
    plt.axis('off')

    left_right = tf.image.flip_left_right(image.reshape(128,128,3))
    left_right_image = sess.run(left_right)
    plt.subplot(1,5,3)
    plt.title('Flip left right')
    plt.imshow(left_right_image)
    plt.axis('off')

    up_down_random = tf.image.random_flip_up_down(image.reshape(128,128,3))
    up_down_random_image = sess.run(up_down_random)
    plt.subplot(1,5,4)
    plt.title('Flip up down random')
    plt.imshow(up_down_random_image)
    plt.axis('off')

    left_right_random = tf.image.random_flip_left_right(image.reshape(128,128,3))
    left_right_random_image = sess.run(left_right_random)
    plt.subplot(1,5,5)
    plt.title('Flip left right random')
    plt.imshow(left_right_random_image)
    plt.axis('off')

    plt.show()

    # Image process - Rotation
    plt.figure(num='Image process - Rotation', figsize=(15,3))
    plt.subplot(1,5,1)
    plt.title('Original')
    plt.imshow(image.reshape(128,128,3))
    plt.axis('off')

    rot_90 = tf.image.rot90(image.reshape(128,128,3), k=1)
    rot_90_image = sess.run(rot_90)
    plt.subplot(1,5,2)
    plt.title('Rot 90 degree')
    plt.imshow(rot_90_image)
    plt.axis('off')

    rot_180 = tf.image.rot90(image.reshape(128,128,3), k=2)
    rot_180_image = sess.run(rot_180)
    plt.subplot(1,5,3)
    plt.title('Rot 180 degree')
    plt.imshow(rot_180_image)
    plt.axis('off')

    rot_tf_45 = tf.contrib.image.rotate(image.reshape(128,128,3), angles=0.7854)
    rot_tf_45_image = sess.run(rot_tf_45)
    plt.subplot(1,5,4)
    plt.title('Rot 45 degree')
    plt.imshow(rot_tf_45_image)
    plt.axis('off')

    rot_tf_135 = tf.contrib.image.rotate(image.reshape(128,128,3), angles=2.3562)
    rot_tf_135_image = sess.run(rot_tf_135)
    plt.subplot(1,5,5)
    plt.title('Rot 135 degree')
    plt.imshow(rot_tf_135_image)
    plt.axis('off')

    plt.show()

    # Image process - Crop
    plt.figure(num='Image process - Crop', figsize=(15,3))
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(image.reshape(128,128,3))
    plt.axis('off')

    crop = tf.image.central_crop(image.reshape(128,128,3), 0.5)
    crop_image = sess.run(crop)
    plt.subplot(1,2,2)
    plt.title('Central crop')
    plt.imshow(crop_image)

    plt.show()

    # Image process - Translation
    plt.figure(num='Image process - Translation', figsize=(15,3))
    plt.subplot(1,3,1)
    plt.title('Original')
    plt.imshow(image.reshape(128,128,3))
    plt.axis('off')

    translate_right = tf.image.pad_to_bounding_box(image.reshape(128,128,3),
                                                   0, 0, 128, 168)
    translate_right = tf.image.crop_to_bounding_box(translate_right, 0, 20,
                                                    128, 128)
    translate_right_image = sess.run(translate_right)
    plt.subplot(1,3,2)
    plt.title('Translate Right')
    plt.imshow(translate_right_image)
    plt.axis('off')

    translate_upward = tf.image.pad_to_bounding_box(image.reshape(128,128,3),
                                                    20, 0, 148, 128)
    translate_upward = tf.image.crop_to_bounding_box(translate_upward, 0, 0,
                                                     128, 128)
    translate_upward_image = sess.run(translate_upward)
    plt.subplot(1,3,3)
    plt.title('Translate Upward')
    plt.imshow(translate_upward_image)

    plt.show()

    # Image process - Noise
    plt.figure(num='Image process - Noise', figsize=(15,3))
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(image.reshape(128,128,3))
    plt.axis('off')

    gaussian_noise = tf.random_normal(shape=(128,128,3), mean=0.0, stddev=0.1,
                                      dtype=tf.float32)
    gaussian_noise = tf.add(image.reshape(128,128,3), gaussian_noise)
    gaussian_noise_image = sess.run(gaussian_noise)
    plt.subplot(1,2,2)
    plt.title('Gaussian Noise')
    plt.imshow(gaussian_noise_image)
    plt.axis('off')

    plt.show()

    # Image process - Bright(亮度)
    plt.figure(num='Image process - Bright', figsize=(15,3))
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(image.reshape(128,128,3))
    plt.axis('off')

    brightness = tf.image.random_brightness(image.reshape(128,128,3),
                                                max_delta=128.0/255.0)
    brightness_image = sess.run(brightness)
    plt.subplot(1,2,2)
    plt.title('Bright')
    plt.imshow(brightness_image)
    plt.axis('off')

    plt.show()

    # Image process - Saturation(飽和)
    plt.figure(num='Image process - Saturation', figsize=(15,3))
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(image.reshape(128,128,3))
    plt.axis('off')

    saturation = tf.image.random_saturation(image.reshape(128,128,3), lower=0.5,
                                           upper=1.5)
    saturation_image = sess.run(saturation)
    plt.subplot(1,2,2)
    plt.title('Staturation')
    plt.imshow(saturation_image)
    plt.axis('off')

    plt.show()

    # Image process - Hue(調色)
    plt.figure(num='Image process - Hue', figsize=(15,3))
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(image.reshape(128,128,3))
    plt.axis('off')

    hue = tf.image.random_hue(image.reshape(128,128,3), max_delta=0.05)
    hue_image = sess.run(hue)
    plt.subplot(1,2,2)
    plt.title('Hue')
    plt.imshow(hue_image)
    plt.axis('off')

    plt.show()

    # Image process - Contrast(對比)
    plt.figure(num='Image process - Contrast', figsize=(15,3))
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(image.reshape(128,128,3))
    plt.axis('off')

    contrast = tf.image.random_contrast(image.reshape(128,128,3), lower=0.3,
                                        upper=1.0)
    contrast_image = sess.run(contrast)
    plt.subplot(1,2,2)
    plt.title('Contrast')
    plt.imshow(contrast_image)
    plt.axis('off')

    plt.show()

    # Image process - Contrast()
    plt.figure(num='Image process - Contrast', figsize=(15,3))
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(image.reshape(128,128,3))
    plt.axis('off')

    contrast = tf.image.random_contrast(image.reshape(128,128,3), lower=0.3,
                                        upper=1.0)
    contrast_image = sess.run(contrast)
    plt.subplot(1,2,2)
    plt.title('Contrast')
    plt.imshow(contrast_image)
    plt.axis('off')

    plt.show()

