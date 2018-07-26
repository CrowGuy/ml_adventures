#!/usr/bin/env python
"""The basic display practicees by python matplotlib lib.
   author: randy xu
   date: 2018.07.18
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# Create an window which named 'birds', and set the size
plt.figure(num='birds', figsize=(8,8))

# Use `mpimg` to read the image
image_owl = mpimg.imread('images/owl.jpg')
plt.subplot(2,2,1)
plt.title('owl')
plt.imshow(image_owl)

# Use `mpimg` to read the image and set color as gray
image_owl_gray = mpimg.imread('images/owl.jpg')
plt.subplot(2,2,3)
plt.title('owl as gray')
plt.imshow(image_owl_gray, plt.cm.gray)

# Use `opencv` to read the image
image_owl_cv = cv2.imread('images/owl.jpg')
plt.subplot(2,2,2)
plt.title('owl read from opencv')
plt.imshow(image_owl_cv)
plt.axis('off')    # disable axis

# Use `opencv` to read the image and set color to RGB
image_owl_cv_ = cv2.imread('images/owl.jpg')
plt.subplot(2,2,4)
plt.title('owl read from opencv and set color')
plt.imshow(cv2.cvtColor(image_owl_cv_, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
