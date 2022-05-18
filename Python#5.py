import cv2
import numpy as np
import matplotlib.pyplot as plt

# original image
original_image = cv2.imread('image.png', 0)

# blur and sharpen convolution kernel
M = 3
blur_kernel = np.ones((M,M)) * 1/(M*M)

sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], 
                          dtype=np.float32)

# apply the covolution
blur_image = cv2.filter2D(src=original_image, 
                          ddepth=-1, 
                          kernel=blur_kernel)

sharpen_image = cv2.filter2D(src=original_image, 
                             ddepth=-1, 
                             kernel=sharpen_kernel)

# display the result
plt.imshow(original_image, cmap='gray')
plt.axis('off')
plt.show()

plt.imshow(blur_image, cmap='gray')
plt.axis('off')
plt.show()

plt.imshow(sharpen_image, cmap='gray')
plt.axis('off')
plt.show()

# built in image bluring
builtin_blur = cv2.blur(src=original_image, ksize=(3,3))

plt.imshow(builtin_blur, cmap='gray')
plt.axis('off')
plt.show()

builtin_median = cv2.medianBlur(src=original_image, ksize=3)

plt.imshow(builtin_median, cmap='gray')
plt.axis('off')
plt.show()

builtin_gaussian = cv2.GaussianBlur(src=original_image, 
                                    ksize=(3,3), 
                                    sigmaX=0, 
                                    sigmaY=0)

plt.imshow(builtin_gaussian, cmap='gray')
plt.axis('off')
plt.show()
