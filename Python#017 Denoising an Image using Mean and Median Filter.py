"""
@author: Made Python
https://www.youtube.com/channel/UCfoRD9cQgM8toJ375urVOpQ/about
"""

import cv2
import numpy as np
from skimage import img_as_ubyte

## original image
img = cv2.imread('Lenna.png', 0)
img = img/img.max() # normalize the pixel value (0~1)

## noise image
# Gaussian Noise
# =============================================================================
# x, y = img.shape
# mean = 0
# var = 0.01
# sigma = np.sqrt(var)
# n = np.random.normal(loc=mean, 
#                      scale=sigma, 
#                      size=(x,y))
# img_noise = img + n
# =============================================================================

# Salt and Pepper Noise
x,y = img.shape
g = np.zeros((x,y), dtype=np.float32)
pepper = 0.1
salt = 0.95  
for i in range(x):
    for j in range(y):
        rdn = np.random.random()
        if rdn < pepper:
            g[i][j] = 0
        elif rdn > salt:
            g[i][j] = 1
        else:
            g[i][j] = img[i][j]

img_noise = g

# preview the images
cv2.imshow('Original Image', img)
cv2.imshow('Image + Noise', img_noise)

cv2.waitKey(0)
cv2.destroyAllWindows()

## denoise image
# mean filter (average)
m = 5
n = 5
denoise_mean = cv2.blur(img_noise, (m,n))

# median filter
img_noise_median = np.clip(img_noise, -1, 1) #pixel value range
img_noise_median = img_as_ubyte(img_noise_median) #convert to uint8
denoise_median = cv2.medianBlur(img_noise_median, 5)

# preview the images
cv2.imshow('Original Image', img)
cv2.imshow('Image + Noise', img_noise)
cv2.imshow('Denoise Mean', denoise_mean)
cv2.imshow('Denoise Median', denoise_median)

cv2.waitKey(0)
cv2.destroyAllWindows()

# (optional) save the result
cv2.imwrite('Denoise mean.jpg', img_as_ubyte(denoise_mean))
cv2.imwrite('Denoise median.jpg', img_as_ubyte(denoise_median))