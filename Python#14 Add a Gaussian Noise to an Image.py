import cv2
import numpy as np
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt

# original image
f = cv2.imread('Lenna.png', 0)
f = f/255 

cv2.imshow('original image', f)
cv2.waitKey(0)
cv2.destroyAllWindows()

# create gaussian noise
x, y = f.shape
mean = 0
var = 0.01
sigma = np.sqrt(var)
n = np.random.normal(loc=mean, 
                     scale=sigma, 
                     size=(x,y))

cv2.imshow('Gaussian noise', n)
cv2.waitKey(0)
cv2.destroyAllWindows()

# display the probability density function (pdf)
kde = gaussian_kde(n.reshape(int(x*y)))
dist_space = np.linspace(np.min(n), np.max(n), 100)
plt.plot(dist_space, kde(dist_space))
plt.xlabel('Noise pixel value'); plt.ylabel('Frequency')
plt.show()

# add a gaussian noise
g = f + n

cv2.imshow('Corrupted Image', g)
cv2.waitKey(0)
cv2.destroyAllWindows()

# display all
cv2.imshow('original image', f)
cv2.imshow('Gaussian noise', n)
cv2.imshow('Corrupted Image', g)

cv2.waitKey(0)
cv2.destroyAllWindows()
