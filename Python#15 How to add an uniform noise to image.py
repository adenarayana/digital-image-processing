'''
Made Python: https://www.youtube.com/channel/UCfoRD9cQgM8toJ375urVOpQ
'''

# libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

# orginal image
img = cv2.imread('Lenna.png',0)
img = img/255

cv2.imshow('original image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# uniform noise
x, y = img.shape
a = 0
b = 0.2
n = np.zeros((x,y), dtype=np.float64)
for i in range(x):
    for j in range(y):
        n[i][j] = np.random.uniform(a,b)

cv2.imshow('noise', n)
cv2.waitKey(0)
cv2.destroyAllWindows()
       
# add noise to image
noise_img = img + n
noise_img = np.clip(noise_img, 0, 1)

cv2.imshow('image with noise', noise_img)
cv2.imshow('original image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()     

# hitogram: original image
plt.hist(img.flat)
plt.xlim([0,1]); plt.ylim([0,60000])
plt.xlabel('pixel value'); plt.ylabel('frequency')
plt.show() 

# histogram: noise
plt.hist(n.flat)
plt.xlim([0,1]); plt.ylim([0,60000])
plt.xlabel('noise pixel value'); plt.ylabel('frequency')
plt.show()

# hitogram: noise image
plt.hist(noise_img.flat)
plt.xlim([0,1]); plt.ylim([0,60000])
plt.xlabel('pixel value'); plt.ylabel('frequency')
plt.show()  

# save the image
cv2.imwrite('uniform noise.jpg', img_as_ubyte(n)) #uint8
cv2.imwrite('noise image.jpg', img_as_ubyte(noise_img))
cv2.imwrite('lenna grayscale.jpg', img_as_ubyte(img))
