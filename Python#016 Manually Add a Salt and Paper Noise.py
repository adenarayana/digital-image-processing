'''
Made Python: https://www.youtube.com/channel/UCfoRD9cQgM8toJ375urVOpQ
'''

# libraries
import cv2
import numpy as np

# orginal image
img = cv2.imread('Lenna.png',0)
img = img/255

cv2.imshow('original image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# blank image
x,y = img.shape
g = np.zeros((x,y), dtype=np.float32)

# salt and pepper amount
pepper = 0.1
salt = 0.95

# create salt and peper noise image    
for i in range(x):
    for j in range(y):
        rdn = np.random.random()
        if rdn < pepper:
            g[i][j] = 0
        elif rdn > salt:
            g[i][j] = 1
        else:
            g[i][j] = img[i][j]

cv2.imshow('image with noise', g)
cv2.waitKey(0)
cv2.destroyAllWindows()

# (optional) save the image
from skimage import img_as_ubyte
cv2.imwrite('5percent.jpg', img_as_ubyte(g))