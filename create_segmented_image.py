# import the library
import cv2
import numpy as np
from matplotlib import pyplot as plt

# read a video file and create a matrix A
video = cv2.VideoCapture('problem_1_surveillance.mp4')

matrixA = []

while True:
    ret, image = video.read()  # read each image from video 
    if ret == False:           # break the while loop
        break
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
    matrixA.append(image)      # append to matrixA

matrixA = np.array(matrixA)    # convert to numpy array

video.release()

# create the background image b
b = np.median(matrixA,axis=0) # calculate the median
b = b.astype(np.uint8) # change the data type

plt.imshow(b, cmap='gray')
plt.axis('off')
plt.show()

# select any image
t = 0

plt.imshow(matrixA[t,:,:], cmap='gray')
plt.axis('off')
plt.show()

# image subtraction (absolute)
g = cv2.absdiff(matrixA[t,:,:], b)

plt.imshow(g, cmap='gray')
plt.axis('off')
plt.show()

# image thresholding
_, gT = cv2.threshold(g, 25, 255, cv2.THRESH_BINARY)

plt.imshow(gT, cmap='gray')
plt.axis('off')
plt.show()




