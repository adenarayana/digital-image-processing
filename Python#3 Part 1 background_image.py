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

# display image at time t
t = 10
plt.imshow(matrixA[t,:,:], cmap='gray')
plt.axis('off')
plt.show()

# create the background image b
b = np.median(matrixA,axis=0) # calculate the median
b = b.astype(np.uint8) # change the data type

# showing the result
plt.imshow(b, cmap='gray')
plt.axis('off')
plt.show()
