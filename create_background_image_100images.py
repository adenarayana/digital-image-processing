# import the library
import cv2
import numpy as np
from matplotlib import pyplot as plt

# read a video file and create a matrix A
video = cv2.VideoCapture('problem_1_surveillance.mp4')

matrixA = []

for i in range(100):
    ret, image = video.read() # read each image from video 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
    matrixA.append(image) # append the image into matrix A

matrixA = np.array(matrixA) # convert to numpy array

video.release() # release the video object

# display image at time t
plt.imshow(matrixA[50,:,:], cmap='gray')
plt.axis('off')
plt.show()

# create the background image b
b = np.median(matrixA,axis=0)
b = b.astype(np.uint8)

# showing the result
plt.imshow(b, cmap='gray')
plt.axis('off')
plt.show()