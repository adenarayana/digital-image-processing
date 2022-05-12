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

# video segmentation
video = cv2.VideoCapture('problem_1_surveillance.mp4')
while True:
    ret, image = video.read()  
    if ret == False:           
        break
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    g = cv2.absdiff(image, b) # image differece
    _, gT = cv2.threshold(g, 25, 255, cv2.THRESH_BINARY) # image thresholding
    cv2.imshow('Image Subtraction', gT)
    if cv2.waitKey(30) & 0xFFF == 27:
        break

video.release()
cv2.destroyAllWindows()