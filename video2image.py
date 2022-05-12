# Extract image from video file

import cv2

video = cv2.VideoCapture('problem_1_surveillance.mp4')

i = 1
while True:
    ret, image = video.read()
    
    if ret == False:
        break
    
    cv2.imwrite('Frames/image' + str(i) +'.jpg', image)
    
    i += 1
    
video.release()
