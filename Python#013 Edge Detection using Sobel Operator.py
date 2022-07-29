import numpy as np
import cv2
import matplotlib.pyplot as plt

# function for displaying image
def display(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    ax.axis('off')

# function for creating an image
def create_img():
    blank_img = np.zeros((200,200))
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img=blank_img, 
                text='H', 
                org=(50,150), 
                fontFace=font, 
                fontScale=5, 
                color=(255,255,255),
                thickness=25,
                lineType=cv2.LINE_AA)
    return blank_img

# sobel kernel
sobel_x = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

sobel_y = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

# input image
i = create_img()
display(i)

# partial derivative in x-direction
edge_x = cv2.filter2D(src=i, ddepth=-1, kernel=sobel_x)
display(edge_x)

edge_x[edge_x != 0] = 255
display(edge_x)

# partial derivative in y-direction
edge_y = cv2.filter2D(src=i, ddepth=-1, kernel=sobel_y)
display(edge_y)

edge_y[edge_y != 0] = 255
display(edge_y)

# combinte the x and y edge
add_edge = edge_x + edge_y
display(add_edge)

add_edge[add_edge != 0] = 255
display(add_edge)
