import cv2
import numpy as np
import matplotlib.pyplot as plt

# input image f(x,y)
f = cv2.imread('Lenna.png', 0)
f = f / 255

plt.imshow(f, cmap='gray'); plt.axis('off'); plt.show()

# blur image
f_blur = cv2.GaussianBlur(src=f, 
                          ksize=(31,31), 
                          sigmaX=0, 
                          sigmaY=0)

plt.imshow(f_blur, cmap='gray'); plt.axis('off'); plt.show()

# mask
g_mask = f - f_blur
plt.imshow(g_mask, cmap='gray'); plt.axis('off'); plt.show()

# unsharp masking
k = 5
g = f + k*g_mask
plt.imshow(g, cmap='gray'); plt.axis('off'); plt.show()

g = np.clip(g, 0, 1)
plt.imshow(g, cmap='gray'); plt.axis('off'); plt.show()