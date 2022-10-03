"""
@author: Made Python
"""
import numpy as np
import cv2
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

# f(x,y), input image in spatial domain
f = cv2.imread('License Plate.jpg', 0)

f = f/f.max() # normalize

# F(u,v), image in frequency domain
F = np.fft.fft2(f)

plt.imshow(np.log1p(np.abs(F)), cmap='gray')
plt.axis('off')
plt.show()

# H(u,v), motion blur function in frequency domain
# Create matrix H (motion blur function H(u,v))
M,N = F.shape 
H = np.zeros((M+1,N+1), dtype=np.complex128) # +1 to avoid zero division

# Motion blur parameters
T = 0.5 # duration of exposure
a = 0.05 # vertical motion
b = 0 # horizontal motion

# Fill matrix H
for u in range(1,M+1):
    for v in range(1,N+1):
        s = np.pi*(u*a + v*b)
        H[u,v] = (T/s) * np.sin(s) * np.exp(-1j*s)

# index slicing
H = H[1:,1:]     

plt.imshow(np.log1p(np.abs(H)), cmap='gray')
plt.axis('off')
plt.show()
   
# G(u,v), blurred image in frequency domain
G = H * F

plt.imshow(np.log1p(np.abs(G)), cmap='gray')
plt.axis('off')
plt.show()

# g(x,y), blurred image in spatial domain
g = np.fft.ifft2(G)
g = np.abs(g)

g = img_as_ubyte(g) # if you want to save the image

# display the result
cv2.imshow('original', f)
cv2.imshow('blur', g)
cv2.waitKey(0)
cv2.destroyAllWindows()