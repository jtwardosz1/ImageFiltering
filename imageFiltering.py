


import numpy as np
import cv2 
import math
import random
from matplotlib import pyplot as plt
from scipy.signal import convolve2d


#resize and create grayscale image

#reads image
im=cv2.imread('bearsnacks.jpg')

#creates 2 separate grayscale images
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

#all for resizing image
scale_percent = 10
width1 = int(img.shape[1] * scale_percent / 100)
height1 = int(img.shape[0] * scale_percent / 100)
dim = (width1, height1)

img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

plt.figure()
plt.title('Resized/Gray image')
plt.imshow(img,cmap='gray', vmin=0, vmax=255) 


#average filter

#create 5x5 kernel
avgKern =np.zeros(shape=(5, 5))

#fill kernel with 1/25 value
avgKern.fill(1/25)

average = cv2.filter2D(img,-1,avgKern)    

plt.figure()
plt.title('Average Filter')
plt.imshow(average,cmap='gray', vmin=0, vmax=255)  


#sobel filter

#sobel kernels
sobel_vert = np.array([
         [-1.0, 0.0, 1.0]
        ,[-2.0, 0.0, 2.0]
        ,[-1.0, 0.0, 1.0]
        ])
sobel_horiz = np.array([
         [-1.0, -2.0, -1.0]
        ,[0.0, 0.0, 0.0]
        ,[1.0, 2.0, 1.0]
        ])

edge_horiz = convolve2d(img, sobel_horiz, mode='same', boundary = 'symm', fillvalue=0)

edge_vert = convolve2d(img, sobel_vert, mode='same', boundary = 'symm', fillvalue=0)

#create gradient by taking square root of the vertical^2 + horizontal^2
gradient = np.sqrt(np.square(edge_horiz) + np.square(edge_vert))

gradient *= 255.0 / np.max(gradient)

plt.figure()
plt.title('Vertical Edge Detection')
plt.imshow(edge_vert,cmap='gray', vmin=0, vmax=255) 

plt.figure()
plt.title('Horizontal Edge Detection')
plt.imshow(edge_horiz,cmap='gray', vmin=0, vmax=255) 

plt.figure()
plt.title('Gradient Edge Detection')
plt.imshow(gradient,cmap='gray', vmin=0, vmax=255) 


#laplacian filter

#laplacian kernel
l_kern1 = np.array([
         [1.0,  1.0, 1.0]
        ,[1.0, -8.0, 1.0]
        ,[1.0,  1.0, 1.0]
        ])

#laplacian kernel with center positive
l_kern2 = np.array([
         [-1.0,  -1.0, -1.0]
        ,[-1.0, 8.0, -1.0]
        ,[-1.0,  -1.0, -1.0]
        ])


l_Edge1 = cv2.filter2D(img, -1, l_kern1) 

l_Edge2 = cv2.filter2D(img, -1, l_kern2) 

plt.figure()
plt.title('Laplacian Edge Detection with diagonals')
plt.imshow(l_Edge1,cmap='gray', vmin=0, vmax=255) 

plt.figure()
plt.title('Laplacian Edge Detection with diagonals inverted center')
plt.imshow(l_Edge2,cmap='gray', vmin=0, vmax=255) 


#median filter

#create shape
height,width = np.shape(img)
median = np.zeros((height,width),dtype=float)
for i in range(2,height-4):
    for j in range(2,width-4):
        sorted_pixels = sorted(np.ndarray.flatten(img[i-2:i+4,j-2:j+4]))
        median[i][j] = sorted_pixels[5]
        
plt.figure()
plt.title('Median filter')
plt.imshow(median,cmap='gray', vmin=0, vmax=255) 


#guassian filter

#gaussian kernel
G_ker=np.array([
    [1, 4, 7, 4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]])

#divide kernel by the sum of the kernel
G_ker =G_ker/np.sum(G_ker)

Gaussian = cv2.filter2D(img, -1, G_ker) 

plt.figure()
plt.title('Gaussian blur')
plt.imshow(Gaussian, cmap='gray', vmin=0, vmax=255) 


#prewitt edge detection

#prewitt kernels
prewitt_vert = np.array([
         [-1.0, 0.0, 1.0]
        ,[-1.0, 0.0, 1.0]
        ,[-1.0, 0.0, 1.0]
        ])
prewitt_horiz = np.array([
         [-1.0, -1.0, -1.0]
        ,[0.0, 0.0, 0.0]
        ,[1.0, 1.0, 1.0]
        ])

pre_horiz = convolve2d(img, prewitt_horiz, mode='same', boundary = 'symm', fillvalue=0)

pre_vert = convolve2d(img, prewitt_vert, mode='same', boundary = 'symm', fillvalue=0)

#create gradient by taking square root of the vertical^2 + horizontal^2
gradient2 = np.sqrt(np.square(pre_horiz) + np.square(pre_vert))

gradient2 *= 255.0 / np.max(gradient2)

plt.figure()
plt.title('Prewitt Vertical Edge Detection')
plt.imshow(pre_vert,cmap='gray', vmin=0, vmax=255) 

plt.figure()
plt.title('Prewitt Horizontal Edge Detection')
plt.imshow(pre_horiz,cmap='gray', vmin=0, vmax=255) 

plt.figure()
plt.title('Prewitt Gradient Edge Detection')
plt.imshow(gradient2,cmap='gray', vmin=0, vmax=255) 


#laplacian of Gaussian

#log kernel
log_kern = np.array([
        [0, 0, 1, 2, 2, 2, 1, 0, 0],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [1, 3, 5, 3, 0, 3, 5, 3, 1],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [2, 5, 0, -23, -40, -23, 0, 5, 2],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [1, 3, 5, 3, 0, 3, 5, 3, 1],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [0, 0, 1, 2, 2, 2, 1, 0, 0]])

log_edge = cv2.filter2D(img3, -1, log_kern)
log_edge = np.absolute(log_edge)


plt.figure()
plt.title('Laplacian of Gaussian Edge Detection')
plt.imshow(log_edge,cmap='gray', vmin=0, vmax=255)

plt.show()
