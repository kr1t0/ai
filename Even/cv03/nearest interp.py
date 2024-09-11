import cv2
import numpy as np

def function(image, size):
    height, width ,channels = image.shape
    emptyImage = np.zeros((size,size,channels),np.uint8)

    sh = size / height
    sw = size/width

    for i in range(size):
        for j in range(size):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            emptyImage[i,j] = image[x,y]

    return emptyImage

def function2(image, size):
    height, width ,channels = image.shape
    emptyImage = np.zeros((size,size,channels),np.uint8)

    sh = size / height
    sw = size/width

    for i in range(size):
        for j in range(size):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            emptyImage[i,j] = image[x,y]

    return emptyImage

image = cv2.imread('my.png')
zoom = function(image, 25)
# cv2.resize(image, (800,800,c), near)
print(zoom)
print(zoom.shape)

cv2.imshow("nerest interp", zoom)
cv2.imshow("image", image)
cv2.waitKey(0)