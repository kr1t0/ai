import random
import cv2
import numpy as np

def GaussianNoise(src, means, sigma, percetage):

    NoiseImg = src
    NoiseNum= int (percetage * src.shape[0] * src.shape[1])

    for i in range(NoiseNum):

        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[0] - 1)

        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)

        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY]  = 255


    return NoiseImg

img = cv2.imread("../source/my.png", 0)

img1 = GaussianNoise(img, 2 , 4, 0.8)

img = cv2.imread("../source/my.png")

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cv2.imshow("gauss",img1)
cv2.imshow("src",img2)
cv2.waitKey(0)