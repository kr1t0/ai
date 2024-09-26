import random
import cv2
import numpy as np

def PepperSaltNoise(src, means, sigma, percetage):

    NoiseImg = src
    NoiseNum= int (percetage * src.shape[0] * src.shape[1])

    for i in range(NoiseNum):

        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[0] - 1)


        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255


    return NoiseImg

img = cv2.imread("../source/my.png", 0)

img1 = PepperSaltNoise(img, 2, 4, 0.01)

img = cv2.imread("../source/my.png")

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cv2.imshow("gauss",img1)
cv2.imshow("src",img2)
cv2.waitKey(0)