import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.imread("../source/my.png")

res_one = img.copy()

src = np.float32([[155,16], [402, 110], [65, 504], [383, 509]])
dst = np.float32([[0,0], [300,00], [00,200], [300, 200]])
#  0 0  0,200
print(img.shape)

m = cv2.getPerspectiveTransform(src, dst)

print("wrap matrix")
print(m)

res = cv2.warpPerspective(res_one, m, (300, 200))

cv2.imshow("img", img)
cv2.imshow("res", res)
cv2.waitKey(0)