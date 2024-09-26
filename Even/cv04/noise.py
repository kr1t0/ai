import cv2
from skimage import util


img = cv2.imread("../source/my.png", 0)

noise_gs_img = util.random_noise(img, mode = "poisson")

cv2.imgshow("source", img)

cv2.imgshow("noise", noise_gs_img)

cv2.waitKey(0)
