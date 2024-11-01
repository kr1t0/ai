import cv2
import numpy as np

# 加载同一图像作为左右图像
img_left = cv2.imread('../source/left.jpg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('../source/right.jpg', cv2.IMREAD_GRAYSCALE)

# 创建立体匹配对象
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# 计算视差图
disparity = stereo.compute(img_left, img_right)

# 归一化视差图以便显示
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# 显示结果
cv2.imshow('Left Image', img_left)
cv2.imshow('Right Image', img_right)
cv2.imshow('Disparity', disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
