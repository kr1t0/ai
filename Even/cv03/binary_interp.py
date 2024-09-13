import cv2
import numpy as np

def function(img, parm):
    src_h , src_w, channel = img.shape
    dst_h , dst_w = parm[0], parm[1]
    scale_x = float(dst_h) / src_h
    scale_y = float(dst_w) / src_w
    dst_img = np.zeros((dst_h, dst_w, channel), np.uint8)
    for i in range(channel):
        for dst_x in range(dst_h):
            for dst_y in range(dst_w):
                src_x = (dst_x + 0.5) / scale_x - 0.5
                src_y = (dst_y + 0.5) / scale_y - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_h - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_w - 1)

                temp0 = (src_x - src_x0) * img[src_x1][src_y0][i] + (src_x1-src_x) * img[src_x0][src_y0][i]
                temp1 = (src_x - src_x0) * img[src_x1][src_y1][i] + (src_x1-src_x) * img[src_x0][src_y1][i]

                dst_img[dst_x][dst_y][i] = int((src_y1-src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img

img = cv2.imread("my.png")

zoom = function(img, (300,600,3))

cv2.imshow("binary",zoom)
cv2.imshow("src",img)
cv2.waitKey(0)