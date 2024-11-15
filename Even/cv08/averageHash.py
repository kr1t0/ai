from PIL import Image
import numpy as np


def average_hash(image_path, hash_size=8):
    # 打开图像并转换为灰度图
    img = Image.open(image_path).convert('L')

    # 缩放图像
    img = img.resize((hash_size, hash_size), Image.ANTIALIAS)

    # 将图像转换为numpy数组
    img_array = np.array(img)

    # 计算平均值
    mean_val = np.mean(img_array)

    # 生成哈希值
    hash_str = ''
    for i in range(hash_size):
        for j in range(hash_size):
            if img_array[i, j] > mean_val:
                hash_str += '1'
            else:
                hash_str += '0'

    return hash_str


# 使用示例
hash_value = average_hash('../source/my.png')
print(hash_value)
hash_value = average_hash('../source/left.jpg')
print(hash_value)