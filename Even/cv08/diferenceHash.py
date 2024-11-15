from PIL import Image


def difference_hash(image_path, hash_size=8):
    # 打开图像并转换为灰度图
    img = Image.open(image_path).convert('L')

    # 缩放图像
    img = img.resize((hash_size + 1, hash_size), Image.LANCZOS)

    # 将图像转换为一维数组
    pixels = list(img.getdata())

    # 生成哈希值
    hash_str = ''
    for i in range(hash_size-1):
        for j in range(hash_size):
            if pixels[i * (hash_size + 1) + j] > pixels[(i + 1) * (hash_size + 1) + j]:
                hash_str += '1'
            else:
                hash_str += '0'

    return hash_str


# 使用示例
hash_value = difference_hash('../source/my.png')
print(hash_value)