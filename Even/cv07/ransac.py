import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def fit_line_least_squares(points):
    """使用最小二乘法拟合直线"""
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = LinearRegression().fit(X, y)
    return model.coef_[0], model.intercept_


def dist_to_line(line, points):
    """计算点到直线的距离"""
    m, c = line
    return np.abs(m * points[:, 0] - points[:, 1] + c) / np.sqrt(m ** 2 + 1)


def ransac(points, num_iterations=100, inlier_threshold=1.0):
    """RANSAC算法实现"""
    best_inliers_count = -1
    best_line = None

    for i in range(num_iterations):
        # 随机选择两个点
        rand_idxs = random.sample(range(len(points)), 2)
        sample_points = points[rand_idxs, :]

        # 拟合直线
        line = fit_line_least_squares(sample_points)

        # 计算所有点到直线的距离
        distances = dist_to_line(line, points)

        # 确定内点
        inliers = distances < inlier_threshold
        inliers_count = np.sum(inliers)

        # 更新最佳拟合直线
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_line = line

    return best_line, best_inliers_count


# 示例数据
np.random.seed(42)
points = np.random.rand(100, 2) * 10  # 100个随机点
# 添加一些异常值
points[-10:] += np.random.randn(10, 2) * 3

# 最小二乘法拟合
m_least_squares, c_least_squares = fit_line_least_squares(points)

# RANSAC拟合
best_line, inliers_count = ransac(points)
m_ransac, c_ransac = best_line

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(points[:, 0], points[:, 1], color='blue', label='Data Points')

# 绘制最小二乘法拟合的直线
x = np.linspace(0, 10, 100)
y_least_squares = m_least_squares * x + c_least_squares
plt.plot(x, y_least_squares, color='red', linewidth=2, label='Least Squares Line')

# 绘制RANSAC拟合的直线
y_ransac = m_ransac * x + c_ransac
plt.plot(x, y_ransac, color='green', linewidth=2, label='RANSAC Line')

plt.title('Comparison of Least Squares and RANSAC Line Fitting')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()