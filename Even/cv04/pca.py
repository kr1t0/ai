import numpy as np
import matplotlib.pyplot as plt


# 生成随机数据
def generate_data(n_samples, n_features):
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    return X


# 计算数据的协方差矩阵
def covariance_matrix(X):
    mean_X = np.mean(X, axis=0)
    X_adjusted = X - mean_X
    cov_matrix = np.zeros((X_adjusted.shape[1], X_adjusted.shape[1]), dtype=np.float64)

    # 计算协方差矩阵
    for i in range(X_adjusted.shape[1]):
        for j in range(X_adjusted.shape[1]):
            cov_matrix[i, j] = np.sum((X_adjusted[:, i] - X_adjusted[:, i].mean()) * (X_adjusted[:, j] - X_adjusted[:, j].mean())) / (
                        X_adjusted.shape[0] - 1)
    return cov_matrix


# 计算协方差矩阵的特征值和特征向量
def pca(X):
    cov_matrix = covariance_matrix(X)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


# 可视化数据
def visualize_data(X, eigenvalues, eigenvectors, n_components=2):
    # 计算主成分得分
    Z = np.dot(X, eigenvectors[:, :n_components])

    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c='b')
    plt.colorbar()
    plt.show()


# 主程序
if __name__ == '__main__':
    X = generate_data(100, 10)
    eigenvalues, eigenvectors = pca(X)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:", eigenvectors)
    visualize_data(X, eigenvalues, eigenvectors)