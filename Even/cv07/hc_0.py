import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

# 生成样本数据
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.60, random_state=0)

# 执行层次聚类
cluster = AgglomerativeClustering(n_clusters=3)
cluster.fit(X)

# 获取聚类标签
labels = cluster.labels_

# 绘制样本数据点
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

# 绘制树状图
# 首先计算样本之间的距离
distances = pairwise_distances(X, metric='euclidean')
# 然后使用scipy.cluster.hierarchy.linkage来构建树状图
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(distances, 'ward')

# 绘制树状图
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
dendrogram(Z)
plt.show()