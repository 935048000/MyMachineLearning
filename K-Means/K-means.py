"""
k-均值是一种非常简单的聚类算法（集群属于无监督学习）。给定一个固定数量的集群和一个输入数据集，该算法尝试将数据划分为集群，这样集群具有较高的内部相似性和较低的类间相似性。

### Algorithm

1. 初始化集群中心，在输入数据的范围内或（建议）中随机地使用一些现有的培训示例

2. 直到收敛

   2.1.将每个数据点分配给最近的集群。用欧氏距离测量点和聚类中心之间的距离。

   2.2. 更新集群中心的当前估计，将其设置为属于该集群的所有实例的平均值
   
   
### Objective function

底层目标函数试图找到集群中心，如果数据被分区到相应的集群中，那么数据点和它们最接近的集群中心之间的距离就越小越好。


### Disadvantages of K-Means
- 集群的数量必须在开始时设置
- 结果取决于医院的群集中心
- 是敏感的离群值
- 它不适合寻找非凸群
- 它不能保证全局最优，因此它会被困在局部极小值中
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_blobs
np.random.seed(123)

X, y = make_blobs(centers=4, n_samples=1000)
print(f'Shape of dataset: {X.shape}')

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Dataset with 4 clusters")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()

class KMeans():
    def __init__(self, n_clusters=4):
        self.k = n_clusters

    def fit(self, data):
        """
        Fits the k-means model to the given dataset
        """
        n_samples, _ = data.shape
        # initialize cluster centers
        self.centers = np.array(random.sample(list(data), self.k))
        self.initial_centers = np.copy(self.centers)

        # We will keep track of whether the assignment of data points
        # to the clusters has changed. If it stops changing, we are
        # done fitting the model
        old_assigns = None
        n_iters = 0

        while True:
            new_assigns = [self.classify(datapoint) for datapoint in data]

            if new_assigns == old_assigns:
                print(f"Training finished after {n_iters} iterations!")
                return

            old_assigns = new_assigns
            n_iters += 1

            # recalculate centers
            for id_ in range(self.k):
                points_idx = np.where(np.array(new_assigns) == id_)
                datapoints = data[points_idx]
                self.centers[id_] = datapoints.mean(axis=0)

    def l2_distance(self, datapoint):
        dists = np.sqrt(np.sum((self.centers - datapoint)**2, axis=1))
        return dists

    def classify(self, datapoint):
        """
        Given a datapoint, compute the cluster closest to the
        datapoint. Return the cluster ID of that cluster.
        """
        dists = self.l2_distance(datapoint)
        return np.argmin(dists)

    def plot_clusters(self, data):
        plt.figure(figsize=(12,10))
        plt.title("Initial centers in black, final centers in red")
        plt.scatter(data[:, 0], data[:, 1], marker='.', c=y)
        plt.scatter(self.centers[:, 0], self.centers[:,1], c='r')
        plt.scatter(self.initial_centers[:, 0], self.initial_centers[:,1], c='k')
        plt.show()


kmeans = KMeans(n_clusters=4)
kmeans.fit(X)


kmeans.plot_clusters(X)








