"""
k-nn算法是一种简单的监督机器学习算法，可用于分类和回归。这是一个* *基于实例的* *算法。因此，它不需要估计模型，而是将所有训练样本存储在内存中，并使用相似度量来进行预测。

给定一个输入示例，k-nn算法从内存中检索k最相似的实例。相似性是在距离上定义的，也就是说，与输入示例之间最小的（欧氏）距离的训练示例被认为是最相似的。

输入示例的目标值如下：

Classification:
a) 未加权：在最近的邻居中输出最常见的分类
b) 加权：为每个分类值和最高权重的输出分类，总结最近的邻居的权重。

Regression:
a) 未加权：输出k最近邻居的值的平均值
b) 加权：对于所有的分类值，加起来的分类值$权重，并将结果槽除以所有权重的总和。

加权k-nn版本是算法的改进版本，其中每个邻居的贡献都根据其与查询点的距离进行加权。下面，我们将从sklearn中实现k-nn算法的基本无加权版本。

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
np.random.seed(123)

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

# Example digits
fig = plt.figure(figsize=(10,8))
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    plt.imshow(X[i].reshape((8,8)), cmap='gray')
    

class kNN():
    def __init__(self):
        pass

    def fit(self, X, y):
        self.data = X
        self.targets = y

    def euclidean_distance(self, X):
        """
        Computes the euclidean distance between the training data and
        a new input example or matrix of input examples X
        """
        # input: single data point
        if X.ndim == 1:
            l2 = np.sqrt(np.sum((self.data - X)**2, axis=1))

        # input: matrix of data points
        if X.ndim == 2:
            n_samples, _ = X.shape
            l2 = [np.sqrt(np.sum((self.data - X[i])**2, axis=1)) for i in range(n_samples)]

        return np.array(l2)

    def predict(self, X, k=1):
        """
        Predicts the classification for an input example or matrix of input examples X
        """
        # step 1: compute distance between input and training data
        dists = self.euclidean_distance(X)

        # step 2: find the k nearest neighbors and their classifications
        if X.ndim == 1:
            if k == 1:
                nn = np.argmin(dists)
                return self.targets[nn]
            else:
                knn = np.argsort(dists)[:k]
                y_knn = self.targets[knn]
                max_vote = max(y_knn, key=list(y_knn).count)
                return max_vote

        if X.ndim == 2:
            knn = np.argsort(dists)[:, :k]
            y_knn = self.targets[knn]
            if k == 1:
                return y_knn.T
            else:
                n_samples, _ = X.shape
                max_votes = [max(y_knn[i], key=list(y_knn[i]).count) for i in range(n_samples)]
                return max_votes

if __name__ == '__main__':
    
    
    knn = kNN()
    knn.fit(X_train, y_train)
    
    print("Testing one datapoint, k=1")
    print(f"Predicted label: {knn.predict(X_test[0], k=1)}")
    print(f"True label: {y_test[0]}")
    print()
    print("Testing one datapoint, k=5")
    print(f"Predicted label: {knn.predict(X_test[20], k=5)}")
    print(f"True label: {y_test[20]}")
    print()
    print("Testing 10 datapoint, k=1")
    print(f"Predicted labels: {knn.predict(X_test[5:15], k=1)}")
    print(f"True labels: {y_test[5:15]}")
    print()
    print("Testing 10 datapoint, k=4")
    print(f"Predicted labels: {knn.predict(X_test[5:15], k=4)}")
    print(f"True labels: {y_test[5:15]}")
    print()
    
    # Compute accuracy on test set
    y_p_test1 = knn.predict(X_test, k=1)
    test_acc1= np.sum(y_p_test1[0] == y_test)/len(y_p_test1[0]) * 100
    print(f"Test accuracy with k = 1: {format(test_acc1)}")
    
    y_p_test8 = knn.predict(X_test, k=5)
    test_acc8= np.sum(y_p_test8 == y_test)/len(y_p_test8) * 100
    print(f"Test accuracy with k = 8: {format(test_acc8)}")