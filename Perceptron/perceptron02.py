import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 自适应线性神经元与学习的融合。
# 在Python中实现一个自适应的线性神经元

class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


from Perceptron import perceptron01
X = perceptron01.X
y = perceptron01.y

plt.rcParams['font.sans-serif'] = ['SimHei']


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('迭代次数')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('学习率 = 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('迭代次数')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('学习率 = 0.0001')

# plt.tight_layout()
# plt.savefig('./adaline_1.png', dpi=300)
# plt.show()



# 标准化特征和再训练adaline
# 标准化的特征
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()


ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

perceptron01.plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - 梯度下降')
plt.xlabel('萼片长度 [standardized]')
plt.ylabel('花瓣长度 [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./adaline_2.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('迭代次数')
plt.ylabel('Sum-squared-error')

plt.tight_layout()
# plt.savefig('./adaline_3.png', dpi=300)
plt.show()


