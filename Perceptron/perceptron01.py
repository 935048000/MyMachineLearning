import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """感知器分类器.

    参数
    ------------
    eta : float
        学习率 (between 0.0 and 1.0)
    n_iter : int
        通过训练数据集。

    属性
    -----------
    w_ : 1d-array
        拟合后的重量。
    errors_ : list
        每一个迭代的错误分类数目。

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """训练数据.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            训练向量，其中n_samples为样本数目。n_features是特性的数量。
        y : array-like, shape = [n_samples]
            目标值。

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """计算净输入"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """在单位步骤后返回类标签。"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# 绘制决策区域的函数。
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap (colors[:len (np.unique (y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min () - 1, X[:, 0].max () + 1
    x2_min, x2_max = X[:, 1].min () - 1, X[:, 1].max () + 1
    xx1, xx2 = np.meshgrid (np.arange (x1_min, x1_max, resolution),
                            np.arange (x2_min, x2_max, resolution))
    Z = classifier.predict (np.array ([xx1.ravel (), xx2.ravel ()]).T)
    Z = Z.reshape (xx1.shape)
    plt.contourf (xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim (xx1.min (), xx1.max ())
    plt.ylim (xx2.min (), xx2.max ())

    # plot class samples
    for idx, cl in enumerate (np.unique (y)):
        plt.scatter (x=X[y == cl, 0], y=X[y == cl, 1],
                     alpha=0.8, c=cmap (idx),
                     marker=markers[idx], label=cl)

# 取数据
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
# print(df.tail())
# exit(1)


# 挑选 setosa 和 versicolor 类的数据
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# 提取萼片长度和花瓣长度。
X = df.iloc[0:100, [0, 2]].values

if __name__ == '__main__':
    
    # 绘图数据
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='blue', marker='x', label='versicolor')
    
    plt.xlabel('petal length [cm]')
    plt.ylabel('sepal length [cm]')
    plt.legend(loc='upper left')
    
    # plt.tight_layout()
    # plt.savefig('./iris_1.png', dpi=300)
    # plt.show()
    
    
    
    # 训练感知器模型
    ppn = Perceptron(eta=0.1, n_iter=10)
    #
    ppn.fit(X, y)
    #
    # plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    # plt.xlabel('Epochs')
    # plt.ylabel('error')
    
    # plt.tight_layout()
    # plt.savefig('./perceptron_1.png', dpi=300)
    # plt.show()
    
    
    # 绘制决策区域
    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    # plt.savefig('./perceptron_2.png', dpi=300)
    plt.show()