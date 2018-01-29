from numpy.random import seed
from Perceptron import perceptron01
from Perceptron import perceptron02
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 大型机器学习和随机梯度下降。
class AdalineSGD (object):
    """ADAptive LInear NEuron classifier.

    shuffle : bool (default: True)
        如果要防止循环的话，每个阶段都要训练数据。
    random_state : int (default: None)
        设置随机状态，以调整和初始化权重。

    """
    
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed (random_state)
    
    def fit(self, X, y):

        self._initialize_weights (X.shape[1])
        self.cost_ = []
        for i in range (self.n_iter):
            if self.shuffle:
                X, y = self._shuffle (X, y)
            cost = []
            for xi, target in zip (X, y):
                cost.append (self._update_weights (xi, target))
            avg_cost = sum (cost) / len (y)
            self.cost_.append (avg_cost)
        return self
    
    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights (X.shape[1])
        if y.ravel ().shape[0] > 1:
            for xi, target in zip (X, y):
                self._update_weights (xi, target)
        else:
            self._update_weights (X, y)
        return self
    
    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation (len (y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.w_ = np.zeros (1 + m)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input (xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot (error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot (X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return self.net_input (X)
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where (self.activation (X) >= 0.0, 1, -1)


ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(perceptron02.X_std, perceptron02.y)

perceptron01.plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('./adaline_4.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout()
# plt.savefig('./adaline_5.png', dpi=300)
plt.show()







