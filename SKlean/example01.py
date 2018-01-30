
# ## 加载和预处理数据。
# 从scikit-learn加载Iris数据集。
# 第三列表示花瓣长度, 第四列是花卉样品的花瓣宽度。
#  类已经转换为整数标签。 where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.

from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier



iris = datasets.load_iris ()
X = iris.data[:, [2, 3]]
y = iris.target
print ('类标签:', np.unique (y))

# 将数据分为70%的培训和30%的测试数据:
X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.3, random_state=0)

# 标准化的特点:
sc = StandardScaler ()
sc.fit (X_train)
X_train_std = sc.transform (X_train)
X_test_std = sc.transform (X_test)


# ##通过scikit-learn训练一个感知器。
# 重新定义`plot_decision_region` function:

def _perceptron():
    ppn = Perceptron (max_iter=40, eta0=0.1, random_state=0)
    
    print(ppn.fit (X_train_std, y_train))
    print(y_test.shape)
    
    y_pred = ppn.predict (X_test_std)
    print ('分类错误的样本: %d' % (y_test != y_pred).sum ())
    print ('精确度: %.2f' % accuracy_score (y_test, y_pred))

# _perceptron()



# matplotlib 图像输出中文
plt.rcParams['font.sans-serif'] = ['SimHei']

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # 设置标记生成器和颜色映射。
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap (colors[:len (np.unique (y))])
    
    # 绘制决策表面
    x1_min, x1_max = X[:, 0].min () - 1, X[:, 0].max () + 1
    x2_min, x2_max = X[:, 1].min () - 1, X[:, 1].max () + 1
    xx1, xx2 = np.meshgrid (np.arange (x1_min, x1_max, resolution),
                            np.arange (x2_min, x2_max, resolution))
    Z = classifier.predict (np.array ([xx1.ravel (), xx2.ravel ()]).T)
    Z = Z.reshape (xx1.shape)
    plt.contourf (xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim (xx1.min (), xx1.max ())
    plt.ylim (xx2.min (), xx2.max ())
    
    # 绘制所有样品
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate (np.unique (y)):
        plt.scatter (x=X[y == cl, 0], y=X[y == cl, 1],
                     alpha=0.8, c=cmap (idx),
                     marker=markers[idx], label=cl)
    
    # 强调测试样品
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter (X_test[:, 0], X_test[:, 1], c='',
                     alpha=1.0, linewidth=1, marker='o',
                     s=55, label='test set')


# 使用标准化训练数据训练感知器模型:
X_combined_std = np.vstack ((X_train_std, X_test_std))
y_combined = np.hstack ((y_train, y_test))


def perceptron():

    plot_decision_regions (X=X_combined_std, y=y_combined,
                           classifier=ppn, test_idx=range (105, 150))
    plt.xlabel ('花瓣长度 [标准化]')
    plt.ylabel ('花瓣宽度 [标准化]')
    plt.legend (loc='upper left')
    
    plt.tight_layout ()
    # plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)
    plt.show ()

# perceptron()

# --------------------------------------------------------------------------

# 通过逻辑回归建模类概率。
# 绘制 sigmoid 函数
# import matplotlib.pyplot as plt
# import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp (-z))

def PlotSigmoid():
    
    z = np.arange (-7, 7, 0.1)
    phi_z = sigmoid (z)
    
    plt.plot (z, phi_z)
    plt.axvline (0.0, color='k')
    plt.ylim (-0.1, 1.1)
    plt.xlabel ('z')
    plt.ylabel ('$\phi (z)$')
    
    # y轴和网格线。
    plt.yticks ([0.0, 0.5, 1.0])
    ax = plt.gca ()
    ax.yaxis.grid (True)
    
    plt.tight_layout ()
    # plt.savefig('./figures/sigmoid.png', dpi=300)
    plt.show ()
    
# PlotSigmoid()

# 绘制损失函数
def cost_1(z):
    return - np.log (sigmoid (z))

def cost_0(z):
    return - np.log (1 - sigmoid (z))

def PlotCostFunc():
    z = np.arange (-10, 10, 0.1)
    phi_z = sigmoid (z)
    
    c1 = [cost_1 (x) for x in z]
    plt.plot (phi_z, c1, label='J(w) if y=1')
    
    c0 = [cost_0 (x) for x in z]
    plt.plot (phi_z, c0, linestyle='--', label='J(w) if y=0')
    
    plt.ylim (0.0, 5.1)
    plt.xlim ([0, 1])
    plt.xlabel ('$\phi$(z)')
    plt.ylabel ('J(w)')
    plt.legend (loc='best')
    plt.tight_layout ()
    # plt.savefig('./figures/log_cost.png', dpi=300)
    plt.show ()

# PlotCostFunc()



def tLogisticRegression():
    
    lr = LogisticRegression (C=1000.0, random_state=0)
    lr.fit (X_train_std, y_train)
    
    plot_decision_regions (X_combined_std, y_combined,
                           classifier=lr, test_idx=range (105, 150))
    plt.xlabel ('花瓣长度 [标准化]')
    plt.ylabel ('花瓣宽度 [标准化]')
    plt.legend (loc='upper left')
    plt.tight_layout ()
    # plt.savefig('./figures/logistic_regression.png', dpi=300)
    plt.show ()

# tLogisticRegression()


# lr.predict_proba (X_test_std[0, :])

# 正则化路径:
def Regularization():
    weights, params = [], []
    for c in np.arange (-5, 5,dtype  = float):
        lr = LogisticRegression (C=10 ** c, random_state=0)
        lr.fit (X_train_std, y_train)
        weights.append (lr.coef_[1])
        params.append (10 ** c)
    
    weights = np.array (weights)
    plt.plot (params, weights[:, 0],
              label='petal length')
    plt.plot (params, weights[:, 1], linestyle='--',
              label='petal width')
    plt.ylabel ('weight coefficient')
    plt.xlabel ('C')
    plt.legend (loc='upper left')
    plt.xscale ('log')
    # plt.savefig('./figures/regression_path.png', dpi=300)
    plt.show ()

# Regularization()


# #支持向量机的最大边界分类。
def SVM():
    svm = SVC (kernel='linear', C=1.0, random_state=0)
    svm.fit (X_train_std, y_train)
    
    plot_decision_regions (X_combined_std, y_combined,
                           classifier=svm, test_idx=range (105, 150))
    plt.xlabel ('petal length [standardized]')
    plt.ylabel ('petal width [standardized]')
    plt.legend (loc='upper left')
    plt.tight_layout ()
    # plt.savefig('./figures/support_vector_machine_linear.png', dpi=300)
    plt.show ()

# SVM()



# # 使用核支持向量机解决非线性问题。
def testKSVM():
    
    np.random.seed (0)
    X_xor = np.random.randn (200, 2)
    y_xor = np.logical_xor (X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where (y_xor, 1, -1)
    
    plt.scatter (X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
    plt.scatter (X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='-1')
    
    plt.xlim ([-3, 3])
    plt.ylim ([-3, 3])
    plt.legend (loc='best')
    plt.tight_layout ()
    # plt.savefig('./figures/xor.png', dpi=300)
    plt.show ()

    svm = SVC (kernel='rbf', random_state=0, gamma=0.10, C=10.0)
    svm.fit (X_xor, y_xor)
    plot_decision_regions (X_xor, y_xor,
                           classifier=svm)

    plt.legend (loc='upper left')
    plt.tight_layout ()
    # plt.savefig('./figures/support_vector_machine_rbf_xor.png', dpi=300)
    plt.show ()
    

# testKSVM()









def KSVM():
    svm = SVC (kernel='rbf', random_state=0, gamma=0.2, C=1.0)
    svm.fit (X_train_std, y_train)
    
    plot_decision_regions (X_combined_std, y_combined,
                           classifier=svm, test_idx=range (105, 150))
    plt.xlabel ('花瓣长度 [标准化]')
    plt.ylabel ('花瓣宽度 [标准化]')
    plt.legend (loc='upper left')
    plt.tight_layout ()
    # plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
    plt.show ()


# KSVM()



def KSVM_MaxGamma():
    svm = SVC (kernel='rbf', random_state=0, gamma=100.0, C=1.0)
    svm.fit (X_train_std, y_train)
    
    plot_decision_regions (X_combined_std, y_combined,
                           classifier=svm, test_idx=range (105, 150))
    plt.xlabel ('花瓣长度 [标准化]')
    plt.ylabel ('花瓣宽度 [标准化]')
    plt.legend (loc='upper left')
    plt.tight_layout ()
    # plt.savefig('./figures/support_vector_machine_rbf_iris_2.png', dpi=300)
    plt.show ()

# KSVM_MaxGamma()



# # 决策树学习

def DecisionTrees():
    tree = DecisionTreeClassifier (criterion='entropy', max_depth=3, random_state=0)
    tree.fit (X_train, y_train)
    
    X_combined = np.vstack ((X_train, X_test))
    y_combined = np.hstack ((y_train, y_test))
    plot_decision_regions (X_combined, y_combined,
                           classifier=tree, test_idx=range (105, 150))
    
    plt.xlabel ('petal length [cm]')
    plt.ylabel ('petal width [cm]')
    plt.legend (loc='upper left')
    plt.tight_layout ()
    # plt.savefig('./figures/decision_tree_decision.png', dpi=300)
    plt.show ()

# DecisionTrees()


# 基尼系数
def gini(p):
    return (p) * (1 - (p)) + (1 - p) * (1 - (1 - p))

# 熵
def entropy(p):
    return - p * np.log2 (p) - (1 - p) * np.log2 ((1 - p))

# 误差
def error(p):
    return 1 - np.max ([p, 1 - p])

def PlotHEE():
    x = np.arange (0.0, 1.0, 0.01)
    
    ent = [entropy (p) if p != 0 else None for p in x]
    sc_ent = [e * 0.5 if e else None for e in ent]
    err = [error (i) for i in x]
    
    fig = plt.figure ()
    ax = plt.subplot (111)
    for i, lab, ls, c, in zip ([ent, sc_ent, gini (x), err],
                               ['Entropy', 'Entropy (scaled)',
                                'Gini Impurity', 'Misclassification Error'],
                               ['-', '-', '--', '-.'],
                               ['black', 'lightgray', 'red', 'green', 'cyan']):
        line = ax.plot (x, i, label=lab, linestyle=ls, lw=2, color=c)
    
    ax.legend (loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=False)
    
    ax.axhline (y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline (y=1.0, linewidth=1, color='k', linestyle='--')
    plt.ylim ([0, 1.1])
    plt.xlabel ('p(i=1)')
    plt.ylabel ('Impurity Index')
    plt.tight_layout ()
    # plt.savefig ('./figures/impurity.png', dpi=300, bbox_inches='tight')
    plt.show ()

# PlotHEE()


def randomForests():
    tree = DecisionTreeClassifier (criterion='entropy', max_depth=3, random_state=0)
    tree.fit (X_train, y_train)
    export_graphviz (tree,
                     out_file='tree.dot',
                     feature_names=['petal length', 'petal width'])

    X_combined = np.vstack ((X_train, X_test))
    y_combined = np.hstack ((y_train, y_test))
    # # 通过随机森林将弱者与强者结合。
    
    forest = RandomForestClassifier (criterion='entropy',
                                     n_estimators=10,
                                     random_state=1,
                                     n_jobs=2)
    forest.fit (X_train, y_train)
    
    plot_decision_regions (X_combined, y_combined,
                           classifier=forest, test_idx=range (105, 150))
    
    plt.xlabel ('花瓣长度(厘米)')
    plt.ylabel ('花瓣宽(厘米)')
    plt.legend (loc='upper left')
    plt.tight_layout ()
    # plt.savefig('./figures/random_forest.png', dpi=300)
    plt.show ()

# randomForests()


#k最近的邻居-一个懒惰的学习算法。
def KNN():
    knn = KNeighborsClassifier (n_neighbors=5, p=2, metric='minkowski')
    knn.fit (X_train_std, y_train)
    
    plot_decision_regions (X_combined_std, y_combined,
                           classifier=knn, test_idx=range (105, 150))
    
    plt.xlabel ('petal length [standardized]')
    plt.ylabel ('petal width [standardized]')
    plt.legend (loc='upper left')
    plt.tight_layout ()
    # plt.savefig('./figures/k_nearest_neighbors.png', dpi=300)
    plt.show ()

# KNN()
