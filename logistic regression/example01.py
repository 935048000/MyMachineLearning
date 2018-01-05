#Import Library
from sklearn.linear_model import LogisticRegression
# 假设您有X(预测器)和Y(目标)用于训练数据集和x_test(预测器)的test_dataset
# 创建逻辑回归对象
model = LogisticRegression()

# 使用训练集和检查分数训练模型
model.fit(X, y)
#model.score(X, y)

#方程系数和截距
print('系数:', model.coef_)
print('截距:', model.intercept_)

# 预测输出
predicted= model.predict(x_test)
print("预测:",predicted)


"""
优化方法：
    加入交互项
    精简模型特性
    使用正则化方法
    使用非线性模型

"""