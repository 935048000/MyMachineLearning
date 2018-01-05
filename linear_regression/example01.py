from sklearn import linear_model


# 负载训练和测试数据集
# 标识特性和响应变量，值必须是数字和numpy数组
# x 训练集
input_variables_values_training_datasets = [[150.0], [200.0], [250.0], [300.0], [350.0], [400.0], [600.0]]
x_train = input_variables_values_training_datasets
# y 训练集
target_variables_values_training_datasets = [6450.0, 7450.0, 8450.0, 9450.0, 11450.0, 15450.0, 18450.0]
y_train = target_variables_values_training_datasets
# x测试集
x_test = 700

# 创建线性回归对象
linear = linear_model.LinearRegression()

# 使用训练集和检查分数训练模型
linear.fit(x_train, y_train)
# linear.score(x_train, y_train)

# 方程系数和截距
print('截距:', linear.intercept_)
print('系数:', linear.coef_)


# 预测输出
predicted= linear.predict(x_test)
print("预测值：",predicted)
"""
输出：
截距值  1771.80851064
系数 [ 28.77659574]
预测值:  [ 21915.42553191]
"""